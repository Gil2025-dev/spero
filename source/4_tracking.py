"""
실시간 추론 + 객체 추적 프로그램
한 번 분류된 객체를 자동으로 추적합니다.
"""
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image

class RealtimeInferenceWithTracking:
    def __init__(self, model_path, img_size=224, use_depth=False):
        """
        Args:
            model_path: 학습된 모델 경로
            img_size: 입력 이미지 크기
            use_depth: Depth 정보 사용 여부
        """
        self.img_size = img_size
        self.use_depth = use_depth
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델 로드
        self.load_model(model_path)
        
        # 이미지 전처리 변환
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # ROI 선택 상태
        self.roi_start = None
        self.roi_end = None
        self.is_drawing = False
        self.roi_selected = False
        
        # 추적 상태
        self.template = None
        self.template_size = None
        self.last_bbox = None
        self.tracking = False
        self.tracked_class = None
        self.tracked_confidence = 0.0
        self.tracked_probabilities = None
        
        print(f"✓ 모델 로드 완료")
        print(f"  클래스: {self.class_names}")
        print(f"  디바이스: {self.device}")
    
    def load_model(self, model_path):
        """학습된 모델 로드"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 설정 및 클래스 정보
        self.class_names = checkpoint['class_names']
        self.num_classes = len(self.class_names)
        config = checkpoint['config']
        
        # 모델 생성
        model_name = config['model_name']
        
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=False)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, self.num_classes)
            
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=False)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, self.num_classes)
            
        elif model_name == 'mobilenet_v2':
            self.model = models.mobilenet_v2(pretrained=False)
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, self.num_classes)
        
        # 가중치 로드
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def mouse_callback(self, event, x, y, flags, param):
        """마우스 이벤트 처리"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_drawing = True
            self.roi_start = (x, y)
            self.roi_end = (x, y)
            self.roi_selected = False
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing:
                self.roi_end = (x, y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
            self.roi_end = (x, y)
            self.roi_selected = True
    
    def preprocess_roi(self, color_image, roi_coords):
        """ROI 전처리"""
        x1, y1, x2, y2 = roi_coords
        roi = color_image[y1:y2, x1:x2]
        
        # BGR to RGB
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # PIL Image로 변환
        roi_pil = Image.fromarray(roi_rgb)
        
        # Transform 적용
        roi_tensor = self.transform(roi_pil)
        roi_tensor = roi_tensor.unsqueeze(0)  # 배치 차원 추가
        
        return roi_tensor
    
    def predict(self, roi_tensor):
        """추론 수행"""
        with torch.no_grad():
            roi_tensor = roi_tensor.to(self.device)
            outputs = self.model(roi_tensor)
            
            # Softmax로 확률 계산
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = predicted.item()
            confidence_score = confidence.item()
            
            return predicted_class, confidence_score, probabilities[0].cpu().numpy()
    
    def init_tracker(self, frame, bbox):
        """트래커 초기화 - 템플릿 매칭 사용"""
        x, y, w, h = [int(v) for v in bbox]
        
        # 템플릿 저장 (추적할 영역)
        self.template = frame[y:y+h, x:x+w].copy()
        self.template_size = (w, h)
        self.last_bbox = bbox
        self.tracking = True
        
    def update_tracker(self, frame):
        """트래커 업데이트 - 템플릿 매칭으로 위치 찾기"""
        if self.template is None:
            return False, None
        
        # 이전 위치 주변에서 검색 (효율성)
        x, y, w, h = [int(v) for v in self.last_bbox]
        
        # 검색 영역 설정 (이전 위치 ±50 픽셀)
        search_margin = 50
        x1 = max(0, x - search_margin)
        y1 = max(0, y - search_margin)
        x2 = min(frame.shape[1], x + w + search_margin)
        y2 = min(frame.shape[0], y + h + search_margin)
        
        search_region = frame[y1:y2, x1:x2]
        
        # 템플릿 매칭
        try:
            result = cv2.matchTemplate(search_region, self.template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # 신뢰도 체크 (0.5 이상이면 성공)
            if max_val > 0.5:
                # 새로운 위치 계산
                new_x = x1 + max_loc[0]
                new_y = y1 + max_loc[1]
                new_bbox = (new_x, new_y, w, h)
                
                # 위치 업데이트
                self.last_bbox = new_bbox
                
                # 템플릿 업데이트 (적응형 추적)
                self.template = frame[new_y:new_y+h, new_x:new_x+w].copy()
                
                return True, new_bbox
            else:
                return False, None
        except:
            return False, None
    
    def draw_results(self, image, bbox, predicted_class, confidence, probabilities, is_tracking=False):
        """결과 시각화"""
        x, y, w, h = [int(v) for v in bbox]
        x1, y1, x2, y2 = x, y, x + w, y + h
        
        # 박스 색상 (추적 중이면 파란색, 아니면 초록/주황)
        if is_tracking:
            color = (255, 0, 0)  # 파란색 - 추적 중
        else:
            color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 예측 결과 텍스트
        class_name = self.class_names[predicted_class]
        status = "[TRACKING]" if is_tracking else "[DETECTED]"
        result_text = f"{status} {class_name}: {confidence*100:.1f}%"
        
        # 텍스트 배경
        text_size = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x1
        text_y = y1 - 10
        
        if text_y < 30:
            text_y = y2 + 25
        
        # 배경 박스
        cv2.rectangle(image, (text_x - 5, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 5, text_y + 5),
                     (0, 0, 0), -1)
        
        # 텍스트
        cv2.putText(image, result_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 모든 클래스 확률 표시 (오른쪽 상단)
        prob_y = 30
        cv2.putText(image, "Class Probabilities:", (image.shape[1] - 220, prob_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        for i, (cls_name, prob) in enumerate(zip(self.class_names, probabilities)):
            prob_text = f"{cls_name}: {prob*100:.1f}%"
            prob_color = (0, 255, 0) if i == predicted_class else (200, 200, 200)
            cv2.putText(image, prob_text, (image.shape[1] - 200, prob_y + (i+1) * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, prob_color, 1)
    
    def run(self):
        """실시간 추론 + 추적 실행"""
        # RealSense 파이프라인 설정
        pipeline = rs.pipeline()
        config = rs.config()
        
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # 스트리밍 시작
        try:
            pipeline.start(config)
        except RuntimeError as e:
            print("=" * 60)
            print("ERROR: RealSense 카메라를 찾을 수 없습니다!")
            print("=" * 60)
            print("\n다음 사항을 확인해주세요:")
            print("  1. RealSense 카메라가 USB 포트에 연결되어 있는지 확인")
            print("  2. 카메라의 LED가 켜져 있는지 확인")
            print("  3. 다른 프로그램에서 카메라를 사용 중인지 확인")
            print("\n원본 에러 메시지:", str(e))
            print("=" * 60)
            return
        
        # 윈도우 생성 및 마우스 콜백
        cv2.namedWindow('RealSense Tracking')
        cv2.setMouseCallback('RealSense Tracking', self.mouse_callback)
        
        print("\n" + "=" * 60)
        print("실시간 추론 + 추적 시작")
        print("=" * 60)
        print("사용법:")
        print("  1. 마우스로 드래그하여 ROI 선택")
        print("  2. 자동으로 추론 및 추적 시작")
        print("  3. [T] 키로 추적 시작/중지")
        print("  4. [C] 키로 ROI 초기화")
        print("  5. [R] 키로 재분류 (추적 중)")
        print("  6. [Q] 키로 종료")
        print("=" * 60 + "\n")
        
        try:
            while True:
                # 프레임 대기
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
                
                # 이미지 변환
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                # 표시용 이미지
                display_image = color_image.copy()
                
                # 추적 모드
                if self.tracking and self.template is not None:
                    # 트래커 업데이트
                    success, bbox = self.update_tracker(color_image)
                    
                    if success:
                        # 추적 성공 - 결과 표시
                        self.draw_results(display_image, bbox, 
                                        self.tracked_class, 
                                        self.tracked_confidence,
                                        self.tracked_probabilities,
                                        is_tracking=True)
                    else:
                        # 추적 실패
                        cv2.putText(display_image, "Tracking Lost! Press [C] to reset", 
                                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        self.tracking = False
                        self.template = None
                
                # ROI 선택 모드
                elif self.roi_start is not None and self.roi_end is not None:
                    x1 = max(0, min(self.roi_start[0], self.roi_end[0]))
                    y1 = max(0, min(self.roi_start[1], self.roi_end[1]))
                    x2 = min(639, max(self.roi_start[0], self.roi_end[0]))
                    y2 = min(479, max(self.roi_start[1], self.roi_end[1]))
                    
                    roi_coords = (x1, y1, x2, y2)
                    
                    # 그리는 중
                    if self.is_drawing:
                        cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    
                    # 선택 완료 시 추론
                    elif self.roi_selected and x2 > x1 + 10 and y2 > y1 + 10:
                        # ROI 전처리
                        roi_tensor = self.preprocess_roi(color_image, roi_coords)
                        
                        # 추론
                        predicted_class, confidence, probabilities = self.predict(roi_tensor)
                        
                        # 결과 저장 (추적용)
                        self.tracked_class = predicted_class
                        self.tracked_confidence = confidence
                        self.tracked_probabilities = probabilities
                        
                        # bbox 형식으로 변환 (x, y, w, h)
                        bbox = (x1, y1, x2 - x1, y2 - y1)
                        
                        # 결과 시각화
                        self.draw_results(display_image, bbox,
                                        predicted_class, confidence, probabilities,
                                        is_tracking=False)
                
                # UI 안내
                status_y = 30
                if self.tracking:
                    cv2.putText(display_image, "MODE: TRACKING", (10, status_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                else:
                    cv2.putText(display_image, "MODE: DETECTION", (10, status_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.putText(display_image, "[T] Track | [C] Clear | [R] Reclassify | [Q] Quit", 
                           (10, status_y + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 화면 표시
                window_name = 'RealSense Tracking'

                # 윈도우 닫기 버튼(X) 클릭 감지
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                
                # imshow() 호출 전에 닫힘을 먼저 확인하면, “닫자마자 imshow가 창을 재생성”하는 상황을 피할 수 있어요.
                cv2.imshow(window_name, display_image)

                # 키 입력
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                
                elif key == ord('c'):
                    # 초기화
                    self.roi_start = None
                    self.roi_end = None
                    self.is_drawing = False
                    self.roi_selected = False
                    self.tracking = False
                    self.template = None
                    print("✓ 초기화 완료")
                    
                elif key == ord('t'):
                    # 추적 시작/중지
                    if not self.tracking and self.roi_selected:
                        # 추적 시작
                        x1 = min(self.roi_start[0], self.roi_end[0])
                        y1 = min(self.roi_start[1], self.roi_end[1])
                        x2 = max(self.roi_start[0], self.roi_end[0])
                        y2 = max(self.roi_start[1], self.roi_end[1])
                        
                        bbox = (x1, y1, x2 - x1, y2 - y1)
                        self.init_tracker(color_image, bbox)
                        print(f"✓ 추적 시작: {self.class_names[self.tracked_class]}")
                    else:
                        # 추적 중지
                        self.tracking = False
                        self.template = None
                        print("✓ 추적 중지")
                        
                elif key == ord('r'):
                    # 재분류 (추적 중일 때)
                    if self.tracking and self.template is not None:
                        success, bbox = self.update_tracker(color_image)
                        if success:
                            x, y, w, h = [int(v) for v in bbox]
                            roi_coords = (x, y, x + w, y + h)
                            
                            # 재분류
                            roi_tensor = self.preprocess_roi(color_image, roi_coords)
                            predicted_class, confidence, probabilities = self.predict(roi_tensor)
                            
                            # 결과 업데이트
                            self.tracked_class = predicted_class
                            self.tracked_confidence = confidence
                            self.tracked_probabilities = probabilities
                            print(f"✓ 재분류: {self.class_names[predicted_class]} ({confidence*100:.1f}%)")
        
        finally:
            pipeline.stop()
            cv2.destroyAllWindows()
            print("\n프로그램 종료")


def main():
    # 모델 경로
    model_path = "models/best_model.pth"
    
    if not os.path.exists(model_path):
        print(f"ERROR: 모델 파일을 찾을 수 없습니다: {model_path}")
        print("먼저 2_2_train.py를 실행하여 모델을 학습하세요.")
        return
    
    # 추론 + 추적 실행
    tracker = RealtimeInferenceWithTracking(model_path)
    tracker.run()


if __name__ == "__main__":
    main()
