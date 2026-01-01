import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime
import json

class DataCollector:
    def __init__(self):
        # 전역 변수 초기화
        self.roi_start = None
        self.roi_end = None
        self.is_drawing = False
        self.roi_selected = False
        self.current_label = ""
        self.save_count = 0
        
        # 데이터 저장 경로
        self.BASE_DIR = "dataset"
        self.METADATA_FILE = "metadata.json"
        
        # RealSense 파이프라인 설정
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.depth_scale = 0

    def mouse_callback(self, event, x, y, flags, param):
        """마우스 이벤트 처리 - ROI 선택"""
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

    def create_directory_structure(self):
        """데이터셋 디렉토리 구조 생성"""
        if not os.path.exists(self.BASE_DIR):
            os.makedirs(self.BASE_DIR)
            print(f"✓ 디렉토리 생성: {self.BASE_DIR}")

    def save_roi_data(self, color_image, depth_image, roi_coords):
        """ROI 데이터 저장"""
        if not self.current_label:
            print("⚠ 라벨이 설정되지 않았습니다. 먼저 라벨을 입력하세요.")
            return False
        
        x1, y1, x2, y2 = roi_coords
        
        # 클래스별 디렉토리 생성
        class_dir = os.path.join(self.BASE_DIR, self.current_label)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
            print(f"✓ 새 클래스 디렉토리 생성: {class_dir}")
        
        # 타임스탬프 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # ROI 영역 추출
        roi_color = color_image[y1:y2, x1:x2]
        roi_depth = depth_image[y1:y2, x1:x2]
        
        # 파일명 생성
        color_filename = f"{timestamp}_color.png"
        depth_filename = f"{timestamp}_depth.png"
        
        color_path = os.path.join(class_dir, color_filename)
        depth_path = os.path.join(class_dir, depth_filename)
        
        # 이미지 저장
        cv2.imwrite(color_path, roi_color)
        
        # Depth 이미지를 16-bit로 저장
        cv2.imwrite(depth_path, roi_depth)
        
        # 메타데이터 저장
        valid_depth = roi_depth[roi_depth > 0]
        metadata = {
            "timestamp": timestamp,
            "label": self.current_label,
            "roi": [x1, y1, x2, y2],
            "roi_size": [x2 - x1, y2 - y1],
            "color_image": color_filename,
            "depth_image": depth_filename,
            "depth_avg": float(np.mean(valid_depth) * self.depth_scale) if len(valid_depth) > 0 else 0.0,
            "depth_min": float(np.min(valid_depth) * self.depth_scale) if len(valid_depth) > 0 else 0.0,
            "depth_max": float(np.max(valid_depth) * self.depth_scale) if len(valid_depth) > 0 else 0.0,
        }
        
        # 메타데이터 파일에 추가
        metadata_path = os.path.join(class_dir, self.METADATA_FILE)
        metadata_list = []
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_list = json.load(f)
        
        metadata_list.append(metadata)
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, indent=2, ensure_ascii=False)
        
        self.save_count += 1
        print(f"✓ 저장 완료 [{self.save_count}]: {self.current_label}/{color_filename}")
        return True

    def draw_ui(self, image, roi_coords=None):
        """UI 요소 그리기"""
        height, width = image.shape[:2]
        
        # 상단 정보 패널
        panel_height = 120
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # 제목
        cv2.putText(image, "Data Collector - RealSense ROI Labeling", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 현재 라벨 표시
        label_text = f"Current Label: {self.current_label if self.current_label else '[Not Set]'}"
        label_color = (0, 255, 0) if self.current_label else (0, 0, 255)
        cv2.putText(image, label_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
        
        # 저장 카운트
        cv2.putText(image, f"Saved: {self.save_count}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 하단 도움말
        help_y = height - 100
        cv2.rectangle(image, (0, help_y), (width, height), (0, 0, 0), -1)
        
        help_texts = [
            "[ L ] Set Label  |  [ S ] Save ROI  |  [ C ] Clear ROI  |  [ Q ] Quit",
            "Drag mouse to select ROI"
        ]
        
        for i, text in enumerate(help_texts):
            cv2.putText(image, text, (10, help_y + 25 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # ROI 정보 표시
        if roi_coords:
            x1, y1, x2, y2 = roi_coords
            roi_info = f"ROI: ({x2-x1}x{y2-y1})"
            cv2.putText(image, roi_info, (width - 200, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def run(self):
        # 디렉토리 구조 생성
        self.create_directory_structure()
        
        # 스트리밍 시작
        try:
            profile = self.pipeline.start(self.config)
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
        
        # 깊이 스케일 가져오기
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        
        # 윈도우 생성 및 마우스 콜백 설정
        cv2.namedWindow('Data Collector')
        cv2.setMouseCallback('Data Collector', self.mouse_callback)
        
        print("\n" + "=" * 60)
        print("데이터 수집 프로그램 시작")
        print("=" * 60)
        print("사용법:")
        print("  1. [L] 키를 눌러 라벨(클래스명) 입력")
        print("  2. 마우스로 드래그하여 ROI 선택")
        print("  3. [S] 키를 눌러 저장")
        print("  4. [C] 키로 ROI 초기화")
        print("  5. [Q] 키로 종료")
        print("=" * 60 + "\n")
        
        try:
            while True:
                # 프레임 대기
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
                
                # 이미지를 numpy 배열로 변환
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                # 표시용 이미지 복사
                display_image = color_image.copy()
                
                # ROI 그리기
                roi_coords = None
                if self.roi_start is not None and self.roi_end is not None:
                    x1 = max(0, min(self.roi_start[0], self.roi_end[0]))
                    y1 = max(0, min(self.roi_start[1], self.roi_end[1]))
                    x2 = min(639, max(self.roi_start[0], self.roi_end[0]))
                    y2 = min(479, max(self.roi_start[1], self.roi_end[1]))
                    
                    roi_coords = (x1, y1, x2, y2)
                    
                    # ROI 사각형 그리기
                    color = (0, 255, 255) if self.is_drawing else (0, 255, 0)
                    cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
                    
                    # ROI 영역 반투명 오버레이
                    if self.roi_selected and x2 > x1 and y2 > y1:
                        overlay = display_image.copy()
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
                        cv2.addWeighted(overlay, 0.2, display_image, 0.8, 0, display_image)
                
                # UI 그리기
                self.draw_ui(display_image, roi_coords)
                
                window_name = 'Data Collector'
                # 윈도우 닫기 버튼(X) 클릭 감지
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

                # 화면 표시
                cv2.imshow(window_name, display_image)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n프로그램을 종료합니다.")
                    break
                    
                elif key == ord('l'):
                    # 라벨 입력
                    print("\n" + "-" * 40)
                    new_label = input("클래스 라벨을 입력하세요: ").strip()
                    if new_label:
                        self.current_label = new_label
                        print(f"✓ 라벨 설정: {self.current_label}")
                    else:
                        print("⚠ 라벨이 비어있습니다.")
                    print("-" * 40 + "\n")
                    
                elif key == ord('s'):
                    # ROI 저장
                    if self.roi_selected and roi_coords:
                        x1, y1, x2, y2 = roi_coords
                        if x2 > x1 and y2 > y1:
                            self.save_roi_data(color_image, depth_image, roi_coords)
                        else:
                            print("⚠ 유효하지 않은 ROI입니다.")
                    else:
                        print("⚠ ROI를 먼저 선택하세요.")
                        
                elif key == ord('c'):
                    # ROI 초기화
                    self.roi_start = None
                    self.roi_end = None
                    self.is_drawing = False
                    self.roi_selected = False
                    print("✓ ROI 초기화")
        
        finally:
            # 정리
            self.pipeline.stop()
            cv2.destroyAllWindows()
            
            print("\n" + "=" * 60)
            print(f"총 {self.save_count}개의 샘플이 저장되었습니다.")
            print(f"데이터 위치: {os.path.abspath(self.BASE_DIR)}")
            print("=" * 60)

if __name__ == "__main__":
    collector = DataCollector()
    collector.run()
