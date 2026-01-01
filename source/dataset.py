"""
PyTorch Dataset 클래스 정의
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class RealSenseDataset(Dataset):
    """RealSense RGB-D 데이터셋"""
    
    def __init__(self, root_dir, transform=None, use_depth=False, img_size=224):
        """
        Args:
            root_dir: 데이터셋 루트 디렉토리 (train, val, test 중 하나)
            transform: 이미지 변환 (torchvision.transforms)
            use_depth: Depth 정보 사용 여부
            img_size: 이미지 크기 (정사각형)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.use_depth = use_depth
        self.img_size = img_size
        
        # 클래스 이름 및 인덱스 매핑
        self.classes = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # 샘플 수집
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            color_images = [f for f in os.listdir(class_dir) 
                           if f.endswith('_color.png')]
            
            for color_img in color_images:
                timestamp = color_img.replace('_color.png', '')
                color_path = os.path.join(class_dir, color_img)
                depth_path = os.path.join(class_dir, f"{timestamp}_depth.png")
                
                # Depth 파일 존재 확인
                if self.use_depth and not os.path.exists(depth_path):
                    continue
                
                self.samples.append({
                    'color': color_path,
                    'depth': depth_path if self.use_depth else None,
                    'label': self.class_to_idx[class_name],
                    'class_name': class_name
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Color 이미지 로드
        color_image = cv2.imread(sample['color'])
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        
        # 이미지 크기 조정
        color_image = cv2.resize(color_image, (self.img_size, self.img_size))
        
        if self.use_depth:
            # Depth 이미지 로드
            depth_image = cv2.imread(sample['depth'], cv2.IMREAD_UNCHANGED)
            depth_image = cv2.resize(depth_image, (self.img_size, self.img_size))
            
            # Depth 정규화 (0-255 범위로)
            depth_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_image = depth_image.astype(np.uint8)
            
            # RGB + D = 4채널
            image = np.dstack([color_image, depth_image])
        else:
            image = color_image
        
        # PIL Image로 변환 (transforms 적용을 위해)
        if self.use_depth:
            # 4채널은 별도 처리
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        else:
            image = Image.fromarray(image)
            if self.transform:
                image = self.transform(image)
        
        label = sample['label']
        
        return image, label
    
    def get_class_name(self, idx):
        """인덱스로부터 클래스 이름 반환"""
        return self.classes[idx]


def get_transforms(img_size=224, augment=True):
    """
    데이터 변환 정의
    
    Args:
        img_size: 이미지 크기
        augment: 데이터 증강 사용 여부
    """
    if augment:
        # 학습용 변환 (데이터 증강 포함)
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # 검증/테스트용 변환
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


if __name__ == "__main__":
    # 데이터셋 테스트
    print("=" * 60)
    print("데이터셋 테스트")
    print("=" * 60)
    
    if os.path.exists("dataset_split/train"):
        train_transform, val_transform = get_transforms()
        
        train_dataset = RealSenseDataset("dataset_split/train", 
                                        transform=train_transform,
                                        use_depth=False)
        
        print(f"\n클래스: {train_dataset.classes}")
        print(f"클래스 수: {len(train_dataset.classes)}")
        print(f"학습 샘플 수: {len(train_dataset)}")
        
        # 첫 번째 샘플 확인
        if len(train_dataset) > 0:
            image, label = train_dataset[0]
            print(f"\n샘플 확인:")
            print(f"  이미지 shape: {image.shape}")
            print(f"  라벨: {label} ({train_dataset.get_class_name(label)})")
    else:
        print("⚠ dataset_split/train 폴더를 찾을 수 없습니다.")
        print("먼저 dataset_utils.py를 실행하여 데이터셋을 분할하세요.")
