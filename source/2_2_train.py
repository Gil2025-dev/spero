"""
PyTorch 학습 스크립트
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

from dataset import RealSenseDataset, get_transforms

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"사용 디바이스: {self.device}")
        
        # 데이터셋 로드
        self.load_datasets()
        
        # 모델 생성
        self.create_model()
        
        # Loss, Optimizer 설정
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 학습 기록
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        self.best_val_acc = 0.0
    
    def load_datasets(self):
        """데이터셋 로드"""
        train_transform, val_transform = get_transforms(
            img_size=self.config['img_size'],
            augment=self.config['use_augmentation']
        )
        
        self.train_dataset = RealSenseDataset(
            root_dir=self.config['train_dir'],
            transform=train_transform,
            use_depth=self.config['use_depth'],
            img_size=self.config['img_size']
        )
        
        self.val_dataset = RealSenseDataset(
            root_dir=self.config['val_dir'],
            transform=val_transform,
            use_depth=self.config['use_depth'],
            img_size=self.config['img_size']
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers']
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers']
        )
        
        self.num_classes = len(self.train_dataset.classes)
        self.class_names = self.train_dataset.classes
        
        print(f"\n데이터셋 로드 완료:")
        print(f"  클래스: {self.class_names}")
        print(f"  학습 샘플: {len(self.train_dataset)}")
        print(f"  검증 샘플: {len(self.val_dataset)}")
    
    def create_model(self):
        """모델 생성"""
        model_name = self.config['model_name']
        
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=self.config['use_pretrained'])
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, self.num_classes)
            
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=self.config['use_pretrained'])
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, self.num_classes)
            
        elif model_name == 'mobilenet_v2':
            self.model = models.mobilenet_v2(pretrained=self.config['use_pretrained'])
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, self.num_classes)
            
        else:
            raise ValueError(f"지원하지 않는 모델: {model_name}")
        
        self.model = self.model.to(self.device)
        print(f"\n모델 생성 완료: {model_name}")
        print(f"  사전 학습 가중치: {'사용' if self.config['use_pretrained'] else '미사용'}")
    
    def train_epoch(self):
        """1 에폭 학습"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # 통계
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Progress bar 업데이트
            pbar.set_postfix({
                'loss': f'{running_loss/len(pbar):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """검증"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{running_loss/len(pbar):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self):
        """전체 학습 루프"""
        print("\n" + "=" * 60)
        print("학습 시작")
        print("=" * 60)
        
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            print("-" * 60)
            
            # 학습
            train_loss, train_acc = self.train_epoch()
            
            # 검증
            val_loss, val_acc = self.validate()
            
            # 학습률 조정
            self.scheduler.step(val_loss)
            
            # 기록
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # 결과 출력
            print(f"\n결과:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # 최고 성능 모델 저장
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint('best_model.pth', epoch, val_acc)
                print(f"  ✓ 최고 성능 모델 저장 (Val Acc: {val_acc:.2f}%)")
            
            # Early Stopping (선택사항)
            if self.config['early_stopping_patience'] > 0:
                if epoch > self.config['early_stopping_patience']:
                    recent_val_acc = self.history['val_acc'][-self.config['early_stopping_patience']:]
                    if all(acc <= self.best_val_acc for acc in recent_val_acc):
                        print(f"\nEarly stopping at epoch {epoch+1}")
                        break
        
        # 최종 모델 저장
        self.save_checkpoint('final_model.pth', self.config['num_epochs'], val_acc)
        
        # 학습 기록 저장
        self.save_history()
        
        print("\n" + "=" * 60)
        print("학습 완료!")
        print(f"최고 검증 정확도: {self.best_val_acc:.2f}%")
        print("=" * 60)
    
    def save_checkpoint(self, filename, epoch, val_acc):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'class_names': self.class_names,
            'config': self.config
        }
        
        save_path = os.path.join(self.config['save_dir'], filename)
        torch.save(checkpoint, save_path)
    
    def save_history(self):
        """학습 기록 저장"""
        history_path = os.path.join(self.config['save_dir'], 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"학습 기록 저장: {history_path}")


def main():
    # 학습 설정
    config = {
        # 데이터
        'train_dir': 'dataset_split/train',
        'val_dir': 'dataset_split/val',
        'img_size': 224,
        'use_depth': False,  # Depth 정보 사용 여부
        'use_augmentation': True,
        
        # 모델
        'model_name': 'mobilenet_v2',  # 'resnet18', 'resnet50', 'mobilenet_v2'
        'use_pretrained': True,  # Transfer Learning
        
        # 학습
        'batch_size': 16,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'num_workers': 0,  # Windows Jupyter Notebook: 0으로 설정 (멀티프로세싱 이슈 방지)
        'early_stopping_patience': 10,
        
        # 저장
        'save_dir': 'models'
    }
    
    # 저장 디렉토리 생성
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # 설정 출력
    print("=" * 60)
    print("학습 설정")
    print("=" * 60)
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 학습 시작
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
