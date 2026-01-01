"""
데이터셋 유틸리티 - train/val/test 분할 및 데이터 로딩
"""
import os
import shutil
import json
from pathlib import Path
import random

def split_dataset(source_dir="dataset", output_dir="dataset_split", 
                  train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    데이터셋을 train/val/test로 분할
    
    Args:
        source_dir: 원본 데이터셋 디렉토리
        output_dir: 분할된 데이터셋 저장 디렉토리
        train_ratio: 학습 데이터 비율
        val_ratio: 검증 데이터 비율
        test_ratio: 테스트 데이터 비율
        seed: 랜덤 시드
    """
    random.seed(seed)
    
    # 비율 검증
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train_ratio + val_ratio + test_ratio must equal 1.0"
    
    # 출력 디렉토리 생성
    splits = ['train', 'val', 'test']
    for split in splits:
        split_dir = os.path.join(output_dir, split)
        if os.path.exists(split_dir):
            print(f"⚠ {split_dir} 이미 존재합니다. 건너뜁니다.")
        else:
            os.makedirs(split_dir)
    
    # 클래스별 처리
    class_dirs = [d for d in os.listdir(source_dir) 
                  if os.path.isdir(os.path.join(source_dir, d))]
    
    print(f"\n발견된 클래스: {class_dirs}")
    print(f"분할 비율 - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}\n")
    
    total_stats = {'train': 0, 'val': 0, 'test': 0}
    
    for class_name in class_dirs:
        class_path = os.path.join(source_dir, class_name)
        
        # 클래스별 이미지 파일 수집 (color 이미지만)
        color_images = [f for f in os.listdir(class_path) 
                       if f.endswith('_color.png')]
        
        # 타임스탬프 추출 (중복 방지)
        timestamps = list(set([img.replace('_color.png', '') for img in color_images]))
        
        # 셔플
        random.shuffle(timestamps)
        
        # 분할 인덱스 계산
        n_total = len(timestamps)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_timestamps = timestamps[:n_train]
        val_timestamps = timestamps[n_train:n_train + n_val]
        test_timestamps = timestamps[n_train + n_val:]
        
        # 각 split에 복사
        split_data = {
            'train': train_timestamps,
            'val': val_timestamps,
            'test': test_timestamps
        }
        
        for split, timestamps_list in split_data.items():
            # 클래스 디렉토리 생성
            split_class_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(split_class_dir, exist_ok=True)
            
            # 파일 복사
            for timestamp in timestamps_list:
                # Color 이미지
                color_src = os.path.join(class_path, f"{timestamp}_color.png")
                color_dst = os.path.join(split_class_dir, f"{timestamp}_color.png")
                
                # Depth 이미지
                depth_src = os.path.join(class_path, f"{timestamp}_depth.png")
                depth_dst = os.path.join(split_class_dir, f"{timestamp}_depth.png")
                
                if os.path.exists(color_src):
                    shutil.copy2(color_src, color_dst)
                if os.path.exists(depth_src):
                    shutil.copy2(depth_src, depth_dst)
            
            total_stats[split] += len(timestamps_list)
            print(f"  {class_name}/{split}: {len(timestamps_list)} samples")
    
    print(f"\n총 분할 결과:")
    print(f"  Train: {total_stats['train']} samples")
    print(f"  Val: {total_stats['val']} samples")
    print(f"  Test: {total_stats['test']} samples")
    print(f"  Total: {sum(total_stats.values())} samples")
    print(f"\n✓ 데이터셋 분할 완료: {output_dir}")

def get_class_names(dataset_dir):
    """데이터셋에서 클래스 이름 추출"""
    class_names = sorted([d for d in os.listdir(dataset_dir) 
                         if os.path.isdir(os.path.join(dataset_dir, d))])
    return class_names

def count_samples(dataset_dir):
    """데이터셋의 샘플 수 계산"""
    class_names = get_class_names(dataset_dir)
    stats = {}
    
    for class_name in class_names:
        class_path = os.path.join(dataset_dir, class_name)
        color_images = [f for f in os.listdir(class_path) 
                       if f.endswith('_color.png')]
        stats[class_name] = len(color_images)
    
    return stats

if __name__ == "__main__":
    # 데이터셋 정보 출력
    print("=" * 60)
    print("데이터셋 분석")
    print("=" * 60)
    
    if os.path.exists("dataset"):
        stats = count_samples("dataset")
        print("\n클래스별 샘플 수:")
        for class_name, count in stats.items():
            print(f"  {class_name}: {count} samples")
        print(f"\n총 샘플 수: {sum(stats.values())}")
        
        # 데이터셋 분할
        print("\n" + "=" * 60)
        print("데이터셋 분할 시작")
        print("=" * 60)
        split_dataset()
    else:
        print("⚠ dataset 폴더를 찾을 수 없습니다.")
