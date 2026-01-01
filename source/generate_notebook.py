import json
import os

def read_file(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

def create_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True)
    }

def create_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(True)
    }

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# 1. Title
notebook["cells"].append(create_markdown_cell("# Project Spero - RealSense Object Detection & Tracking\n\nThis notebook consolidates the data collection, training, inference, and tracking modules."))

# 2. Imports
imports_code = """
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import json
from datetime import datetime
import shutil
import random
import pyrealsense2 as rs
import matplotlib.pyplot as plt

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
"""
notebook["cells"].append(create_code_cell(imports_code))

# 3. Data Collection
notebook["cells"].append(create_markdown_cell("## 1. Data Collection\n\nRun the cell below to start data collection.\n\n*   **[L]** Set Label\n*   **[S]** Save ROI\n*   **[C]** Clear ROI\n*   **[Q]** Quit"))

data_collector_code = read_file("1_data_collector.py")
# Remove imports as they are already imported
# Wrap main logic in a function
# data_collector_code = data_collector_code.replace("if __name__ == \"__main__\":", "# if __name__ == \"__main__\":")
# data_collector_code = data_collector_code.replace("main()", "# main()")
# data_collector_code += "\n\n# Run Data Collector\n# main()  # Uncomment to run"
notebook["cells"].append(create_code_cell(data_collector_code))

# 4. Dataset Preparation
notebook["cells"].append(create_markdown_cell("## 2. Dataset Preparation\n\nSplit the collected data into train/val/test sets."))

dataset_utils_code = read_file("2_1_dataset_utils.py")
dataset_code = read_file("dataset.py")

# Merge and clean up
dataset_prep_code = dataset_utils_code + "\n\n" + dataset_code
dataset_prep_code = dataset_prep_code.replace("if __name__ == \"__main__\":", "# if __name__ == \"__main__\":")
dataset_prep_code += "\n\nif True: # Modified for Notebook\n    split_dataset()\n    \n    # 데이터셋 테스트 실행\n    print('\\n' + '=' * 60)\n    print('데이터셋 로드 테스트')\n    print('=' * 60)\n    try:\n        train_dataset = RealSenseDataset('dataset_split/train', transform=get_transforms(train=True))\n        val_dataset = RealSenseDataset('dataset_split/val', transform=get_transforms(train=False))\n        print(f'Train 데이터셋 크기: {len(train_dataset)}')\n        print(f'Val 데이터셋 크기: {len(val_dataset)}')\n        print(f'클래스 목록: {train_dataset.classes}')\n    except Exception as e:\n        print(f'데이터셋 로드 중 오류 발생: {e}')"
notebook["cells"].append(create_code_cell(dataset_prep_code))

# 5. Training
notebook["cells"].append(create_markdown_cell("## 3. Training\n\nTrain the model using the collected dataset."))

train_code = read_file("2_2_train.py")
train_code = train_code.replace("from dataset import RealSenseDataset, get_transforms", "# from dataset import RealSenseDataset, get_transforms")
# Comment out the if __name__ == "__main__": block selectively
train_code = train_code.replace('if __name__ == "__main__":', 'if True: # Modified for Notebook')
#train_code = train_code.replace('    main()', '    # main() # Uncomment to run manual training')
# Ensure num_workers is 0 again just in case (redundant but safe)
train_code = train_code.replace("'num_workers': 2", "'num_workers': 0") 
notebook["cells"].append(create_code_cell(train_code))

# 6. Inference
notebook["cells"].append(create_markdown_cell("## 4. Inference\n\nReal-time inference using the trained model."))

inference_code = read_file("3_inference.py")
inference_code = inference_code.replace('if __name__ == "__main__":', 'if True: # Modified for Notebook')
# inference_code = inference_code.replace('    main()', '    # main() # Uncomment to run manual inference')
notebook["cells"].append(create_code_cell(inference_code))

# 7. Tracking
notebook["cells"].append(create_markdown_cell("## 5. Tracking\n\nReal-time object tracking."))

tracking_code = read_file("4_tracking.py")
tracking_code = tracking_code.replace('if __name__ == "__main__":', 'if True: # Modified for Notebook')
# tracking_code = tracking_code.replace('    main()', '    # main() # Uncomment to run manual tracking')
notebook["cells"].append(create_code_cell(tracking_code))

notebook_filename = "Project_Spero.ipynb"
with open(notebook_filename, "w", encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook created successfully: [{notebook_filename}]")
