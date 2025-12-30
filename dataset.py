import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from DataLoader import load_dataset 

# ==========================================
# 1. Custom Dataset Class
# ==========================================
class ContrastiveDataset(Dataset):
    def __init__(self, X, y, mode='train', noise_level=0.01):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.mode = mode
        self.noise_level = noise_level

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.mode == 'train' and y == 1:
            # Masking Augmentation
            mask = torch.rand_like(x) > self.noise_level
            x = x * mask.float()

        return x, y

# ==========================================
# 2. Data Loader Function (수정됨)
# ==========================================
def get_dataloaders(config):
    
    df = load_dataset('fraud') 
    
    X = df.drop(columns=['Class']).values
    y = df['Class'].astype(int).values

    print(f"\n[Data Loaded] Input Dim: {X.shape[1]}")

    # 3. 데이터 분할 (Train / Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config['SEED'], stratify=y
    )

    # 4. Contrastive Learning을 위한 데이터 재구성 (증강)
    X_train_normal = X_train[y_train == 0]
    X_train_fraud  = X_train[y_train == 1]
    

    target_fraud_count = int(len(X_train_normal) * 0.2)
    
    indices = np.random.choice(len(X_train_fraud), target_fraud_count, replace=True)
    X_train_fraud_augmented = X_train_fraud[indices]
    y_train_fraud_augmented = np.ones(target_fraud_count)

    # 합치기
    X_train_final = np.concatenate([X_train_normal, X_train_fraud_augmented], axis=0)
    y_train_final = np.concatenate([np.zeros(len(X_train_normal)), y_train_fraud_augmented], axis=0)

    print(f"Augmented Train Counts -> Normal: {len(X_train_normal)}, Fraud: {len(X_train_fraud_augmented)}")

    # 5. Dataset & DataLoader 생성
    train_dataset = ContrastiveDataset(X_train_final, y_train_final, mode='train')
    test_dataset = ContrastiveDataset(X_test, y_test, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'], shuffle=False)

    return train_loader, test_loader, X.shape[1]