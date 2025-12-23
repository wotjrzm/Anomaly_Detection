import pandas as pd
import numpy as np
import urllib.request
import requests
import zipfile
import io
import re
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_openml
import torch
from torchvision import datasets, transforms
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.datasets import MovieLens

def load_dataset(name):
    """
    최종 프로젝트용 데이터셋을 로드하고 전처리하여 반환하는 통합 함수입니다.
    
    Args:
        name (str): 데이터셋 이름 
                    Options: ['nasa', 'telco', 'fraud', 'cora', 'movielens', 'fmnist', 'mnist', 'news']
    
    Returns:
        Data: 전처리가 완료된 데이터 (DataFrame, Tensor, 또는 PyG Data 객체)
    """
    name = name.lower().strip()
    print(f"Loading and Preprocessing dataset: [{name}]...")

    # ==========================================
    # Theme A. 생존 분석 (Survival Analysis)
    # ==========================================
    
    if name == 'nasa':
        # 1. 데이터 로드
        col_names = ['unit_nr', 'time_cycles', 'os_1', 'os_2', 'os_3'] + [f's_{i}' for i in range(1, 22)]
        df = pd.read_csv('./data/NASS_CMAPSS/train_FD001.txt', sep='\s+', header=None, names=col_names)

        # 2. 전처리: RUL(잔여 수명) 라벨 생성
        # 각 엔진(unit_nr)별 최대 수명을 구함
        max_life = df.groupby('unit_nr')['time_cycles'].max().reset_index()
        max_life.columns = ['unit_nr', 'max_life']
        
        # 원본에 병합 후 RUL 계산 (최대 수명 - 현재 사이클)
        df = df.merge(max_life, on='unit_nr', how='left')
        df['RUL'] = df['max_life'] - df['time_cycles']
        df.drop(columns=['max_life'], inplace=True)
        
        print(f"Success! Shape: {df.shape}. Target column: 'RUL'")
        return df

    elif name == 'telco':
        # 1. 데이터 다운로드
        url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        df = pd.read_csv(url)

        # 2. 전처리
        # (1) TotalCharges의 빈 문자열을 숫자로 변환 (NaN은 0으로 대체)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
        
        # (2) 생존 분석을 위한 Event(E) 변수 생성 (Yes=1, No=0)
        if 'Churn' in df.columns:
            df['E'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
            df.drop(columns=['Churn'], inplace=True)
        
        # (3) 불필요한 ID 컬럼 제거
        if 'customerID' in df.columns:
            df.drop(columns=['customerID'], inplace=True)
            
        # (4) 범주형 변수 One-hot Encoding
        df = pd.get_dummies(df, drop_first=True)
        
        print(f"Success! Shape: {df.shape}. Time: 'tenure', Event: 'E'")
        return df

    # ==========================================
    # Theme B. 그래프 신경망 (GNN)
    # ==========================================

    elif name == 'cora':
        # 1. 다운로드 및 Row-Normalize 전처리
        dataset = Planetoid(root='./data/Cora', name='Cora', transform=T.NormalizeFeatures())
        data = dataset[0]
        print(f"Success! Nodes: {data.num_nodes}, Edges: {data.num_edges}, Classes: {dataset.num_classes}")
        return dataset, data

    elif name == 'movielens':
        print("Note: This requires 'sentence-transformers' library.")
        # 1. 다운로드 및 텍스트 임베딩 전처리 (all-MiniLM-L6-v2 모델 사용)
        dataset = MovieLens(root='./data/MovieLens', model_name='all-MiniLM-L6-v2')
        data = dataset[0]
        print(f"Success! Nodes: {data.num_nodes}, Edges: {data.num_edges}")
        return data

    # ==========================================
    # Theme C. 신뢰할 수 있는 AI (Uncertainty)
    # ==========================================

    elif name == 'fraud':
        # 1. OpenML에서 신용카드 사기 데이터 로드
        data = fetch_openml(data_id=42175, as_frame=True, parser='auto')
        df = data.frame
        
        # 2. 전처리: Amount 변수 스케일링 (값의 범위가 너무 큼)
        scaler = StandardScaler()
        df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
        df.drop(columns=['Time', 'Amount'], inplace=True)
        
        print(f"Success! Shape: {df.shape}. Fraud Ratio: {df['Class'].mean():.4f}")
        return df

    elif name == 'mnist':
        # 1. OOD(Out-of-Distribution) 탐지용 MNIST (In-distribution)
        # ToTensor()가 0~255 픽셀값을 0~1로 정규화해줌
        data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        print(f"Success! Size: {len(data)}. Use this as In-Distribution data.")
        return data

    # ==========================================
    # Theme D. 심층 군집화 (Deep Clustering)
    # ==========================================

    elif name == 'fmnist':
        # 1. 군집화용 Fashion-MNIST
        # 비지도 학습이므로 Label을 사용하지 않도록 안내
        data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        print(f"Success! Size: {len(data)}.")
        print("CAUTION: Do not use 'labels' for training. Use only images for clustering.")
        return data

    elif name == 'news':
        # 1. UCI 뉴스 데이터 다운로드 (Zip 파일)
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00359/NewsAggregatorDataset.zip"
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        
        # 2. CSV 로드
        with z.open('newsCorpora.csv') as f:
            col_names = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP']
            df = pd.read_csv(f, sep='\t', names=col_names, header=None)
            
        # 3. 전처리: 텍스트 정제 및 TF-IDF 벡터화
        # (1) 알파벳만 남기고 소문자 변환
        df['clean_title'] = df['TITLE'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x).lower().strip())
        
        # (2) TF-IDF 변환 (최대 2000개 단어)
        vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
        X = vectorizer.fit_transform(df['clean_title']).toarray()
        
        print(f"Success! Vectorized Shape: {X.shape}.")
        print("Returns tuple: (X_features, y_categories)")
        # 군집화용 입력 데이터 X와, 나중에 군집 해석용으로 쓸 정답 y를 튜플로 반환
        return X, df['CATEGORY']

    else:
        available = ['nasa', 'telco', 'fraud', 'cora', 'movielens', 'fmnist', 'mnist', 'news']
        raise ValueError(f"Unknown dataset name: '{name}'. Available options: {available}")