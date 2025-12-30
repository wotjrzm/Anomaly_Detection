import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve, confusion_matrix, auc
from tqdm import tqdm
from dataset import get_dataloaders
from model import TransformerVAE

# ==========================================
# 1. Configuration
# ==========================================
CONFIG = {
    'SEED': 42,
    'BATCH_SIZE': 256,
    'LATENT_DIM': 16,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def load_model_and_data():
    print(f"Loading Model & Data on {CONFIG['DEVICE']}...")
    _, test_loader, input_dim = get_dataloaders(CONFIG)
    
    model = TransformerVAE(input_dim=input_dim, latent_dim=CONFIG['LATENT_DIM']).to(CONFIG['DEVICE'])
    try:
        model.load_state_dict(torch.load("model/best_model.pth", map_location=CONFIG['DEVICE']))
        print("'best_model.pth' 로드 성공!")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        exit()
        
    model.eval()
    return model, test_loader

def get_mahalanobis_params(model, dataloader, device):
    """
    정상 데이터의 분포(평균, 공분산 역행렬)를 미리 계산
    """
    print("1. 정상 데이터 분포 학습 중...")
    z_normals = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            # 정상(0) 데이터만 추출
            if (y == 0).sum() > 0:
                mu, _ = model.encode(x[y==0])
                z_normals.append(mu.cpu().numpy())
    
    z_normals = np.concatenate(z_normals)
    
    # 평균과 공분산 계산
    mean = np.mean(z_normals, axis=0)
    cov = np.cov(z_normals, rowvar=False)
    # 역행렬 계산 (특이행렬 방지용 pinv 사용)
    inv_cov = np.linalg.pinv(cov)
    
    return mean, inv_cov

def get_hybrid_scores(model, dataloader, device, mean, inv_cov):
    """
    Recon Error + Mahalanobis Distance 계산
    """
    print("2. 전체 데이터 스코어링 (Mahalanobis 적용)...")
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            
            # 1) 재구축 오차 (Reconstruction Error)
            recon_x, mu, _, _ = model(x)
            recon_loss = torch.mean(torch.abs(x - recon_x), dim=1).cpu().numpy()
            
            # 2) 마할라노비스 거리 (Mahalanobis Distance)
            # (x - mu)^T * Sigma^-1 * (x - mu)
            z_numpy = mu.cpu().numpy()
            diff = z_numpy - mean
            # Vectorized implementation for batch processing
            # (Batch, Dim) @ (Dim, Dim) -> (Batch, Dim)
            left = np.dot(diff, inv_cov) 
            # Row-wise dot product
            mahal_dist = np.sqrt(np.sum(left * diff, axis=1))
            
            # 3) 최종 점수 (단순 합산 혹은 가중치 적용 가능)
            final_score = recon_loss + mahal_dist
            
            all_scores.extend(final_score)
            all_labels.extend(y.cpu().numpy())
            
    return np.array(all_scores), np.array(all_labels)

# ==========================================
# 시각화 함수들 (추가된 PR Curve, Confusion Matrix)
# ==========================================

def plot_performance_metrics(scores, labels):
    """
    성과 입증용 핵심 그래프 3종 세트
    """
    print("\n[Vis] 성과 분석 그래프 생성 중...")
    plt.figure(figsize=(18, 5))
    
    # --- 1. Score Distribution (Separability) ---
    plt.subplot(1, 3, 1)
    sns.histplot(x=scores, hue=labels, kde=True, bins=50, 
                 palette={0: 'blue', 1: 'red'}, common_norm=False, stat='density')
    plt.title('Anomaly Score Distribution', fontsize=14)
    plt.xlabel('Hybrid Score (Recon + Mahalanobis)')
    plt.legend(title='Label', labels=['Fraud', 'Normal'])
    plt.grid(alpha=0.3)

    # --- 2. PR Curve & Best F1 ---
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    pr_auc = auc(recall, precision)
    
    plt.subplot(1, 3, 2)
    plt.plot(recall, precision, marker='.', label=f'AUPRC = {pr_auc:.4f}', color='darkorange')
    plt.scatter(recall[best_idx], precision[best_idx], marker='o', c='black', s=100, label=f'Best F1 ({f1_scores[best_idx]:.4f})', zorder=10)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(alpha=0.3)

    # --- 3. Confusion Matrix (at Best F1) ---
    plt.subplot(1, 3, 3)
    preds = (scores >= best_thresh).astype(int)
    cm = confusion_matrix(labels, preds)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 14})
    plt.title(f'Confusion Matrix (Thresh={best_thresh:.3f})', fontsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0.5, 1.5], ['Normal', 'Fraud'])
    plt.yticks([0.5, 1.5], ['Normal', 'Fraud'])

    plt.tight_layout()
    if not os.path.exists('image'):
        os.makedirs('image')
    
    save_path = "image/vis_performance_metrics.png"
    plt.savefig(save_path)
    print(f"✅ 저장 완료: {save_path}")

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # 1. 모델 로드
    model, test_loader = load_model_and_data()
    
    # 2. 정상 데이터 분포(Mean, Cov) 계산
    mean, inv_cov = get_mahalanobis_params(model, test_loader, CONFIG['DEVICE'])
    
    # 3. Mahalanobis 기반 점수 계산
    scores, labels = get_hybrid_scores(model, test_loader, CONFIG['DEVICE'], mean, inv_cov)
    
    # 4. 핵심 그래프 3종 그리기 (분포, PR Curve, 혼동행렬)
    plot_performance_metrics(scores, labels)
    
    print("\n분석 완료 'image/vis_performance_metrics.png'")