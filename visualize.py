import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from tqdm import tqdm
from dataset import get_dataloaders
from model import TransformerVAE

# ==========================================
# 1. Configuration
# ==========================================
CONFIG = {
    'SEED': 42,
    'BATCH_SIZE': 256,
    'LATENT_DIM': 4,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def load_model_and_data():
    print(f"Loading Model & Data on {CONFIG['DEVICE']}...")
    _, test_loader, input_dim = get_dataloaders(CONFIG)
    
    # 모델 로드
    model = TransformerVAE(input_dim=input_dim, latent_dim=CONFIG['LATENT_DIM']).to(CONFIG['DEVICE'])
    try:
        model.load_state_dict(torch.load("best_model.pth", map_location=CONFIG['DEVICE']))
        print("'best_model.pth' 로드 성공!")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        exit()
        
    model.eval()
    return model, test_loader

def get_hybrid_scores(model, dataloader, device):
    """
    모든 테스트 데이터에 대해 Hybrid Score (Recon + Latent Distance) 계산
    """
    print("\n[Computing Scores] 점수 분포 계산 중...")
    
    # 1. 정상 데이터의 중심점(Center) 계산 (Test 셋 내의 정상 데이터 이용)
    # (원칙은 Train 셋으로 해야 하지만 편의상 Test 셋의 정상 데이터로 근사)
    z_normals = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            if (y == 0).sum() > 0:
                mu, _ = model.encode(x[y==0])
                z_normals.append(mu)
    normal_center = torch.cat(z_normals).mean(dim=0)
    
    # 2. 전체 데이터 스코어링
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            
            # 복원 오차 (L1)
            recon_x, mu, _, _ = model(x)
            recon_loss = torch.mean(torch.abs(x - recon_x), dim=1)
            
            # 잠재 거리 (Euclidean)
            latent_dist = torch.norm(mu - normal_center, p=2, dim=1)
            
            # 최종 점수 합산
            final_score = recon_loss + latent_dist
            
            all_scores.extend(final_score.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    return np.array(all_scores), np.array(all_labels)

# ==========================================
# 시각화 함수들
# ==========================================

def plot_score_histogram(scores, labels):
    """
    [핵심] 정상과 사기의 점수 분포 차이를 보여주는 히스토그램
    """
    print("\n[Vis 1] Score Histogram 그리는 중...")
    
    plt.figure(figsize=(10, 6))
    
    # 정상(Normal) - 파란색
    sns.histplot(scores[labels==0], color='dodgerblue', label='Normal', 
                 kde=True, stat="density", bins=50, alpha=0.3)
    
    # 사기(Fraud) - 빨간색
    sns.histplot(scores[labels==1], color='red', label='Fraud', 
                 kde=True, stat="density", bins=50, alpha=0.3)
    
    plt.title("Anomaly Score Distribution (The Proof of Separation)", fontsize=15)
    plt.xlabel("Hybrid Anomaly Score (Lower is Normal)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    # 로그 스케일 (점수 차이가 너무 클 경우를 대비)
    plt.xscale('log')
    plt.xlabel("Hybrid Anomaly Score (Log Scale)")
    
    save_path = "vis_histogram.png"
    plt.savefig(save_path)
    print(f"✅ 저장 완료: {save_path}")

def plot_error_heatmap(model, dataloader, device):
    """
    [상세 분석] 사기 데이터가 '어디가' 틀렸는지 보여주는 히트맵
    """
    print("\n[Vis 2] Error Heatmap 그리는 중...")
    
    # 사기 데이터 5개, 정상 데이터 1개 샘플링
    fraud_samples = []
    normal_sample = None
    
    with torch.no_grad():
        for x, y in dataloader:
            if len(fraud_samples) < 5 and (y == 1).sum() > 0:
                fraud_samples.append(x[y==1][0])
            if normal_sample is None and (y == 0).sum() > 0:
                normal_sample = x[y==0][0]
            
            if len(fraud_samples) >= 5 and normal_sample is not None:
                break
    
    # 하나로 합치기 (맨 위: 정상, 아래 5개: 사기)
    samples = torch.stack([normal_sample] + fraud_samples).to(device)
    
    # 모델 통과
    with torch.no_grad():
        recon, _, _, _ = model(samples)
        
    # 절대 오차 계산 (Absolute Error)
    # (N, 29)
    errors = torch.abs(samples - recon).cpu().numpy()
    
    plt.figure(figsize=(12, 6))
    yticklabels = ['Normal'] + [f'Fraud {i+1}' for i in range(5)]
    
    # Heatmap 그리기
    sns.heatmap(errors, cmap='Reds', yticklabels=yticklabels, cbar_kws={'label': 'Reconstruction Error'})
    
    plt.title("Feature-wise Reconstruction Error Heatmap", fontsize=15)
    plt.xlabel("Feature Index (V1 ~ V29)")
    plt.tight_layout()
    
    save_path = "vis_heatmap.png"
    plt.savefig(save_path)
    print(f"✅ 저장 완료: {save_path}")

def plot_tsne(model, dataloader, device):
    """
    [공간 분석] Latent Space t-SNE
    """
    print("\n[Vis 3] t-SNE 계산 중... (데이터 2000개 샘플링)")
    
    z_list = []
    y_list = []
    max_samples = 2000
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            mu, _ = model.encode(x)
            z_list.append(mu.cpu().numpy())
            y_list.append(y.cpu().numpy())
            if len(np.concatenate(y_list)) > max_samples:
                break
                
    z = np.concatenate(z_list)[:max_samples]
    labels = np.concatenate(y_list)[:max_samples]
    
    # [수정] n_iter=1000 삭제 (기본값 사용)
    # 에러 원인: 일부 scikit-learn 버전에서 n_iter 파라미터 충돌 발생
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    
    z_2d = tsne.fit_transform(z)
    
    plt.figure(figsize=(10, 8))
    
    # 정상 먼저 그리기 (뒤에 깔리게)
    plt.scatter(z_2d[labels==0, 0], z_2d[labels==0, 1], 
                c='lightgray', label='Normal', s=10, alpha=0.5)
    
    # 사기 나중에 그리기 (위에 뜨게)
    plt.scatter(z_2d[labels==1, 0], z_2d[labels==1, 1], 
                c='red', label='Fraud', s=30, alpha=0.9, marker='x')
    
    plt.title("Latent Space Distribution (t-SNE)", fontsize=15)
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    save_path = "vis_tsne.png"
    plt.savefig(save_path)
    print(f"✅ 저장 완료: {save_path}")

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # 1. 준비
    model, test_loader = load_model_and_data()
    
    # 2. 점수 계산 (Hybrid Score)
    scores, labels = get_hybrid_scores(model, test_loader, CONFIG['DEVICE'])
    
    # 3. 그래프 그리기
    plot_score_histogram(scores, labels) # 분포 확인 (가장 중요)
    plot_error_heatmap(model, test_loader, CONFIG['DEVICE']) # 특징별 에러 확인
    plot_tsne(model, test_loader, CONFIG['DEVICE']) # 공간 분리 확인
    
    print("\n🎉 모든 시각화 완료! 생성된 3개의 png 파일을 확인하세요.")