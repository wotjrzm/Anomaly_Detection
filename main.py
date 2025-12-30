import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score
from tqdm import tqdm
from dataset import get_dataloaders
from model import TransformerVAE, LossFunction

# ==========================================
# 1. Configuration
# ==========================================
CONFIG = {
    'PROJECT_NAME': 'Credit_Fraud_Hybrid_Score',
    'SEED': 42,
    'EPOCHS': 10,
    'BATCH_SIZE': 64,
    'LR': 1e-4,
    'LATENT_DIM': 16,
    'CONTRASTIVE_WEIGHT': 0.1, # Step 2 설정 유지
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ==========================================
# 2. Helper Functions
# ==========================================
def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        
        # Forward
        recon_x, mu, logvar, z = model(x)
        
        # Loss Calculation
        loss, r_loss, k_loss, c_loss = criterion(recon_x, x, mu, logvar, z, y)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Logging
        total_loss += loss.item()
        pbar.set_postfix({'Recon': r_loss.item(), 'CL': c_loss.item()})
        
    return total_loss / len(dataloader)

def get_normal_center(model, dataloader, device):
    """
    학습 데이터 중 '정상(Label=0)'인 데이터들의 잠재 벡터(z) 중심을 계산합니다.
    """
    model.eval()
    vectors = []
    
    print("Calculating Normal Center...")
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Extracting Vectors"):
            x = x.to(device)
            if (y == 0).sum() > 0:
                mu, _ = model.encode(x[y==0].to(device))
                vectors.append(mu.cpu())
    
    
    # Calculate Mean and Covariance
    center = torch.cat(vectors, dim=0).mean(dim=0)
    cov = torch.cov(torch.cat(vectors, dim=0).T)
    
    # Inverse Covariance (add epsilon for stability)
    cov_inv = torch.linalg.inv(cov + torch.eye(cov.size(0)).to(cov.device) * 1e-5)
    
    return center.to(device), cov_inv.to(device)

def evaluate_hybrid(model, dataloader, normal_center, cov_inv, device):
    """
    Hybrid Scoring: Reconstruction Error + Mahalanobis Distance
    """
    model.eval()
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating"):
            x = x.to(device)
            y = y.to(device)
            
            # 1. Reconstruction Score (L1)
            recon_x, mu, _, _ = model(x)
            recon_score = torch.mean(torch.abs(x - recon_x), dim=1)
            
            # 2. Latent Distance Score (Mahalanobis)
            # dist = sqrt( (z-mu)^T * S^-1 * (z-mu) )
            diff = mu - normal_center
            # Batch-wise Mahalanobis: diag(diff @ cov_inv @ diff.T)
            # More efficient: sum((diff @ cov_inv) * diff, dim=1)
            
            left_term = torch.matmul(diff, cov_inv) # (B, D)
            mahalanobis_sq = torch.sum(left_term * diff, dim=1) # (B,)
            latent_dist = torch.sqrt(mahalanobis_sq)
            
            # 3. Final Hybrid Score
            final_score = recon_score + latent_dist
            
            all_scores.extend(final_score.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    return np.array(all_scores), np.array(all_labels)

# ==========================================
# 3. Main Execution
# ==========================================
if __name__ == "__main__":
    print(f"Project: {CONFIG['PROJECT_NAME']} | Device: {CONFIG['DEVICE']}")
    seed_everything(CONFIG['SEED'])

    # 1. Data Load
    train_loader, test_loader, input_dim = get_dataloaders(CONFIG)
    
    # 2. Model Init
    model = TransformerVAE(
        input_dim=input_dim, 
        latent_dim=CONFIG['LATENT_DIM']
    ).to(CONFIG['DEVICE'])
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['LR'])
    
    criterion = LossFunction(contrastive_weight=CONFIG['CONTRASTIVE_WEIGHT'])

    # 3. Training Loop
    print("\nStarting Training...")
    for epoch in range(1, CONFIG['EPOCHS'] + 1):
        avg_loss = train(model, train_loader, optimizer, criterion, CONFIG['DEVICE'])
        print(f"Epoch [{epoch}/{CONFIG['EPOCHS']}] Loss: {avg_loss:.4f}")

    # 4. Save Model
    torch.save(model.state_dict(), "best_model.pth")
    
    normal_center, normal_cov_inv = get_normal_center(model, train_loader, CONFIG['DEVICE'])
    print(f"Normal Center Calculated. Shape: {normal_center.shape}")

    # 5. Hybrid Evaluation
    print("\nStarting Hybrid Evaluation...")
    scores, labels = evaluate_hybrid(model, test_loader, normal_center, normal_cov_inv, CONFIG['DEVICE'])

    # 6. Metrics
    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)

    print(f"\n[Final Result (Hybrid Score)]")
    print(f"AUROC : {auroc:.4f}")
    print(f"AUPRC : {auprc:.4f} (기대하세요!)")

    # 7. Threshold
    precisions, recalls, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]
    
    print(f"Best F1-Score: {best_f1:.4f}")