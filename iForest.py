import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from dataset import get_dataloaders

# ==========================================
# 1. 설정
# ==========================================
CONFIG = {
    'BATCH_SIZE': 2048, 
    'DEVICE': 'cpu',
    'SEED': 42 
}

def run_baseline():
    print("데이터 로드 및 전처리 중...")
    
    _, test_loader, _ = get_dataloaders(CONFIG)
    
    X_test = []
    y_test = []
    
    for x, y in test_loader:
        X_test.append(x.numpy())
        y_test.append(y.numpy())
        
    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)
    
    print(f"✅ 데이터 준비 완료: {X_test.shape}")

    # ==========================================
    # 2. Isolation Forest 학습 및 예측
    # ==========================================
    print("🌲 Isolation Forest 학습 중... (Baseline)")
    
    # contamination: 사기 데이터 비율 (약 0.0017)
    iso_forest = IsolationForest(
        n_estimators=100, 
        contamination=0.0017, 
        random_state=42, 
        n_jobs=-1
    )
    
    iso_forest.fit(X_test) 
    
    scores = -iso_forest.score_samples(X_test)

    # ==========================================
    # 3. 성능 평가 (AUROC, AUPRC, F1)
    # ==========================================
    auroc = roc_auc_score(y_test, scores)
    auprc = average_precision_score(y_test, scores)
    
    precisions, recalls, thresholds = precision_recall_curve(y_test, scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_f1 = np.max(f1_scores)

    print("\n" + "="*40)
    print(f"📊 [Baseline Result: Isolation Forest]")
    print(f" - AUROC : {auroc:.4f}")
    print(f" - AUPRC : {auprc:.4f}")
    print(f" - Best F1 : {best_f1:.4f}")
    print("="*40)

if __name__ == "__main__":
    run_baseline()