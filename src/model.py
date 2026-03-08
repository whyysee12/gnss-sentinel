import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_curve, auc, confusion_matrix

def main():
    print("[MODEL] Loading data...")
    df = pd.read_csv('../data/features_train.csv')
    
    label_cols = ['label', 'Label', 'spoofed', 'class', 'target']
    target_col = next((c for c in label_cols if c in df.columns), 'label')
    
    exclude_cols = ['PRN', 'RX_time', 'TOW_at_current_symbol_s', 'channel', 'time', target_col]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols]
    y = df[target_col]
    
    print("[MODEL] Class Distribution:")
    print(y.value_counts())
    
    print("\n[MODEL] Starting 5-Fold Stratified CV...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_lgbm_proba = np.zeros(len(df))
    oof_iso_score = np.zeros(len(df))
    
    best_iterations = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        lgbm = LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=7,
            class_weight='balanced',
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1
        )
        
        try:
            # Newer LightGBM versions
            lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        except TypeError:
            # Older LightGBM versions
            lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
            
        best_iterations.append(lgbm.best_iteration_)
        
        # Out-of-fold probability
        oof_lgbm_proba[val_idx] = lgbm.predict_proba(X_val)[:, 1]
        
        # Isolation Forest
        iso = IsolationForest(contamination='auto', random_state=42, n_estimators=100)
        iso.fit(X_train)
        oof_iso_score[val_idx] = iso.score_samples(X_val)
        
        # Fold F1 Score
        fold_preds = (oof_lgbm_proba[val_idx] > 0.5).astype(int)
        fold_f1 = f1_score(y_val, fold_preds, average='weighted')
        print(f"[MODEL] Fold {fold} Weighted F1 (threshold 0.5): {fold_f1:.4f}")
        
    print("\n[MODEL] Threshold Optimization...")
    thresholds = np.arange(0.1, 0.91, 0.01)
    best_thresh = 0.5
    best_f1 = 0.0
    
    for t in thresholds:
        t_preds = (oof_lgbm_proba > t).astype(int)
        t_f1 = f1_score(y, t_preds, average='weighted')
        if t_f1 > best_f1:
            best_f1 = t_f1
            best_thresh = t
            
    print(f"[MODEL] Best Threshold: {best_thresh:.2f}")
    print(f"[MODEL] Best OOF Weighted F1: {best_f1:.4f}")
    
    print("\n[MODEL] Training Meta-Learner...")
    meta_features = np.column_stack((oof_lgbm_proba, oof_iso_score))
    meta_learner = LogisticRegression(class_weight='balanced', random_state=42)
    meta_learner.fit(meta_features, y)
    
    print("[MODEL] Training Final Models on ALL Data...")
    final_n_estimators = int(np.mean(best_iterations))
    
    lgbm_final = LGBMClassifier(
        n_estimators=final_n_estimators,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=7,
        class_weight='balanced',
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1
    )
    lgbm_final.fit(X, y)
    
    iso_final = IsolationForest(contamination='auto', random_state=42, n_estimators=100)
    iso_final.fit(X)
    
    print("[MODEL] Saving Models...")
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../reports', exist_ok=True)
    
    joblib.dump(lgbm_final, '../models/lgbm_final.pkl')
    joblib.dump(iso_final, '../models/iso_final.pkl')
    joblib.dump(meta_learner, '../models/meta_learner.pkl')
    joblib.dump(best_thresh, '../models/best_threshold.pkl')
    joblib.dump(feature_cols, '../models/feature_names.pkl')
    
    print("[MODEL] Generating Reports...")
    
    # 1. Feature Importance
    importances = lgbm_final.feature_importances_
    indices = np.argsort(importances)[::-1][:30]
    top_features = [feature_cols[i] for i in indices]
    top_importances = importances[indices]
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=top_importances, y=top_features, palette='vlag')
    plt.title('Top 30 Feature Importances')
    plt.tight_layout()
    plt.savefig('../reports/feature_importance.png', dpi=150)
    plt.close()
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y, oof_lgbm_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='coral', lw=2, label=f'OOF ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('../reports/roc_curve.png', dpi=150)
    plt.close()
    
    # 3. Confusion Matrix
    final_preds = (oof_lgbm_proba > best_thresh).astype(int)
    cm = confusion_matrix(y, final_preds)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (Threshold={best_thresh:.2f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('../reports/confusion_matrix.png', dpi=150)
    plt.close()
    
    print("===============================")
    print("GNSS-SENTINEL TRAINING COMPLETE")
    print(f"OOF Weighted F1: {best_f1:.4f}")
    print(f"Best Threshold: {best_thresh:.2f}")
    print("===============================")

if __name__ == '__main__':
    main()
