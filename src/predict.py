import pandas as pd
import numpy as np
import os
import joblib

def main():
    print("[PREDICT] Loading test features...")
    df_test = pd.read_csv('../data/features_test.csv')
    
    print("[PREDICT] Loading models...")
    lgbm_final = joblib.load('../models/lgbm_final.pkl')
    iso_final = joblib.load('../models/iso_final.pkl')
    meta_learner = joblib.load('../models/meta_learner.pkl')
    best_thresh = joblib.load('../models/best_threshold.pkl')
    feature_cols = joblib.load('../models/feature_names.pkl')
    
    print("[PREDICT] Selecting feature columns...")
    X_test = df_test[feature_cols]
    
    print("[PREDICT] Generating base model predictions...")
    test_proba = lgbm_final.predict_proba(X_test)[:, 1]
    test_iso = iso_final.score_samples(X_test)
    
    print("[PREDICT] Stacking meta-features...")
    meta_features = np.column_stack((test_proba, test_iso))
    
    print("[PREDICT] Generating final predictions...")
    final_proba = meta_learner.predict_proba(meta_features)[:, 1]
    
    # Add predictions to test dataframe
    df_test['final_proba'] = final_proba
    
    print("[PREDICT] Aggregating predictions per time epoch...")
    # The evaluation requires a single prediction per time stamp.
    # If any channel is strongly predicted as spoofed, the epoch is spoofed.
    epoch_preds = df_test.groupby('time')['final_proba'].max().reset_index()
    epoch_preds['Spoofed'] = (epoch_preds['final_proba'] > best_thresh).astype(int)
    epoch_preds = epoch_preds.rename(columns={'final_proba': 'Confidence'})
    
    print("[PREDICT] Formatting submission...")
    sub = pd.read_csv('../data/sample_submission.csv')
    
    # Make sure we use the same columns as required
    if 'Spoofed' in sub.columns and 'Confidence' in sub.columns:
        sub = sub.drop(columns=['Spoofed', 'Confidence'])
        
    sub = sub.merge(epoch_preds[['time', 'Spoofed', 'Confidence']], on='time', how='left')
    sub['Spoofed'] = sub['Spoofed'].fillna(0).astype(int)
    sub['Confidence'] = sub['Confidence'].fillna(0.0)
    
    os.makedirs('../outputs', exist_ok=True)
    sub.to_csv('../outputs/submission.csv', index=False)
    
    print("Submission saved to outputs/submission.csv")
    print("Predicted class distribution (per time epoch):")
    class_counts = sub['Spoofed'].value_counts()
    print(f"  Genuine (0): {class_counts.get(0, 0)}")
    print(f"  Spoofed (1): {class_counts.get(1, 0)}")

if __name__ == '__main__':
    main()
