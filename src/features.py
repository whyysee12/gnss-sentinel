import pandas as pd
import numpy as np
import os

def engineer_features(df, threshold_phase=None):
    df = df.copy()
    
    # Standardize column names if necessary
    col_map = {'TOW': 'TOW_at_current_symbol_s', 'Carrier_phase': 'Carrier_phase_cycles'}
    df = df.rename(columns=col_map)
    
    # Coerce numerical columns to handle bad rows
    num_cols = ['Carrier_Doppler_hz', 'Pseudorange_m', 'RX_time', 'TOW_at_current_symbol_s',
                'Carrier_phase_cycles', 'EC', 'LC', 'PC', 'PIP', 'PQP', 'TCD', 'CN0']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Drop corrupted rows containing NaNs in essential columns
    df = df.dropna(subset=[c for c in num_cols if c in df.columns])
    
    # Sort data by PRN and RX_time
    df = df.sort_values(by=['PRN', 'RX_time']).reset_index(drop=True)
    
    print("[FEATURES] Engineering Group 1: Correlator Integrity...")
    df['correlator_asymmetry'] = (df['EC'] - df['LC']) / (df['PC'] + 1e-9)
    df['S_curve_distortion'] = (df['EC'] - df['LC']) / (df['EC'] + df['LC'] + 1e-9)
    df['correlator_ratio_EL'] = df['EC'] / (df['LC'] + 1e-9)
    df['correlator_power'] = df['EC']**2 + df['LC']**2 + df['PC']**2
    df['EC_PC_ratio'] = df['EC'] / (df['PC'] + 1e-9)
    df['LC_PC_ratio'] = df['LC'] / (df['PC'] + 1e-9)
    
    print("[FEATURES] Engineering Group 2: Doppler-Pseudorange Consistency...")
    prn_group = df.groupby('PRN')
    df['pseudorange_rate'] = prn_group['Pseudorange_m'].diff() / prn_group['RX_time'].diff()
    df['doppler_ms'] = df['Carrier_Doppler_hz'] * 0.1903
    df['doppler_pseudorange_residual'] = df['pseudorange_rate'] - df['doppler_ms']
    df['doppler_pseudorange_residual_abs'] = abs(df['doppler_pseudorange_residual'])
    
    print("[FEATURES] Engineering Group 3: Carrier Phase Continuity...")
    df['phase_delta'] = prn_group['Carrier_phase_cycles'].diff()
    # To do phase_delta diff again grouped by PRN, we use groupby again since phase_delta is now a column
    df['phase_acceleration'] = df.groupby('PRN')['phase_delta'].diff()
    df['phase_delta_abs'] = abs(df['phase_delta'])
    
    if threshold_phase is None:
        threshold_phase = df['phase_delta_abs'].quantile(0.99)
        print(f"[FEATURES] Adaptive threshold for phase jumps set to {threshold_phase:.4f}")
        
    df['phase_jump_flag'] = (df['phase_delta_abs'] > threshold_phase).astype(int)
    
    print("[FEATURES] Engineering Group 4: CN0 Temporal Stability...")
    df['CN0_rolling_mean'] = prn_group['CN0'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    df['CN0_rolling_std'] = prn_group['CN0'].transform(lambda x: x.rolling(window=5, min_periods=1).std())
    df['CN0_deviation'] = df['CN0'] - df['CN0_rolling_mean']
    df['CN0_zscore'] = df['CN0_deviation'] / (df['CN0_rolling_std'] + 1e-9)
    
    print("[FEATURES] Engineering Group 5: Timing Consistency...")
    df['timing_offset'] = df['RX_time'] - df['TOW_at_current_symbol_s']
    df['timing_offset_rolling_mean'] = prn_group['timing_offset'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    df['timing_offset_deviation'] = df['timing_offset'] - df['timing_offset_rolling_mean']
    df['timing_offset_deviation_abs'] = abs(df['timing_offset_deviation'])
    
    print("[FEATURES] Engineering Group 6: Constellation-level Outlier Score...")
    rxtime_group = df.groupby('RX_time')
    epoch_median_CN0 = rxtime_group['CN0'].transform('median')
    epoch_std_CN0 = rxtime_group['CN0'].transform('std')
    df['CN0_vs_constellation'] = df['CN0'] - epoch_median_CN0
    df['CN0_constellation_zscore'] = df['CN0_vs_constellation'] / (epoch_std_CN0 + 1e-9)
    
    epoch_median_doppler = rxtime_group['Carrier_Doppler_hz'].transform('median')
    epoch_std_doppler = rxtime_group['Carrier_Doppler_hz'].transform('std')
    df['doppler_vs_constellation'] = df['Carrier_Doppler_hz'] - epoch_median_doppler
    df['doppler_constellation_zscore'] = df['doppler_vs_constellation'] / (epoch_std_doppler + 1e-9)
    
    df['epoch_n_satellites'] = rxtime_group['PRN'].transform('count')
    
    print("[FEATURES] Engineering Group 7: Signal Quality Cross-Checks...")
    df['quality_ratio'] = df['PIP'] / (df['PQP'] + 1e-9)
    df['quality_product'] = df['PIP'] * df['PQP']
    df['quality_sum'] = df['PIP'] + df['PQP']
    df['TCD_rolling_mean'] = prn_group['TCD'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    df['TCD_residual'] = df['TCD'] - df['TCD_rolling_mean']
    df['TCD_residual_abs'] = abs(df['TCD_residual'])
    
    print("[FEATURES] Engineering Group 8: Lag Features...")
    cols_to_lag = ['CN0', 'Carrier_Doppler_hz', 'S_curve_distortion', 'correlator_asymmetry', 'CN0_deviation']
    # prn_group needs to be recreated if we rely on newly added columns, but we will just group by again
    new_prn_group = df.groupby('PRN')
    for col in cols_to_lag:
        df[f'lag1_{col}'] = new_prn_group[col].shift(1)
        df[f'lag2_{col}'] = new_prn_group[col].shift(2)
        
    print("[FEATURES] Cleaning up NaN and Inf values...")
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)
    
    return df, threshold_phase

def main():
    os.makedirs('../data', exist_ok=True)
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../outputs', exist_ok=True)
    os.makedirs('../reports', exist_ok=True)
    
    print("[FEATURES] Loading data...")
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    
    print("[FEATURES] Processing Train Set...")
    train_features, threshold = engineer_features(train)
    
    print("[FEATURES] Processing Test Set...")
    test_features, _ = engineer_features(test, threshold_phase=threshold)
    
    # Ensure label column remains in train and not test
    # Find label column dynamically if 'label' is not the exact name, though we assume 'label' or similar
    possible_labels = ['label', 'Label', 'spoofed', 'class']
    target_col = next((col for col in possible_labels if col in train.columns), None)
    if not target_col:
        try:
            sample_sub = pd.read_csv('../data/sample_submission.csv')
            overlap = set(train.columns).intersection(sample_sub.columns)
            potential = [c for c in overlap if c.lower() in ('label', 'spoofed', 'class', 'target')]
            if potential:
                target_col = potential[0]
            else:
                target_col = 'label'
        except:
            target_col = 'label'
            
    if target_col in test_features.columns:
        test_features = test_features.drop(columns=[target_col])
        
    print(f"[FEATURES] Shape of train feature matrix: {train_features.shape}")
    print(f"[FEATURES] Shape of test feature matrix: {test_features.shape}")
    print("[FEATURES] Feature names:")
    print(list(train_features.columns))
    
    train_features.to_csv('../data/features_train.csv', index=False)
    test_features.to_csv('../data/features_test.csv', index=False)
    
    print("[FEATURES] Done.")

if __name__ == '__main__':
    main()
