# 🛰️ GNSS-SENTINEL: Physics-Informed GNSS Spoofing Detection

> *"Every other team built a pattern matcher. We built a physics consistency checker. Their model learned what spoofing looks like in this dataset. Our model knows what authentic GNSS signals must look like — always — because physics doesn't change between datasets."*

---

## 📌 Table of Contents
1. [Problem Framing](#problem-framing)
2. [Our Core Insight](#our-core-insight)
3. [Why Physics Features?](#why-physics-features)
4. [Feature Engineering](#feature-engineering)
5. [Model Architecture](#model-architecture)
6. [Handling Class Imbalance](#handling-class-imbalance)
7. [Validation Strategy](#validation-strategy)
8. [Results](#results)
9. [Repository Structure](#repository-structure)
10. [How to Run](#how-to-run)
11. [Production Readiness](#production-readiness)

---

## Problem Framing

### What is GNSS Spoofing?

Global Navigation Satellite Systems (GPS, GLONASS, Galileo, BeiDou) work by triangulating signals from multiple satellites. A receiver computes its position by measuring the time-of-flight of signals from at least 4 satellites simultaneously.

A **spoofing attack** transmits fake satellite signals that are carefully crafted to look authentic. The receiver, deceived, computes a false position, velocity, or time. Real-world consequences include:

- Aircraft navigating to wrong coordinates
- Autonomous drones being redirected mid-flight
- Financial systems receiving incorrect timestamps (critical for trade sequencing)
- Ships being guided into exclusion zones or hazardous waters

### Why Is Detection Hard?

Spoofed signals are designed to **mimic genuine signals** at the physical layer. A naive approach — checking if signal strength looks normal — fails because sophisticated spoofers can replicate individual signal characteristics. The challenge is that **no spoofer can simultaneously satisfy all physical constraints** that govern authentic GNSS signals.

This is the key insight our solution exploits.

---

## Our Core Insight

Most spoofing detection approaches ask: *"Does this signal look like spoofing?"*

We ask a fundamentally different question: *"Is this signal physically consistent with what it must be, given the laws of signal physics?"*

A GNSS signal is not just a number — it is the product of:

1. **Orbital mechanics** — the satellite's position dictates exact Doppler shift and geometric arrival angle
2. **Signal propagation physics** — the correlator outputs (Early, Late, Prompt) must follow a precise mathematical relationship called the S-curve
3. **Temporal continuity** — carrier phase must evolve smoothly; discontinuities indicate signal injection
4. **Constellation geometry** — all visible satellites at a given moment must be mutually geometrically consistent

A spoofer can fake one of these. They cannot simultaneously fake all of them. Our model encodes all four constraints as measurable features.

---

## Why Physics Features?

### The Correlator Triangle (Most Teams Miss This)

Inside every GNSS receiver, three correlators track the incoming signal:
- **Early (EC):** Slightly ahead of expected signal timing
- **Prompt (PC):** Aligned with expected signal timing  
- **Late (LC):** Slightly behind expected signal timing

For a **genuine signal**, the S-curve tracking loop enforces a precise mathematical relationship:
```
EC ≈ LC  (symmetric around the prompt correlator)
(EC - LC) / (EC + LC + ε) ≈ 0
```

A **spoofed signal** distorts this symmetry because the attacker cannot perfectly replicate the internal correlator dynamics without knowing the receiver's exact chip spacing and tracking loop bandwidth.

### The Doppler–Pseudorange Relationship

The rate of change of pseudorange (distance to satellite) must equal the Doppler-derived velocity:
```
dPseudorange/dt ≈ Doppler_Hz × λ_L1
where λ_L1 = 0.1903 m (L1 carrier wavelength)
```

Any deviation from this relationship reveals an injected signal where position and velocity are not physically consistent.

### Carrier Phase Continuity

Carrier phase must change smoothly over time. Spoofing introduces the signal mid-stream, causing a **phase jump** — a discontinuity that violates the continuous-wave nature of satellite signals.

### Constellation-Level Consistency

At any timestamp, multiple satellites are visible. If one satellite's signal characteristics are outliers relative to all others observed at the same moment, it indicates targeted single-satellite spoofing — a pattern invisible when looking at each satellite in isolation.

---

## Feature Engineering

All features are derived from the 12 raw dataset columns using physical principles:

### Raw Features (12 columns)
| Feature | Physical Meaning |
|---|---|
| `PRN` | Satellite identifier (pseudo-random noise code ID) |
| `Carrier_Doppler_hz` | Frequency shift due to relative satellite-receiver motion |
| `Pseudorange_m` | Measured distance to satellite via signal travel time |
| `RX_time` | Timestamp at receiver |
| `TOW_at_current_symbol_s` | Satellite transmit time (Time of Week) |
| `Carrier_phase_cycles` | Integrated phase of carrier wave |
| `EC` | Early correlator output |
| `LC` | Late correlator output |
| `PC` | Prompt correlator output (reference) |
| `PIP` | Signal quality metric |
| `PQP` | Signal quality metric |
| `TCD` | Code delay tracking |
| `CN0` | Carrier-to-noise ratio |

### Engineered Physics Features (20+ derived features)

#### Group 1: Correlator Integrity
```python
correlator_asymmetry    = (EC - LC) / (PC + ε)
S_curve_distortion      = (EC - LC) / (EC + LC + ε)
correlator_ratio_EL     = EC / (LC + ε)
correlator_power        = EC² + LC² + PC²
```
**Why:** Authentic signals have near-zero S_curve_distortion. Spoofed signals introduce tracking loop distortion detectable in this ratio.

#### Group 2: Doppler–Pseudorange Consistency
```python
pseudorange_rate        = dPseudorange/dt  (per PRN, temporal diff)
doppler_ms              = Carrier_Doppler_hz × 0.1903
doppler_pseudorange_residual = pseudorange_rate - doppler_ms
```
**Why:** This residual must be near zero for authentic signals. Spoofed signals introduce velocity-position inconsistencies.

#### Group 3: Carrier Phase Continuity
```python
phase_delta             = dCarrier_phase/dt  (per PRN)
phase_acceleration      = d²Carrier_phase/dt²
phase_jump_flag         = |phase_delta| > adaptive_threshold
```
**Why:** Carrier phase is continuous for genuine signals. Signal injection causes detectable phase jumps (cycle slips).

#### Group 4: CN0 Temporal Stability
```python
CN0_rolling_mean        = rolling mean of CN0 (window=5, per PRN)
CN0_rolling_std         = rolling std of CN0 (window=5, per PRN)
CN0_deviation           = CN0 - CN0_rolling_mean
```
**Why:** Spoofed signals often have unnaturally stable or unnaturally volatile CN0 patterns compared to genuine satellite signals that fluctuate naturally with atmospheric and geometric changes.

#### Group 5: Timing Consistency
```python
timing_offset           = RX_time - TOW_at_current_symbol_s
timing_offset_deviation = timing_offset - rolling_mean(timing_offset)
```
**Why:** The offset between receiver time and satellite transmit time must remain stable. Meaconing attacks (capture and rebroadcast) introduce microsecond-level timing deviations detectable here.

#### Group 6: Constellation-Level Outlier Score
```python
epoch_median_CN0        = median CN0 across all PRNs at same RX_time
CN0_vs_constellation    = CN0 - epoch_median_CN0
epoch_std_doppler       = std(Carrier_Doppler_hz) across all PRNs at RX_time
```
**Why:** This is the most powerful group for detecting targeted single-satellite spoofing. If one satellite's metrics deviate from the rest of the constellation at the same moment, it is flagged.

#### Group 7: Signal Quality Cross-Checks
```python
quality_ratio           = PIP / (PQP + ε)
quality_product         = PIP × PQP
TCD_residual            = TCD - rolling_mean(TCD, per PRN)
```
**Why:** PIP and PQP encode signal tracking quality. Their cross-product captures joint degradation that individual metrics miss.

---

## Model Architecture

We use a **two-stage ensemble** combining temporal and instantaneous anomaly detection:

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT LAYER                               │
│   12 raw features + 20+ physics-derived features            │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┼───────────────┐
         │           │               │
         ▼           ▼               ▼
  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐
  │  LightGBM   │  │   LightGBM   │  │ Isolation      │
  │  (full      │  │  (temporal   │  │ Forest         │
  │  features)  │  │  lag + roll) │  │ (unsupervised  │
  │             │  │              │  │  anomaly score)│
  └──────┬──────┘  └──────┬───────┘  └───────┬────────┘
         │                │                   │
         └────────────────┴───────────────────┘
                          │
                          ▼
              ┌────────────────────┐
              │  Stacking          │
              │  Meta-Learner      │
              │  (Logistic Reg.)   │
              │  with calibrated   │
              │  probabilities     │
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │  Threshold         │
              │  Optimization      │
              │  (maximize         │
              │  Weighted F1)      │
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │  Final Prediction  │
              │  (0 = Genuine,     │
              │   1 = Spoofed)     │
              └────────────────────┘
```

### Why This Architecture?

| Component | Role | Why Chosen |
|---|---|---|
| **LightGBM (full features)** | Captures non-linear feature interactions | Best-in-class for tabular data, handles missing values |
| **LightGBM (temporal)** | Captures how signal evolves over time | Spoofing attacks have temporal signatures (onset, drift) |
| **Isolation Forest** | Unsupervised anomaly baseline | Generalizes to novel attack types not in training data |
| **Stacking Meta-Learner** | Fuses all signals optimally | Learns which model to trust under which conditions |
| **Threshold Optimization** | Maximizes Weighted F1 specifically | Default threshold (0.5) is suboptimal for imbalanced classes |

---

## Handling Class Imbalance

The dataset has imbalanced classes (more genuine signals than spoofed). We address this with three complementary strategies:

### Strategy 1: Algorithmic Balancing
```python
LGBMClassifier(class_weight='balanced')
# Internally weights minority class inversely proportional to frequency
```

### Strategy 2: Threshold Optimization
```python
# Default threshold 0.5 is not optimal for imbalanced + Weighted F1
best_threshold = argmax over t of: weighted_f1(y_val, predictions > t)
# Typically found in range 0.3–0.45 for imbalanced spoofing data
```

### Strategy 3: Stratified Cross-Validation
```python
StratifiedKFold(n_splits=5)
# Ensures each fold has same class ratio as full dataset
# Prevents optimistic validation scores from lucky splits
```

### Why Not SMOTE/Oversampling?
SMOTE creates synthetic samples by interpolating between existing minority class examples. For GNSS spoofing, this would generate physically invalid signal combinations that do not represent real attacks. We prefer algorithm-level balancing to preserve physical validity of all training samples.

---

## Validation Strategy

### Core Principle: No Data Leakage

GNSS data is temporal and per-satellite. Data leakage risks:

1. **Temporal leakage:** Using future signal state to predict current label
2. **PRN leakage:** Rolling statistics that bleed across the train/test boundary

**Our mitigations:**
- All rolling/diff features are computed strictly with `min_periods=1` and no lookahead
- Train/test split is time-ordered (not random shuffle)
- Isolation Forest is fit on training data only; applied to test
- Threshold optimization is done on held-out validation fold, not test set

### Cross-Validation Protocol
```
Full Training Data
├── Fold 1: Train on [2,3,4,5] → Validate on [1]
├── Fold 2: Train on [1,3,4,5] → Validate on [2]
├── Fold 3: Train on [1,2,4,5] → Validate on [3]
├── Fold 4: Train on [1,2,3,5] → Validate on [4]
└── Fold 5: Train on [1,2,3,4] → Validate on [5]

Final Model: Trained on ALL training data
Threshold: Optimized on out-of-fold predictions
```

---

## Results

| Model | Weighted F1 (CV) | Notes |
|---|---|---|
| Baseline LightGBM (raw features only) | ~0.XX | Benchmark |
| + Physics features | +0.0X | Correlator + Doppler features |
| + Temporal features | +0.0X | Lag + rolling window features |
| + Isolation Forest meta-feature | +0.0X | Unsupervised anomaly score |
| + Threshold optimization | +0.0X | Best threshold vs default 0.5 |
| **Full GNSS-Sentinel Ensemble** | **~0.XX** | **Final submission** |

*(Exact numbers populated after training run)*

---

## Repository Structure

```
gnss-sentinel/
├── README.md                          ← This file
├── requirements.txt                   ← All dependencies
│
├── data/
│   └── .gitkeep                       ← Dataset NOT committed
│
├── notebooks/
│   ├── 01_EDA.ipynb                   ← Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb   ← Physics feature derivation
│   └── 03_modeling.ipynb              ← Training, evaluation, submission
│
├── src/
│   ├── features.py                    ← All physics feature functions
│   ├── model.py                       ← Training pipeline + ensemble
│   ├── predict.py                     ← Inference on test set
│   └── utils.py                       ← Shared helpers
│
├── outputs/
│   └── submission.csv                 ← Final predictions (test set)
│
└── reports/
    └── architecture_diagram.png       ← System diagram
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place dataset files
```
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

### 3. Generate features + train + predict
```bash
python src/features.py     # generates features_train.csv, features_test.csv
python src/model.py        # trains ensemble, saves models
python src/predict.py      # generates outputs/submission.csv
```

### 4. Or run end-to-end notebook
```bash
jupyter notebook notebooks/03_modeling.ipynb
```

### Requirements
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
lightgbm>=3.3.0
matplotlib>=3.6.0
seaborn>=0.12.0
jupyter>=1.0.0
```

---

## Production Readiness

### What Would Make This Deployable in Real Infrastructure?

1. **Real-time streaming pipeline:** Replace batch CSV processing with a Kafka stream consumer ingesting live GNSS observables at 1–10 Hz. The physics features are all computable in rolling windows with sub-millisecond latency.

2. **Per-receiver calibration:** The multipath profile and CN0 baseline vary by physical location. Production deployment would run a 1-hour calibration phase per new receiver installation to learn site-specific baselines.

3. **Multi-receiver consensus:** Deploy 3+ receivers at 50–200m spacing. A Byzantine-fault-tolerant consensus layer confirms attacks only when multiple nodes agree — eliminating false positives from local interference.

4. **Ionospheric oracle integration:** Cross-reference observed signal delays with NASA CDDIS IONEX maps (ionospheric electron content) to catch the most sophisticated spoofing attacks that replicate Doppler and correlator behavior but cannot fake atmospheric propagation physics.

5. **Federated threat intelligence:** Each deployed node contributes anonymized anomaly reports to a shared threat feed, building a global real-time map of GNSS spoofing activity — similar to VirusTotal for cybersecurity threats.

---

## Team

Built for **Kaizen 2026 | GNSS Anti-Spoofing Hackathon**
Organized by **NyneOS Technologies × EES × ARIES IIT Delhi**

---

*The atmosphere is the cryptographic key. You cannot fake the sky.*
