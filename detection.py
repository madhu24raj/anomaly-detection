# for LM-TAD, markov chain edition

"""
AIS Vessel Sequence Detection & Anomaly Detection - v4 (Markov LM-TAD)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_recall_curve
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# CONSTANTS & CONFIG
# ──────────────────────────────────────────────────────────────
# Grid size for tokenization (~5.5km at equator). 
# Smaller grid = larger vocabulary, more precise but requires more data.
GRID_SIZE_DEG = 0.05 

# Laplace smoothing factor for the language model (handles unseen transitions)
ALPHA_SMOOTHING = 1.0 

# Spoofing rule physical bounds
SPOOF_DIST_KM = 10
VTYPE_MAX_SPEED = {
    "fishing":   20,
    "cargo":     28,
    "passenger": 35,
    "pleasure":  45,
    "tanker":    20,
}
DEFAULT_MAX_SPEED = 40

# ──────────────────────────────────────────────────────────────
# MATH HELPERS
# ──────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

# ──────────────────────────────────────────────────────────────
# 1. SEQUENCE TOKENIZATION
# ──────────────────────────────────────────────────────────────
def tokenize_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts continuous GPS coordinates into discrete grid cell "tokens".
    A trajectory becomes a sequence of words: Cell_A -> Cell_B -> Cell_C.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)

    # Discretize map into tokens
    df["grid_x"] = (df["lon"] / GRID_SIZE_DEG).astype(int)
    df["grid_y"] = (df["lat"] / GRID_SIZE_DEG).astype(int)
    df["token"]  = df["grid_x"].astype(str) + "_" + df["grid_y"].astype(str)

    # Find the "next word" in the sequence
    df["next_token"] = df.groupby("entity_id")["token"].shift(-1)
    
    # Time gaps and basic features for the rule layer
    df["time_gap_min"] = df.groupby("entity_id")["timestamp"].diff().dt.total_seconds().div(60).fillna(0)
    
    df["prev_lat"] = df.groupby("entity_id")["lat"].shift(1)
    df["prev_lon"] = df.groupby("entity_id")["lon"].shift(1)
    valid = df["prev_lat"].notna()

    df.loc[valid, "dist_from_prev_km"] = haversine_km(
        df.loc[valid, "lat"], df.loc[valid, "lon"],
        df.loc[valid, "prev_lat"], df.loc[valid, "prev_lon"]
    )
    df["dist_from_prev_km"] = df["dist_from_prev_km"].fillna(0)

    safe_time_gap = np.maximum(df["time_gap_min"], 1.0)
    df["implied_speed_kn"] = np.where(
        df["time_gap_min"] > 0,
        (df["dist_from_prev_km"] / 1.852) / (safe_time_gap / 60.0),
        df["sog_knots"],
    )

    return df

# ──────────────────────────────────────────────────────────────
# 2. MARKOV LANGUAGE MODEL (TRAIN & SCORE)
# ──────────────────────────────────────────────────────────────
def build_and_score_markov_lm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates P(Next_Token | Current_Token).
    Assigns a Negative Log-Likelihood (NLL) surprise score to every move.
    """
    df = df.copy()
    
    # 1. Count all historical transitions across the fleet
    transitions = df.dropna(subset=["next_token"]).groupby(["token", "next_token"]).size().reset_index(name="count")
    
    # 2. Total outbound trips from each cell
    total_outbound = transitions.groupby("token")["count"].sum().reset_index(name="total")
    transitions = transitions.merge(total_outbound, on="token")
    
    # 3. Calculate Transition Probabilities with Laplace Smoothing
    # Formula: (Count + Alpha) / (Total + Alpha * Vocab_Size)
    vocab_size = df["token"].nunique()
    
    transitions["prob"] = (transitions["count"] + ALPHA_SMOOTHING) / (
        transitions["total"] + (ALPHA_SMOOTHING * vocab_size)
    )
    
    # 4. Convert to Negative Log-Likelihood (NLL). Higher NLL = Higher Anomaly
    transitions["nll_surprise"] = -np.log(transitions["prob"])
    
    # Create O(1) lookup dictionary
    transition_dict = transitions.set_index(["token", "next_token"])["nll_surprise"].to_dict()
    
    # Default surprise for a completely unseen (novel) transition
    max_surprise = -np.log(ALPHA_SMOOTHING / (ALPHA_SMOOTHING * vocab_size))
    
    # 5. Apply scores to the dataset
    def score_transition(row):
        if pd.isna(row["next_token"]):
            return 0.0 # End of sequence
        return transition_dict.get((row["token"], row["next_token"]), max_surprise)

    df["step_surprise"] = df.apply(score_transition, axis=1)
    
    # Smooth the surprise over a rolling window (3 pings) to catch sustained weird behavior
    df["rolling_surprise"] = (
        df.groupby("entity_id")["step_surprise"]
          .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )
    
    # Normalize the LM score to for blending
    min_surp, max_surp = df["rolling_surprise"].min(), df["rolling_surprise"].max()
    df["lm_anomaly_score"] = (df["rolling_surprise"] - min_surp) / (max_surp - min_surp + 1e-9)
    
    return df

# ──────────────────────────────────────────────────────────────
# 3. HIGH PRECISION RULE LAYER
# ──────────────────────────────────────────────────────────────
def apply_precision_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retaining only the high-precision rules validated in prior runs.
    """
    df = df.copy()

    # Dark Ship (Precision = 1.00)
    df["rule_dark_ship"] = (df["time_gap_min"] > 60).astype(int)

    # Spoofing (Precision = 0.56)
    max_spd = df["vessel_type"].map(VTYPE_MAX_SPEED).fillna(DEFAULT_MAX_SPEED)
    df["rule_spoofing"] = (
        (df["implied_speed_kn"] > (max_spd * 1.5)) &
        (df["dist_from_prev_km"] > SPOOF_DIST_KM)
    ).astype(int)

    # Rule voting weight
    df["rule_vote"] = (df["rule_dark_ship"] * 0.5) + (df["rule_spoofing"] * 0.3)
    return df

# ──────────────────────────────────────────────────────────────
# 4. THRESHOLDING & EVALUATION
# ──────────────────────────────────────────────────────────────
def select_dynamic_threshold(scores: pd.Series, std_multiplier: float = 2.5) -> float:
    mean_score = scores.mean()
    std_score = scores.std()
    threshold = mean_score + (std_multiplier * std_score)
    return float(np.clip(threshold, 0.2, 0.95))

def evaluate(df: pd.DataFrame, threshold: float):
    y_true = df["is_anomalous"].astype(int)
    y_pred = (df["final_score"] >= threshold).astype(int)

    print("=" * 55)
    print("ANOMALY DETECTION — LM-TAD Classification Report")
    print(f"Threshold: {threshold:.4f}")
    print("=" * 55)
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomalous"]))
    
    prec_arr, rec_arr, thr_arr = precision_recall_curve(y_true, df["final_score"])
    f1_arr = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-9)
    best_f1 = f1_arr.max()
    best_thr = thr_arr[f1_arr.argmax()] if len(thr_arr) > 0 else threshold
    
    print(f"Current Pipeline F1: {f1_score(y_true, y_pred):.4f}")
    print(f"\nOracle Best F1 (Theoretical Limit): {best_f1:.4f} at threshold {best_thr:.4f}")

# ──────────────────────────────────────────────────────────────
# PIPELINE EXECUTION
# ──────────────────────────────────────────────────────────────
def run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    print("Step 1: Tokenizing spatial trajectories...")
    df = tokenize_trajectories(df)

    print("Step 2: Training & Scoring Markov Language Model...")
    df = build_and_score_markov_lm(df)

    print("Step 3: Applying high-precision rule votes...")
    df = apply_precision_rules(df)

    # Blend: Sequence Surprise + Precise Rules
    df["final_score"] = np.clip(df["lm_anomaly_score"] + df["rule_vote"], 0, 1)

    print("Step 4: Dynamic Threshold Selection...")
    threshold = select_dynamic_threshold(df["final_score"], std_multiplier=2.5)
    df["pred_anomalous"] = df["final_score"] >= threshold

    print("Step 5: Evaluation...\n")
    evaluate(df, threshold)
    return df

if __name__ == "__main__":
    df = pd.read_csv("ais_small.csv")
    results = run_pipeline(df)
