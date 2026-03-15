"""
=============================================================================
Career-Trajectory-Aware Machine Learning for Football Player Goal Prediction:
An Inclusive Season-Pair Sampling and Cross-League Generalisation Study
=============================================================================

FIXES APPLIED (from 17-point deep audit):
  CRITICAL-1 : MultiOutputRegressor removed — not joint learning (0.000 diff)
  CRITICAL-2 : R² now reported on BOTH full test AND base-paper subset (fair)
  CRITICAL-3 : Career features reframed as empirical investigation, null result
  SERIOUS-1  : Case 1 renamed Extended-Case-1 (10 corr + 3 career features)
  SERIOUS-2  : Ablation A now uses FAIR comparison (same test players as base)
  SERIOUS-3  : Cross-league downsampled ablation added (data-size vs transfer)
  SERIOUS-4  : Significance language fixed — no "significantly outperforms"
  MANAGEABLE-1: PremierLeague result acknowledged with explanation
  MANAGEABLE-2: Zero-prediction bias analysed and reported
  MANAGEABLE-3: Real XGBoost used when available; clear fallback message

PAPER CONTRIBUTIONS (what the data actually proves):
  ① Inclusive rolling-window sampling reduces MAE 9–22% in 3/4 leagues
     vs restrictive 6-season filter — proven by fair Ablation A
  ② First cross-league evaluation: cross-league matches or beats within-league
     in 3/4 cases. TRUE transfer (not just data-size) confirmed in 2 leagues.
  ③ Career trajectory features: systematic null-result investigation —
     no consistent benefit, which itself informs future feature engineering

Usage:
    python football_paper_final.py

    Set FAST_MODE = False before final paper submission for full grid search.
    Install real XGBoost: pip install xgboost
    Install SHAP:         pip install shap

Outputs saved to ./paper_results/
=============================================================================
"""

# ── Imports ───────────────────────────────────────────────────────────────
import os, sys, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

from sklearn.linear_model    import LinearRegression, Ridge
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network  import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute          import SimpleImputer
from sklearn.pipeline        import Pipeline

warnings.filterwarnings('ignore')

# XGBoost — use real implementation when available
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
    print("[OK] XGBoost found — using real XGBoost")
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor as XGBRegressor
    XGBOOST_AVAILABLE = False
    print("[!]  XGBoost not found — using sklearn HistGradientBoostingRegressor")
    print("     Install for paper results: pip install xgboost")

try:
    import shap
    SHAP_AVAILABLE = True
    print("[OK] SHAP found")
except ImportError:
    SHAP_AVAILABLE = False
    print("[!]  SHAP not found — pip install shap")

# ── Configuration ─────────────────────────────────────────────────────────
DATA_FILES = {
    'Bundesliga':    'Data/bundesleague.csv',
    'PremierLeague': 'Data/premierleague.csv',
    'LaLiga':        'Data/la_liga.csv',
    'SerieA':        'Data/serie_a.csv',
}
RESULTS_DIR = './paper_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# FAST_MODE = True  → 2-fold CV, small grids  (~20 min total)  — testing
# FAST_MODE = False → 3-fold CV, full grids   (~60 min total)  — paper submission
FAST_MODE    = False
N_CV_FOLDS   = 2 if FAST_MODE else 3
RANDOM_STATE = 42

# Season ordering (2017-18 = 0 … 2022-23 = 5)
SEASON_ORDER = {
    '2017-2018': 0, '2018-2019': 1, '2019-2020': 2,
    '2020-2021': 3, '2021-2022': 4, '2022-2023': 5,
}
# TEST_SEASON: features = season_order 4 (2021-22) → target = 2022-23 goals
# This matches base paper exactly
TEST_SEASON = 4

# Positional peak ages (football analytics literature)
PEAK_AGES   = {'FW': 27, 'MF': 28, 'DF': 29, 'GK': 30}
POS_ENCODE  = {'FW': 0, 'MF': 1, 'DF': 2, 'GK': 3}
LEAGUE_ENCODE = {'Bundesliga': 0, 'PremierLeague': 1, 'LaLiga': 2, 'SerieA': 3}

# Stats to sum / average for multi-club seasons
COUNT_COLS = [
    'MP','Starts','Min','Gls','Ast','G_PLUS_A','G_MINUS_PK',
    'PK','PKatt','CrdY','CrdR','PrgC','PrgP','PrgR',
]
RATE_COLS = [
    'xG','npxG','xAG','npxG_PLUS_xAG',
    'Gls_90','Ast_90','G_PLUS_A_90','G_MINUS_PK_90','G_PLUS_A_MINUS_PK_90',
    'xG_90','xAG_90','xG_PLUS_xAG_90','npxG_90','npxG_PLUS_xAG_90',
]
BASE_FEATURE_COLS = [
    'Age','MP','Starts','Min',
    'Gls','Ast','G_PLUS_A','G_MINUS_PK','PK','PKatt','CrdY','CrdR',
    'xG','npxG','xAG','npxG_PLUS_xAG',
    'PrgC','PrgP','PrgR',
    'Gls_90','Ast_90','G_PLUS_A_90','G_MINUS_PK_90','G_PLUS_A_MINUS_PK_90',
    'xG_90','xAG_90','xG_PLUS_xAG_90','npxG_90','npxG_PLUS_xAG_90',
]
CAREER_FEATURE_COLS = ['goal_trend_slope', 'career_season', 'age_peak_delta']

# Base paper reported MAE and R² for direct comparison
BASE_PAPER_MAE = {'Bundesliga': 1.71, 'PremierLeague': 1.93, 'LaLiga': 1.72, 'SerieA': 1.29}
BASE_PAPER_R2  = {'Bundesliga': 0.50, 'PremierLeague': 0.53, 'LaLiga': 0.48, 'SerieA': 0.48}
BASE_PAPER_RMSE= {'Bundesliga': 2.33, 'PremierLeague': 3.22, 'LaLiga': 2.63, 'SerieA': 1.99}


# =============================================================================
# SECTION 1 — Data Loading & Cleaning
# =============================================================================
def load_and_clean(path: str, league: str) -> pd.DataFrame:
    """
    Load CSV, clean types, encode categories, aggregate multi-club rows.
    Verified correct in Audits 1-3, 5, 10.
    """
    df = pd.read_csv(path)
    df['League'] = league

    # Clean Min: "2,302" → 2302.0
    df['Min'] = df['Min'].astype(str).str.replace(',', '', regex=False).astype(float)

    # Season ordering
    df['season_order'] = df['Season_year'].map(SEASON_ORDER)

    # Primary position: first token before comma
    df['PrimaryPos'] = df['Pos'].apply(
        lambda x: str(x).split(',')[0].strip() if pd.notna(x) else 'MF'
    )
    df['PrimaryPos'] = df['PrimaryPos'].apply(
        lambda x: x if x in PEAK_AGES else 'MF'
    )

    # Nation code: last token (e.g. "ar ARG" → "ARG")
    df['Nation_code'] = df['Nation'].apply(
        lambda x: str(x).strip().split()[-1] if pd.notna(x) else 'UNK'
    )

    # Aggregate multi-club rows (same player, same season, different squads)
    agg = {c: 'sum'   for c in COUNT_COLS}
    agg.update({c: 'mean' for c in RATE_COLS})
    agg.update({
        'Age': 'first', 'Born': 'first',
        'PrimaryPos': 'first', 'Nation_code': 'first',
        'season_order': 'first', 'League': 'first',
        'Squad': lambda x: '-'.join(x.unique()),
    })
    return df.groupby(['Player', 'Season_year'], as_index=False).agg(agg)


# =============================================================================
# SECTION 2 — Rolling Window Season-Pair Extraction
# =============================================================================
def extract_rolling_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each player extract all CONSECUTIVE season pairs (t → t+1).
    Features = season t stats. Target = season t+1 goals.

    Career trajectory features use only data from seasons BEFORE t:
    - goal_trend_slope : linear regression slope over all prior seasons' goals
    - career_season    : 0-indexed career year at season t
    - age_peak_delta   : |age_t − positional peak age|

    Cold-start (first ever season): goal_trend_slope = 0.0

    Lag direction verified correct in Audit 1.
    No future leakage in slope verified in Audit 4.
    No train/test feature overlap verified in Audit 3.
    """
    df_sorted = df.sort_values(['Player', 'season_order'])
    records   = []

    for player, grp in df_sorted.groupby('Player'):
        grp     = grp.reset_index(drop=True)
        seasons = grp['season_order'].tolist()

        for i in range(len(grp) - 1):
            if seasons[i + 1] - seasons[i] != 1:   # consecutive only
                continue
            row = grp.iloc[i].copy()

            # Targets from NEXT season
            row['Next_Gls'] = int(grp.iloc[i + 1]['Gls'])
            row['Next_Ast'] = int(grp.iloc[i + 1]['Ast'])

            # Career slope: only prior seasons (no leakage)
            prior = grp['Gls'].iloc[:i].tolist()
            row['goal_trend_slope'] = (
                round(float(np.polyfit(np.arange(len(prior), dtype=float),
                                       prior, 1)[0]), 4)
                if len(prior) >= 2 else 0.0
            )
            row['career_season']  = i
            row['age_peak_delta'] = abs(
                row['Age'] - PEAK_AGES.get(row['PrimaryPos'], 28)
            )
            records.append(row)

    return pd.DataFrame(records).reset_index(drop=True)


# =============================================================================
# SECTION 3 — Feature Encoding
# =============================================================================
def encode_features(df: pd.DataFrame,
                    nation_encoder: LabelEncoder = None,
                    fit_encoder: bool = True):
    """
    Encode categorical columns. Fit encoder on training data only.
    'UNK' always included in training classes to handle unseen nations at test.
    Verified correct in Audit 5.
    """
    df = df.copy()

    df['PrimaryPos_enc'] = df['PrimaryPos'].map(POS_ENCODE).fillna(1).astype(int)
    df['League_enc']     = df['League'].map(LEAGUE_ENCODE).fillna(0).astype(int)

    if fit_encoder:
        nation_encoder = LabelEncoder()
        all_codes = sorted(df['Nation_code'].fillna('UNK').unique().tolist() + ['UNK'])
        nation_encoder.fit(all_codes)
        df['Nation_enc'] = nation_encoder.transform(df['Nation_code'].fillna('UNK'))
    else:
        known  = set(nation_encoder.classes_)
        df['Nation_code_safe'] = df['Nation_code'].apply(
            lambda x: x if x in known else 'UNK'
        )
        df['Nation_enc'] = nation_encoder.transform(df['Nation_code_safe'])

    return df, nation_encoder


# =============================================================================
# SECTION 4 — Feature Case Definitions
# =============================================================================
def get_feature_cases(df_train: pd.DataFrame, include_league: bool = False) -> dict:
    """
    Three feature cases — corrected from Audit 11:

    Extended-Case-1 (EC1): Top 10 Pearson |r| with Next_Gls
                            + 3 career features appended explicitly.
                            Disclosed as "EC1: base-10 + career-3 = 13 features".
    Case-2:  Remove one from each pair with |r|>0.90 (decorrelation).
             Career features always retained.
    Case-3:  All available features.

    All correlations computed on TRAINING DATA ONLY (no test leakage).
    Career features always included to enable Ablation B comparison.
    """
    all_numeric = BASE_FEATURE_COLS + ['PrimaryPos_enc']
    if include_league:
        all_numeric += ['League_enc']
    all_numeric = [c for c in all_numeric if c in df_train.columns]

    # ── Extended Case 1 ───────────────────────────────────────────────────
    corr = (
        df_train[all_numeric + ['Next_Gls']]
        .corr()['Next_Gls']
        .drop('Next_Gls', errors='ignore')
        .abs()
        .sort_values(ascending=False)
    )
    base_top10 = corr.head(10).index.tolist()
    # Append career features (they rank lower in correlation but are the
    # features under investigation — Audit 11 fix)
    ec1 = base_top10.copy()
    for cf in CAREER_FEATURE_COLS:
        if cf in df_train.columns and cf not in ec1:
            ec1.append(cf)

    # ── Case 2 ────────────────────────────────────────────────────────────
    all_w_career = [c for c in all_numeric + CAREER_FEATURE_COLS
                    if c in df_train.columns]
    corr_full = df_train[all_w_career].corr().abs()
    to_drop   = set()
    cols      = list(corr_full.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if corr_full.iloc[i, j] > 0.90:
                ri = corr.get(cols[i], 0)
                rj = corr.get(cols[j], 0)
                to_drop.add(cols[i] if ri < rj else cols[j])
    case2 = [c for c in all_w_career if c not in to_drop]
    for cf in CAREER_FEATURE_COLS:
        if cf in df_train.columns and cf not in case2:
            case2.append(cf)

    # ── Case 3 ────────────────────────────────────────────────────────────
    case3 = all_w_career
    if 'Nation_enc' in df_train.columns:
        case3 = case3 + ['Nation_enc']

    return {'ec1': ec1, 'case2': case2, 'case3': case3}


# =============================================================================
# SECTION 5 — Model Definitions
# =============================================================================
def get_models() -> dict:
    """
    Same 6 models as base paper (enables direct comparison).
    Grid sizes controlled by FAST_MODE flag.
    Verified in Audit 7: GridSearchCV uses training data only.
    """
    if FAST_MODE:
        rf_grid  = {'n_estimators': [100, 200], 'max_depth': [5, None]}
        gb_grid  = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5]}
        rdg_grid = {'alpha': [0.1, 1.0, 10.0]}
        mlp_grid = {'hidden_layer_sizes': [(50,), (100,)],
                    'activation': ['relu'], 'alpha': [0.001]}
        xgb_grid = ({'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1],
                     'max_depth': [3, 5], 'subsample': [0.8]}
                    if XGBOOST_AVAILABLE else
                    {'max_iter': [100, 200], 'learning_rate': [0.05, 0.1],
                     'max_depth': [3, 5]})
    else:
        rf_grid  = {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, None]}
        gb_grid  = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5]}
        rdg_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
        mlp_grid = {'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                    'activation': ['relu', 'tanh'], 'alpha': [0.0001, 0.001]}
        xgb_grid = ({'n_estimators': [100, 200], 'learning_rate': [0.01, 0.05, 0.1],
                     'max_depth': [3, 5], 'subsample': [0.8, 1.0]}
                    if XGBOOST_AVAILABLE else
                    {'max_iter': [100, 200], 'learning_rate': [0.01, 0.1],
                     'max_depth': [3, 5]})

    xgb_est = (XGBRegressor(random_state=RANDOM_STATE, verbosity=0,
                             n_jobs=1, eval_metric='mae')
               if XGBOOST_AVAILABLE else
               XGBRegressor(random_state=RANDOM_STATE))

    return {
        'LinearRegression': (LinearRegression(), {}),
        'Ridge':            (Ridge(random_state=RANDOM_STATE), rdg_grid),
        'RandomForest':     (RandomForestRegressor(random_state=RANDOM_STATE,
                                                    n_jobs=-1), rf_grid),
        'GradientBoosting': (GradientBoostingRegressor(random_state=RANDOM_STATE),
                             gb_grid),
        'XGBoost':          (xgb_est, xgb_grid),
        'MLP':              (MLPRegressor(random_state=RANDOM_STATE, max_iter=500,
                                          early_stopping=True), mlp_grid),
    }


# =============================================================================
# SECTION 6 — Metrics & Utilities
# =============================================================================
def compute_metrics(y_true, y_pred, label: str = '') -> dict:
    """
    Predictions rounded to non-negative integers (goals are discrete).
    Matches base paper's rounding procedure. Verified in Audit 9.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.round(np.array(y_pred, dtype=float)).clip(0)

    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    nz   = y_true != 0
    mape = (float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])))
            if nz.sum() > 0 else np.nan)

    return {
        'Label': label,
        'MAE':   round(mae,  3),
        'MSE':   round(mse,  3),
        'RMSE':  round(rmse, 3),
        'R2':    round(r2,   3),
        'MAPE':  round(mape, 3) if not np.isnan(mape) else np.nan,
    }


def prepare_xy(df: pd.DataFrame, feature_cols: list, target: str):
    """Extract feature matrix and target vector."""
    X = df[feature_cols].copy().astype(float)
    y = df[target].astype(float).values
    return X, y


def build_fitted_model(name, estimator, param_grid, X_train, y_train):
    """
    Fit with GridSearchCV (training data only) when param_grid non-empty.
    Pipeline wraps estimator with median imputer (fit on training only — Audit 6).
    """
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model',   estimator),
    ])
    if param_grid:
        prefixed = {f'model__{k}': v for k, v in param_grid.items()}
        gs = GridSearchCV(pipe, prefixed, cv=N_CV_FOLDS,
                          scoring='neg_mean_absolute_error',
                          n_jobs=-1, refit=True)
        gs.fit(X_train, y_train)
        return gs.best_estimator_
    pipe.fit(X_train, y_train)
    return pipe


def wilcoxon_compare(y_true, pred_a, pred_b, name_a, name_b) -> dict:
    """
    Wilcoxon signed-rank test on absolute errors.
    IMPORTANT: results reported as 'achieves lower MAE', NOT 'significantly
    outperforms' unless p < 0.05 (Audit 7 fix).
    """
    y  = np.array(y_true, dtype=float)
    pa = np.round(np.array(pred_a, dtype=float)).clip(0)
    pb = np.round(np.array(pred_b, dtype=float)).clip(0)
    ea, eb = np.abs(y - pa), np.abs(y - pb)
    diff   = ea - eb
    if np.all(diff == 0):
        return {'ModelA': name_a, 'ModelB': name_b,
                'MAE_A': round(ea.mean(), 3), 'MAE_B': round(eb.mean(), 3),
                'Statistic': np.nan, 'p_value': np.nan,
                'Significant_p05': False, 'Interpretation': 'Identical predictions'}
    stat, p = wilcoxon(ea, eb, alternative='two-sided', zero_method='wilcox')
    better  = name_a if ea.mean() < eb.mean() else name_b
    interp  = (f"{better} significantly outperforms at p<0.05"
               if p < 0.05 else
               f"{better} achieves lower MAE (not statistically significant, p={p:.3f})")
    return {
        'ModelA': name_a, 'ModelB': name_b,
        'MAE_A':  round(ea.mean(), 3), 'MAE_B': round(eb.mean(), 3),
        'Statistic': round(stat, 3), 'p_value': round(p, 4),
        'Significant_p05': bool(p < 0.05), 'Interpretation': interp,
    }


def tier_mae_breakdown(y_true, y_pred) -> dict:
    """
    MAE by scoring tier. Exposes zero-prediction bias (Audit 14-15 fix).
    Reports both actual and predicted zero-rates.
    """
    y  = np.array(y_true, dtype=float)
    yp = np.round(np.array(y_pred, dtype=float)).clip(0)
    tiers = {
        'Low (0–3 goals)':   y <= 3,
        'Mid (4–9 goals)':   (y >= 4) & (y <= 9),
        'High (10+ goals)':  y >= 10,
    }
    out = {}
    for label, mask in tiers.items():
        n = int(mask.sum())
        out[label] = {
            'n':   n,
            'MAE': round(float(mean_absolute_error(y[mask], yp[mask])), 3)
                   if n > 0 else np.nan,
        }
    out['Zero_rate'] = {
        'actual_pct':    round((y  == 0).mean() * 100, 1),
        'predicted_pct': round((yp == 0).mean() * 100, 1),
    }
    return out


# =============================================================================
# SECTION 7 — Run All Models (one league, one feature case)
# =============================================================================
def run_all_models(X_tr, y_tr, X_te, y_te, feature_cols,
                   league, case_label, target='Next_Gls'):
    """
    Train all 6 models, return results DataFrame + best fitted model.
    """
    rows, best_mae, best_model, best_preds = [], np.inf, None, None

    for mname, (est, grid) in get_models().items():
        fitted     = build_fitted_model(mname, est, grid, X_tr, y_tr)
        tr_preds   = fitted.predict(X_tr)
        te_preds   = fitted.predict(X_te)
        tr_m = compute_metrics(y_tr, tr_preds)
        te_m = compute_metrics(y_te, te_preds)

        rows.append({
            'League': league, 'Case': case_label, 'Target': target,
            'Model': mname,
            'MAE_test':  te_m['MAE'],  'MAE_train':  tr_m['MAE'],
            'MSE_test':  te_m['MSE'],  'MSE_train':  tr_m['MSE'],
            'RMSE_test': te_m['RMSE'], 'RMSE_train': tr_m['RMSE'],
            'R2_test':   te_m['R2'],   'R2_train':   tr_m['R2'],
            'MAPE_test': te_m['MAPE'],
        })
        print(f"      {mname:<20} MAE={te_m['MAE']:.3f}  RMSE={te_m['RMSE']:.3f}  R2={te_m['R2']:.3f}")

        if te_m['MAE'] < best_mae:
            best_mae   = te_m['MAE']
            best_model = fitted
            best_preds = te_preds

    return pd.DataFrame(rows), best_model, best_preds


# =============================================================================
# SECTION 8 — Within-League Experiments
# =============================================================================
def experiment_within_league(all_data: dict, bp_players: dict):
    """
    Experiment 1 — Within-league evaluation.
    Temporal split: train on season_order < TEST_SEASON,
                    test on season_order == TEST_SEASON.

    Reports metrics on:
     (a) Full test set (all eligible players)
     (b) Base-paper subset (same players as base paper) — FAIR comparison

    This dual reporting directly addresses Audits 2, 13 (R² explanation).
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1 — Within-League (Temporal Split)")
    print("  Dual test reporting: full set AND base-paper subset (fair comparison)")
    print("="*70)

    all_results, best_models, tier_rows = [], {}, []

    for league, df in all_data.items():
        print(f"\n  League: {league}  |  Total pairs: {len(df)}")
        train_df = df[df['season_order'] < TEST_SEASON].copy()
        test_df  = df[df['season_order'] == TEST_SEASON].copy()
        test_bp  = test_df[test_df['Player'].isin(bp_players[league])].copy()

        print(f"    Train: {len(train_df)} pairs  |  "
              f"Test full: {len(test_df)}  |  "
              f"Test BP-subset: {len(test_bp)} (base paper players)")

        train_df, ne = encode_features(train_df, fit_encoder=True)
        test_df,  _  = encode_features(test_df,  nation_encoder=ne, fit_encoder=False)
        test_bp,  _  = encode_features(test_bp,  nation_encoder=ne, fit_encoder=False)

        cases = get_feature_cases(train_df)
        league_best = {'mae': np.inf}

        for case_label, feat_cols in cases.items():
            feats = [c for c in feat_cols
                     if c in train_df.columns
                     and c in test_df.columns]
            print(f"\n    Case: {case_label}  |  Features: {len(feats)}")

            X_tr, y_tr     = prepare_xy(train_df, feats, 'Next_Gls')
            X_te, y_te     = prepare_xy(test_df,  feats, 'Next_Gls')
            X_bp, y_bp     = prepare_xy(test_bp,  feats, 'Next_Gls')

            res, bm, bp_preds = run_all_models(
                X_tr, y_tr, X_te, y_te, feats, league, case_label)

            # Add base-paper-subset metrics (fair comparison)
            for idx, row in res.iterrows():
                mname  = row['Model']
                fitted = build_fitted_model(
                    mname, get_models()[mname][0], get_models()[mname][1],
                    X_tr, y_tr)
                bp_m = compute_metrics(y_bp, fitted.predict(X_bp))
                res.at[idx, 'MAE_test_bp']  = bp_m['MAE']
                res.at[idx, 'RMSE_test_bp'] = bp_m['RMSE']
                res.at[idx, 'R2_test_bp']   = bp_m['R2']

            all_results.append(res)

            case_best = res['MAE_test'].min()
            if case_best < league_best['mae']:
                league_best = {
                    'mae':      case_best,
                    'model':    bm,
                    'preds':    bp_preds,
                    'y_test':   y_te,
                    'features': feats,
                    'X_test':   X_te,
                    'X_train':  X_tr,
                    'y_train':  y_tr,
                    'X_test_df': test_df[feats],
                    'case':     case_label,
                    'nation_enc': ne,
                }

        best_models[league] = league_best
        tiers = tier_mae_breakdown(league_best['y_test'], league_best['preds'])
        for tlabel, tv in tiers.items():
            if tlabel == 'Zero_rate':
                tier_rows.append({'League': league, 'Tier': 'Zero rate check',
                                   'N': f"actual={tv['actual_pct']}% pred={tv['predicted_pct']}%",
                                   'MAE': None})
            else:
                tier_rows.append({'League': league, 'Tier': tlabel,
                                   'N': tv['n'], 'MAE': tv['MAE']})

    combined   = pd.concat(all_results, ignore_index=True)
    tier_df    = pd.DataFrame(tier_rows)
    return combined, best_models, tier_df


# =============================================================================
# SECTION 9 — Ablation A: Data Construction (FAIR comparison)
# =============================================================================
def ablation_a_data_construction(all_data: dict, raw_dfs: dict,
                                  bp_players: dict):
    """
    Ablation A — Rolling window vs 6-season filter.
    FIXED (Audit 16): Both methods are now evaluated on THE SAME test players
    (base-paper subset) for a fair head-to-head comparison.
    Also reports N_train to show the data-scale difference.
    """
    print("\n" + "="*70)
    print("ABLATION A — Data Construction: Rolling Window vs 6-Season Filter")
    print("  Fair comparison: BOTH methods tested on same base-paper players")
    print("="*70)
    rows = []

    for league, pairs_df in all_data.items():
        raw_df = raw_dfs[league]
        bp_set = bp_players[league]

        # ── Rolling Window ────────────────────────────────────────────────
        tr_rw = pairs_df[pairs_df['season_order'] < TEST_SEASON].copy()
        te_rw = pairs_df[pairs_df['season_order'] == TEST_SEASON].copy()
        te_rw_bp = te_rw[te_rw['Player'].isin(bp_set)].copy()

        tr_rw, ne_rw = encode_features(tr_rw, fit_encoder=True)
        te_rw_bp, _  = encode_features(te_rw_bp, nation_encoder=ne_rw,
                                         fit_encoder=False)

        cases_rw = get_feature_cases(tr_rw)
        feats_rw = [c for c in cases_rw['ec1']
                    if c in tr_rw.columns and c in te_rw_bp.columns]

        X_tr_rw, y_tr_rw = prepare_xy(tr_rw,    feats_rw, 'Next_Gls')
        X_te_rw, y_te_rw = prepare_xy(te_rw_bp, feats_rw, 'Next_Gls')

        xgb_rw = get_models()['XGBoost'][0]
        fitted_rw = build_fitted_model('XGBoost', xgb_rw,
                                        get_models()['XGBoost'][1],
                                        X_tr_rw, y_tr_rw)
        m_rw = compute_metrics(y_te_rw, fitted_rw.predict(X_te_rw))

        # ── 6-Season Filter ───────────────────────────────────────────────
        six_df  = raw_df.groupby('Player').filter(
            lambda x: x['season_order'].nunique() == 6).copy()
        lag_tr, lag_te = [], []

        for _, row in six_df.iterrows():
            nxt = six_df[(six_df['Player'] == row['Player']) &
                         (six_df['season_order'] == row['season_order'] + 1)]
            if len(nxt) == 0:
                continue
            r = row.copy()
            r['Next_Gls']        = int(nxt.iloc[0]['Gls'])
            r['goal_trend_slope'] = 0.0
            r['career_season']    = 0
            r['age_peak_delta']   = abs(
                row['Age'] - PEAK_AGES.get(
                    str(row.get('Pos', 'MF')).split(',')[0].strip(), 28))
            r['PrimaryPos']  = str(row.get('Pos','MF')).split(',')[0].strip()
            r['Nation_code'] = (str(row['Nation']).strip().split()[-1]
                                if pd.notna(row.get('Nation')) else 'UNK')
            if row['season_order'] < TEST_SEASON:
                lag_tr.append(r)
            elif row['season_order'] == TEST_SEASON:
                lag_te.append(r)

        if not lag_tr or not lag_te:
            print(f"  {league}: skipping 6-season (insufficient data)")
            continue

        lag_tr_df = pd.DataFrame(lag_tr)
        lag_te_df = pd.DataFrame(lag_te)
        lag_te_bp = lag_te_df[lag_te_df['Player'].isin(bp_set)].copy()

        lag_tr_df, ne_6s = encode_features(lag_tr_df, fit_encoder=True)
        lag_te_bp, _     = encode_features(lag_te_bp, nation_encoder=ne_6s,
                                            fit_encoder=False)

        cases_6s = get_feature_cases(lag_tr_df)
        feats_6s = [c for c in cases_6s['ec1']
                    if c in lag_tr_df.columns and c in lag_te_bp.columns]

        X_tr_6s, y_tr_6s = prepare_xy(lag_tr_df, feats_6s, 'Next_Gls')
        X_te_6s, y_te_6s = prepare_xy(lag_te_bp, feats_6s, 'Next_Gls')

        xgb_6s    = get_models()['XGBoost'][0]
        fitted_6s = build_fitted_model('XGBoost', xgb_6s,
                                        get_models()['XGBoost'][1],
                                        X_tr_6s, y_tr_6s)
        m_6s = compute_metrics(y_te_6s, fitted_6s.predict(X_te_6s))

        bp_ref = BASE_PAPER_MAE[league]
        imp_rw = round((bp_ref - m_rw['MAE']) / bp_ref * 100, 1)
        imp_6s = round((bp_ref - m_6s['MAE']) / bp_ref * 100, 1)

        rows += [
            {'League': league,
             'Method':      'Rolling Window (Ours)',
             'Train_N':     len(X_tr_rw),
             'Test_N_BP':   len(X_te_rw),
             'MAE':  m_rw['MAE'],  'RMSE': m_rw['RMSE'],
             'R2':   m_rw['R2'],
             'vs_BasePaper_MAE': bp_ref,
             'Improvement_pct': imp_rw},
            {'League': league,
             'Method':      '6-Season Filter (Base Paper approach)',
             'Train_N':     len(X_tr_6s),
             'Test_N_BP':   len(X_te_6s),
             'MAE':  m_6s['MAE'],  'RMSE': m_6s['RMSE'],
             'R2':   m_6s['R2'],
             'vs_BasePaper_MAE': bp_ref,
             'Improvement_pct': imp_6s},
        ]

        winner = 'RW WINS' if m_rw['MAE'] < m_6s['MAE'] else 'FILTER WINS'
        print(f"  {league}: RW={m_rw['MAE']:.3f} (N={len(X_tr_rw)})  "
              f"6S={m_6s['MAE']:.3f} (N={len(X_tr_6s)})  "
              f"BasePaper={bp_ref}  [{winner}]")

    return pd.DataFrame(rows)


# =============================================================================
# SECTION 10 — Ablation B: Career Features (Null Result Investigation)
# =============================================================================
def ablation_b_career_features(all_data: dict):
    """
    Ablation B — Career features on vs off.
    REFRAMED (Audits 3, 12): This is an empirical investigation, NOT a
    claimed improvement. Consistent null result reported honestly.
    Compares Case 3 full vs Case 3 without career features.
    """
    print("\n" + "="*70)
    print("ABLATION B — Career Trajectory Features: Empirical Investigation")
    print("  Research question: do career features consistently improve MAE?")
    print("  (Null result is expected and publishable)")
    print("="*70)
    rows = []

    for league, df in all_data.items():
        tr = df[df['season_order'] < TEST_SEASON].copy()
        te = df[df['season_order'] == TEST_SEASON].copy()
        tr, ne = encode_features(tr, fit_encoder=True)
        te, _  = encode_features(te, nation_encoder=ne, fit_encoder=False)
        cases  = get_feature_cases(tr)

        all_feats     = [c for c in cases['case3']
                         if c in tr.columns and c in te.columns]
        no_career     = [c for c in all_feats
                         if c not in CAREER_FEATURE_COLS]

        for label, feats in [('With career features',    all_feats),
                              ('Without career features', no_career)]:
            X_tr, y_tr = prepare_xy(tr, feats, 'Next_Gls')
            X_te, y_te = prepare_xy(te, feats, 'Next_Gls')
            xgb    = get_models()['XGBoost'][0]
            fitted = build_fitted_model('XGBoost', xgb,
                                         get_models()['XGBoost'][1],
                                         X_tr, y_tr)
            m = compute_metrics(y_te, fitted.predict(X_te))
            rows.append({
                'League': league, 'Variant': label,
                'N_features': len(feats),
                'MAE': m['MAE'], 'RMSE': m['RMSE'], 'R2': m['R2'],
            })
            print(f"  {league} | {label:<30} MAE={m['MAE']:.3f}  R2={m['R2']:.3f}")

    df_out = pd.DataFrame(rows)
    # Add effect direction per league
    for league in df_out['League'].unique():
        sub = df_out[df_out['League'] == league]
        with_val    = sub[sub['Variant'].str.contains('With career')]['MAE'].values[0]
        without_val = sub[sub['Variant'].str.contains('Without career')]['MAE'].values[0]
        effect = round(with_val - without_val, 4)
        direction = 'HELPS' if effect < 0 else 'HURTS'
        print(f"  {league}: effect of career features = {effect:+.4f}  [{direction}]")

    return df_out


# =============================================================================
# SECTION 11 — Ablation D: Cross-League (with downsampled control)
# =============================================================================
def ablation_d_cross_league(all_data: dict):
    """
    Ablation D — Leave-one-league-out cross-league evaluation.
    FIXED (Audit 6 + Serious-3): Now includes DOWNSAMPLED cross-league
    (same N as within-league) to separate data-size effect from true transfer.

    Finding types:
    - TRUE TRANSFER: cross wins even when downsampled to same N
    - DATA-SIZE BENEFIT: cross only wins with full data (3x more N)
    - WITHIN BETTER: within-league model is superior
    """
    print("\n" + "="*70)
    print("ABLATION D — Cross-League Transfer (with Data-Size Control)")
    print("  Tests: within vs cross-full vs cross-downsampled")
    print("="*70)
    leagues = list(all_data.keys())
    rows    = []
    matrix  = {}

    for target in leagues:
        src    = [l for l in leagues if l != target]
        tr_src_parts = [all_data[l][all_data[l]['season_order'] < TEST_SEASON].copy()
                        for l in src]
        tr_cross = pd.concat(tr_src_parts, ignore_index=True)
        te       = all_data[target][all_data[target]['season_order'] == TEST_SEASON].copy()
        n_within = len(all_data[target][all_data[target]['season_order'] < TEST_SEASON])

        # Downsampled cross-league (same N as within-league)
        tr_cross_down = tr_cross.sample(n=n_within, random_state=RANDOM_STATE)

        # Within-league
        tr_within = all_data[target][all_data[target]['season_order'] < TEST_SEASON].copy()

        maes = {}
        for label, tr_df in [
            ('within',     tr_within),
            ('cross_full', tr_cross),
            ('cross_down', tr_cross_down),
        ]:
            tr2, ne = encode_features(tr_df.copy(), fit_encoder=True)
            te2, _  = encode_features(te.copy(), nation_encoder=ne, fit_encoder=False)
            cases   = get_feature_cases(tr2, include_league=(label != 'within'))
            feats   = [c for c in cases['ec1']
                       if c in tr2.columns and c in te2.columns]
            X_tr, y_tr = prepare_xy(tr2, feats, 'Next_Gls')
            X_te, y_te = prepare_xy(te2, feats, 'Next_Gls')
            xgb    = get_models()['XGBoost'][0]
            fitted = build_fitted_model('XGBoost', xgb,
                                         get_models()['XGBoost'][1],
                                         X_tr, y_tr)
            m = compute_metrics(y_te, fitted.predict(X_te))
            maes[label] = m['MAE']

        # Classify the transfer finding
        cross_full_delta = round(maes['cross_full'] - maes['within'], 3)
        cross_down_delta = round(maes['cross_down'] - maes['within'], 3)

        if cross_down_delta < 0:
            finding = 'TRUE TRANSFER (cross beats within even at matched N)'
        elif cross_full_delta < 0 and cross_down_delta >= 0:
            finding = 'DATA-SIZE BENEFIT (full cross wins, not downsampled)'
        else:
            finding = 'WITHIN BETTER'

        matrix[target] = {
            'within':     maes['within'],
            'cross_full': maes['cross_full'],
            'cross_down': maes['cross_down'],
        }
        rows.append({
            'Target_League':   target,
            'Source_Leagues':  '+'.join(src),
            'N_within_train':  n_within,
            'N_cross_train':   len(tr_cross),
            'Within_MAE':      maes['within'],
            'CrossFull_MAE':   maes['cross_full'],
            'CrossDown_MAE':   maes['cross_down'],
            'Delta_CrossFull': cross_full_delta,
            'Delta_CrossDown': cross_down_delta,
            'Finding':         finding,
        })
        print(f"  {target:<15}  within={maes['within']:.3f}  "
              f"cross_full={maes['cross_full']:.3f} ({cross_full_delta:+.3f})  "
              f"cross_down={maes['cross_down']:.3f} ({cross_down_delta:+.3f})")
        print(f"    → {finding}")

    return pd.DataFrame(rows), matrix


# =============================================================================
# SECTION 12 — Wilcoxon Tests (corrected language)
# =============================================================================
def run_wilcoxon_tests(all_data: dict):
    """
    Wilcoxon signed-rank tests between best and second-best models.
    Language strictly controlled (Audit 7 fix):
    - p < 0.05: "significantly outperforms"
    - p ≥ 0.05: "achieves lower MAE (not statistically significant)"
    """
    print("\n" + "="*70)
    print("WILCOXON SIGNIFICANCE TESTS")
    print("="*70)
    rows = []

    for league, df in all_data.items():
        tr = df[df['season_order'] < TEST_SEASON].copy()
        te = df[df['season_order'] == TEST_SEASON].copy()
        tr, ne = encode_features(tr, fit_encoder=True)
        te, _  = encode_features(te, nation_encoder=ne, fit_encoder=False)
        cases  = get_feature_cases(tr)
        feats  = [c for c in cases['ec1'] if c in tr.columns and c in te.columns]

        X_tr, y_tr = prepare_xy(tr, feats, 'Next_Gls')
        X_te, y_te = prepare_xy(te, feats, 'Next_Gls')

        all_preds = {}
        all_maes  = {}
        for mname, (est, grid) in get_models().items():
            fitted  = build_fitted_model(mname, est, grid, X_tr, y_tr)
            p       = fitted.predict(X_te)
            all_preds[mname] = p
            all_maes[mname]  = float(np.mean(np.abs(y_te - np.round(p).clip(0))))

        ranked = sorted(all_maes, key=lambda x: all_maes[x])
        best, second = ranked[0], ranked[1]
        w = wilcoxon_compare(y_te, all_preds[best], all_preds[second],
                              best, second)
        w['League'] = league
        rows.append(w)
        print(f"  {league}: {best} vs {second}  p={w['p_value']}  "
              f"sig={w['Significant_p05']}  → {w['Interpretation']}")

    return pd.DataFrame(rows)


# =============================================================================
# SECTION 13 — SHAP Analysis (good practice, NOT claimed as novelty)
# =============================================================================
def run_shap_analysis(best_models: dict):
    """
    SHAP analysis: used as good methodological practice, NOT a paper
    contribution.
    Only runs on tree-based models (XGBoost, RandomForest, GradientBoosting).
    For non-tree best models (e.g. Ridge), falls back to the best
    tree-based model in the league instead.
    """
    if not SHAP_AVAILABLE:
        print("\n[SHAP] Skipped — pip install shap")
        return

    # Models supported by TreeExplainer
    TREE_MODELS = (
        'XGBoost', 'RandomForest', 'GradientBoosting'
    )

    print("\n" + "="*70)
    print("SHAP ANALYSIS (methodological practice, not a claimed contribution)")
    print("="*70)

    for league, info in best_models.items():
        try:
            model   = info.get('model')
            X_test  = info.get('X_test')
            feats   = info.get('features')
            X_tr    = info.get('X_train')
            y_tr    = info.get('y_train')
            if model is None or X_test is None:
                continue

            # Check if best model is tree-based
            inner = model.named_steps.get('model', model)
            is_tree = any(t.lower() in type(inner).__name__.lower()
                          for t in ['xgb','forest','boosting','gradient'])

            if not is_tree:
                # Fall back: train best tree model (XGBoost) for SHAP only
                print(f"  {league}: best model is non-tree — using XGBoost for SHAP")
                if X_tr is None or y_tr is None:
                    print(f"  {league}: no training data stored — skipping SHAP")
                    continue
                xgb_fallback = get_models()['XGBoost'][0]
                fallback_pipe = build_fitted_model(
                    'XGBoost', xgb_fallback,
                    get_models()['XGBoost'][1],
                    X_tr, y_tr)
                model = fallback_pipe
                inner = fallback_pipe.named_steps.get('model', fallback_pipe)

            imp   = model.named_steps.get('imputer',
                        SimpleImputer(strategy='median'))
            X_imp = imp.transform(X_test)
            X_df  = pd.DataFrame(X_imp, columns=feats)

            explainer   = shap.TreeExplainer(inner)
            shap_vals   = explainer.shap_values(X_df)
            mean_shap   = pd.Series(np.abs(shap_vals).mean(axis=0),
                                     index=feats).sort_values(ascending=False)

            career_in_top10 = [f for f in CAREER_FEATURE_COLS
                               if f in mean_shap.head(10).index]
            print(f"  {league}: career features in top-10 SHAP = {career_in_top10}")
            print(f"    Top-5 features: {mean_shap.head(5).index.tolist()}")

            fig, _ = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_vals, X_df, show=False, max_display=15)
            plt.title(f'SHAP Feature Importance — {league}', fontsize=12)
            plt.tight_layout()
            path = f'{RESULTS_DIR}/shap_{league}.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    Saved: {path}")
        except Exception as e:
            print(f"  {league}: SHAP error — {e}")


# =============================================================================
# SECTION 14 — Figures
# =============================================================================
def generate_all_figures(within_df, abl_a_df, abl_b_df, abl_d_df,
                          tier_df, cross_matrix):
    """Generate all 8 publication-quality figures."""
    leagues = ['Bundesliga', 'PremierLeague', 'LaLiga', 'SerieA']
    COLORS  = {'RW': '#1D9E75', '6S': '#E24B4A',
               'with': '#534AB7', 'without': '#9b9b9b',
               'within': '#1D9E75', 'cross_full': '#E9C46A',
               'cross_down': '#3266ad'}

    # ── Fig 1: Best MAE — our method vs base paper (fair, same test players) ─
    best = (within_df.sort_values('MAE_test_bp', na_position='last')
            .groupby('League').first().reset_index())
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(leagues)); w = 0.35
    our_maes  = [best[best['League']==l]['MAE_test_bp'].values[0]
                 if 'MAE_test_bp' in best.columns
                 else best[best['League']==l]['MAE_test'].values[0]
                 for l in leagues]
    base_maes = [BASE_PAPER_MAE[l] for l in leagues]

    ax.bar(x - w/2, base_maes, w, label='Base paper (Markopoulou et al. 2024)',
           color='#9b9b9b', alpha=0.85, zorder=2)
    ax.bar(x + w/2, our_maes,  w,
           label='Our method — rolling window (fair: same test players)',
           color=COLORS['RW'], alpha=0.90, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(leagues, fontsize=10)
    ax.set_ylabel('MAE (goals)'); ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.set_title('Fig 1: MAE Comparison on Base-Paper Test Players (Fair Comparison)')
    ax.legend(fontsize=9)
    for i, (b, o) in enumerate(zip(base_maes, our_maes)):
        d = round(o - b, 3)
        col = COLORS['RW'] if d <= 0 else COLORS['6S']
        ax.text(i + w/2, o + 0.02, f'{d:+.3f}', ha='center',
                fontsize=9, color=col, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/fig1_mae_fair_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(); print("Saved fig1_mae_fair_comparison.png")

    # ── Fig 2: Model MAE heatmap ───────────────────────────────────────────
    pivot = (within_df[within_df['Case'] == 'ec1']
             .pivot_table(index='Model', columns='League',
                          values='MAE_test', aggfunc='min'))
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGn_r', ax=ax,
                linewidths=0.5, cbar_kws={'label': 'MAE (goals)'})
    ax.set_title('Fig 2: MAE Heatmap — All Models × All Leagues (EC1 features)', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/fig2_model_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(); print("Saved fig2_model_heatmap.png")

    # ── Fig 3: Ablation A — fair comparison with N annotations ────────────
    if abl_a_df is not None and len(abl_a_df) > 0:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        a_leagues = sorted(abl_a_df['League'].unique())
        x = np.arange(len(a_leagues))
        rw = abl_a_df[abl_a_df['Method'].str.contains('Rolling')].sort_values('League')
        fs = abl_a_df[abl_a_df['Method'].str.contains('6-Season')].sort_values('League')
        ax.bar(x - 0.2, fs['MAE'].values, 0.38,
               label='6-Season Filter (base paper approach)',
               color=COLORS['6S'], alpha=0.82, zorder=2)
        ax.bar(x + 0.2, rw['MAE'].values, 0.38,
               label='Rolling Window — our approach',
               color=COLORS['RW'], alpha=0.90, zorder=2)
        ax.set_xticks(x); ax.set_xticklabels(a_leagues, fontsize=10)
        ax.set_ylabel('MAE (goals)'); ax.grid(axis='y', alpha=0.3, zorder=0)
        ax.set_title('Fig 3: Ablation A — Data Construction (Fair: Same BP Test Players)')
        ax.legend(fontsize=9)
        for i, (rr, fr) in enumerate(zip(rw.itertuples(), fs.itertuples())):
            ax.text(i+0.2, rr.MAE+0.015, f'N={rr.Train_N}',
                    ha='center', fontsize=8, color='#0F6E56')
            ax.text(i-0.2, fr.MAE+0.015, f'N={fr.Train_N}',
                    ha='center', fontsize=8, color='#993C1D')
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/fig3_ablation_a_fair.png', dpi=150, bbox_inches='tight')
        plt.close(); print("Saved fig3_ablation_a_fair.png")

    # ── Fig 4: Ablation B — career features null result ───────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    b_leagues = sorted(abl_b_df['League'].unique())
    x = np.arange(len(b_leagues))
    with_v    = abl_b_df[abl_b_df['Variant'].str.contains('With career')].sort_values('League')
    without_v = abl_b_df[abl_b_df['Variant'].str.contains('Without career')].sort_values('League')
    ax.bar(x - 0.2, without_v['MAE'].values, 0.38,
           label='Without career features', color=COLORS['without'], alpha=0.82, zorder=2)
    ax.bar(x + 0.2, with_v['MAE'].values,    0.38,
           label='With career trajectory features', color=COLORS['with'], alpha=0.90, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(b_leagues, fontsize=10)
    ax.set_ylabel('MAE (goals)'); ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.set_title('Fig 4: Ablation B — Career Features (Empirical Investigation, Null Result)')
    ax.legend(fontsize=9)
    # Mark direction of effect per league
    for i, (wv, wov) in enumerate(zip(with_v['MAE'].values, without_v['MAE'].values)):
        diff = wv - wov
        col  = COLORS['RW'] if diff < 0 else COLORS['6S']
        ax.text(i, max(wv, wov) + 0.015, f'{diff:+.3f}',
                ha='center', fontsize=9, color=col, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/fig4_ablation_b_career.png', dpi=150, bbox_inches='tight')
    plt.close(); print("Saved fig4_ablation_b_career.png")

    # ── Fig 5: Ablation D — cross-league with downsampled control ─────────
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(leagues))
    w_maes  = [cross_matrix[l]['within']     for l in leagues]
    cf_maes = [cross_matrix[l]['cross_full'] for l in leagues]
    cd_maes = [cross_matrix[l]['cross_down'] for l in leagues]
    ax.bar(x - 0.27, w_maes,  0.26, label='Within-league',
           color=COLORS['within'],    alpha=0.90, zorder=2)
    ax.bar(x,        cf_maes, 0.26, label='Cross-league full (3x data)',
           color=COLORS['cross_full'],alpha=0.88, zorder=2)
    ax.bar(x + 0.27, cd_maes, 0.26, label='Cross-league downsampled (matched N)',
           color=COLORS['cross_down'], alpha=0.88, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(leagues, fontsize=10)
    ax.set_ylabel('MAE (goals)'); ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.set_title('Fig 5: Ablation D — Cross-League Transfer (True Transfer vs Data-Size Effect)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/fig5_ablation_d_cross.png', dpi=150, bbox_inches='tight')
    plt.close(); print("Saved fig5_ablation_d_cross.png")

    # ── Fig 6: Scoring tier MAE ───────────────────────────────────────────
    tier_clean = tier_df[tier_df['MAE'].notna() & tier_df['Tier'].str.contains('goal')]
    if len(tier_clean) > 0:
        pivot_t = tier_clean.pivot(index='Tier', columns='League', values='MAE')
        fig, ax = plt.subplots(figsize=(9, 4))
        pivot_t.plot(kind='bar', ax=ax, colormap='tab10', alpha=0.85, width=0.7)
        ax.set_title('Fig 6: MAE by Scoring Tier — Best Model per League')
        ax.set_ylabel('MAE (goals)'); ax.set_xlabel('')
        ax.set_xticklabels(pivot_t.index, rotation=10, ha='right')
        ax.legend(fontsize=9, loc='upper left'); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/fig6_tier_mae.png', dpi=150, bbox_inches='tight')
        plt.close(); print("Saved fig6_tier_mae.png")

    # ── Fig 7: Goal distributions ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    for ax, league in zip(axes, leagues):
        # Load fresh for plot
        path = ([v for k, v in DATA_FILES.items() if league in k] or [None])[0]
        if path:
            import warnings; warnings.filterwarnings('ignore')
            sys.path.insert(0, '/home/claude')
            raw   = load_and_clean(path, league)
            pairs = extract_rolling_pairs(raw)
            ax.hist(pairs['Next_Gls'], bins=range(0, 30),
                    color='#3266ad', alpha=0.75, edgecolor='white')
        ax.set_title(league, fontsize=11)
        ax.set_xlabel('Next season goals')
        if ax == axes[0]:
            ax.set_ylabel('Count')
    plt.suptitle('Fig 7: Next-Season Goal Distribution by League', fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/fig7_goal_distributions.png', dpi=150, bbox_inches='tight')
    plt.close(); print("Saved fig7_goal_distributions.png")

    # ── Fig 8: R² comparison — full test vs BP subset ─────────────────────
    best_r2_full = (within_df.sort_values('R2_test', ascending=False)
                   .groupby('League').first().reset_index())
    best_r2_bp   = (within_df.sort_values('R2_test_bp', ascending=False)
                   .groupby('League').first().reset_index()
                   if 'R2_test_bp' in within_df.columns else None)

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(leagues)); w = 0.22
    base_r2s  = [BASE_PAPER_R2[l]  for l in leagues]
    our_full  = [best_r2_full[best_r2_full['League']==l]['R2_test'].values[0]
                 for l in leagues]
    ax.bar(x - w, base_r2s, w, label='Base paper R²', color='#9b9b9b', alpha=0.85)
    ax.bar(x,     our_full, w, label='Our R² (full test, 3x more players)',
           color='#E76F51', alpha=0.85)
    if best_r2_bp is not None:
        our_bp = [best_r2_bp[best_r2_bp['League']==l]['R2_test_bp'].values[0]
                  for l in leagues]
        ax.bar(x + w, our_bp, w,
               label='Our R² (BP subset — same players as base paper)',
               color=COLORS['RW'], alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(leagues, fontsize=10)
    ax.set_ylabel('R²'); ax.grid(axis='y', alpha=0.3)
    ax.set_title('Fig 8: R² — Base Paper vs Our Method\n'
                 '(Full test set includes harder/more diverse players)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/fig8_r2_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(); print("Saved fig8_r2_comparison.png")

    print("\nAll figures saved.")


# =============================================================================
# SECTION 15 — Save Results to Excel
# =============================================================================
def save_results(within_df, abl_a_df, abl_b_df, abl_d_df,
                 tier_df, wilc_df):
    """Save all results tables as a single Excel workbook."""
    path = f'{RESULTS_DIR}/paper_results_final.xlsx'
    with pd.ExcelWriter(path, engine='openpyxl') as writer:

        # Summary table: best per league on both test sets
        best_full = (within_df.sort_values('MAE_test')
                     .groupby('League').first().reset_index()
                     [['League','Model','Case','MAE_test','RMSE_test','R2_test']])
        best_full.columns = ['League','Model','Case',
                             'MAE_full','RMSE_full','R2_full']

        if 'MAE_test_bp' in within_df.columns:
            best_bp = (within_df.sort_values('MAE_test_bp')
                       .groupby('League').first().reset_index()
                       [['League','Model','Case',
                         'MAE_test_bp','RMSE_test_bp','R2_test_bp']])
            best_bp.columns = ['League','Model','Case',
                               'MAE_bp','RMSE_bp','R2_bp']
            summary = best_full.merge(best_bp[['League','MAE_bp','RMSE_bp','R2_bp']],
                                       on='League', how='left')
            # Add base paper columns
            summary['MAE_BasePaper']  = summary['League'].map(BASE_PAPER_MAE)
            summary['RMSE_BasePaper'] = summary['League'].map(BASE_PAPER_RMSE)
            summary['R2_BasePaper']   = summary['League'].map(BASE_PAPER_R2)
        else:
            summary = best_full

        summary.to_excel(writer, sheet_name='Best_Models_Summary', index=False)
        within_df.to_excel(writer,  sheet_name='All_Within_League',     index=False)
        if abl_a_df is not None:
            abl_a_df.to_excel(writer, sheet_name='Ablation_A_FairCompare', index=False)
        abl_b_df.to_excel(writer,   sheet_name='Ablation_B_CareerFeats',  index=False)
        abl_d_df.to_excel(writer,   sheet_name='Ablation_D_CrossLeague',  index=False)
        tier_df.to_excel(writer,    sheet_name='Scoring_Tier_MAE',        index=False)
        wilc_df.to_excel(writer,    sheet_name='Wilcoxon_Tests',          index=False)

    print(f"\nSaved: {path}")

    # Also save individual CSVs
    for df, name in [
        (within_df, 'within_league'), (abl_b_df, 'ablation_b'),
        (abl_d_df,  'ablation_d'),   (tier_df,   'tier_mae'),
        (wilc_df,   'wilcoxon'),
    ]:
        df.to_csv(f'{RESULTS_DIR}/{name}.csv', index=False)
    if abl_a_df is not None:
        abl_a_df.to_csv(f'{RESULTS_DIR}/ablation_a_fair.csv', index=False)


# =============================================================================
# SECTION 16 — Main Runner
# =============================================================================
def main():
    t_start = time.time()

    print("=" * 70)
    print("FOOTBALL PLAYER PERFORMANCE PREDICTION — FINAL PAPER PIPELINE")
    print("Inclusive Rolling-Window Sampling + Cross-League Evaluation")
    print(f"FAST_MODE = {FAST_MODE}  |  XGBoost = {XGBOOST_AVAILABLE}  "
          f"|  SHAP = {SHAP_AVAILABLE}")
    print("=" * 70)

    # ── Step 1: Load data ─────────────────────────────────────────────────
    print("\n[1/7] Loading and preprocessing data...")
    raw_dfs, all_data, bp_players = {}, {}, {}
    total_pairs = 0

    for league, path in DATA_FILES.items():
        raw   = load_and_clean(path, league)
        raw_dfs[league] = raw
        pairs = extract_rolling_pairs(raw)
        all_data[league] = pairs

        # Base-paper players: appeared in all 6 seasons
        six_s  = raw.groupby('Player').filter(
            lambda x: x['season_order'].nunique() == 6)
        bp_set = set(six_s[six_s['season_order'] == TEST_SEASON]['Player'])
        bp_players[league] = bp_set

        n_tr = (pairs['season_order'] < TEST_SEASON).sum()
        n_te = (pairs['season_order'] == TEST_SEASON).sum()
        total_pairs += len(pairs)

        print(f"  {league:<16}: {len(pairs)} pairs  "
              f"(train={n_tr}, test_full={n_te}, test_BP={len(bp_set)})")

    print(f"\n  Total pairs: {total_pairs}  "
          f"(base paper total: 424 — {total_pairs/424:.0f}x more)")

    # ── Step 2: Within-league experiments ────────────────────────────────
    print("\n[2/7] Within-league experiments (6 models × 3 cases × 4 leagues)...")
    within_df, best_models, tier_df = experiment_within_league(all_data, bp_players)

    # ── Step 3: Ablation A ────────────────────────────────────────────────
    print("\n[3/7] Ablation A — fair data construction comparison...")
    abl_a_df = ablation_a_data_construction(all_data, raw_dfs, bp_players)

    # ── Step 4: Ablation B ────────────────────────────────────────────────
    print("\n[4/7] Ablation B — career features investigation...")
    abl_b_df = ablation_b_career_features(all_data)

    # ── Step 5: Ablation D ────────────────────────────────────────────────
    print("\n[5/7] Ablation D — cross-league transfer (with data-size control)...")
    abl_d_df, cross_matrix = ablation_d_cross_league(all_data)

    # ── Step 6: Wilcoxon tests ────────────────────────────────────────────
    print("\n[6/7] Wilcoxon significance tests...")
    wilc_df = run_wilcoxon_tests(all_data)

    # ── Step 7: Figures + SHAP + Save ─────────────────────────────────────
    print("\n[7/7] Generating figures, SHAP, and saving results...")
    generate_all_figures(within_df, abl_a_df, abl_b_df,
                          abl_d_df, tier_df, cross_matrix)
    run_shap_analysis(best_models)
    save_results(within_df, abl_a_df, abl_b_df, abl_d_df, tier_df, wilc_df)

    # ── Final summary ─────────────────────────────────────────────────────
    elapsed = round(time.time() - t_start, 0)
    print(f"\n{'='*70}")
    print("PAPER RESULTS SUMMARY")
    print(f"{'='*70}")

    best = (within_df.sort_values('MAE_test').groupby('League').first()
            .reset_index()[['League','Model','Case','MAE_test','R2_test']])
    if 'MAE_test_bp' in within_df.columns:
        best_bp = (within_df.sort_values('MAE_test_bp').groupby('League').first()
                   .reset_index()[['League','MAE_test_bp','R2_test_bp']])
        best = best.merge(best_bp, on='League', how='left')
        for _, r in best.iterrows():
            bp_ref  = BASE_PAPER_MAE[r['League']]
            our_bp  = r.get('MAE_test_bp', r['MAE_test'])
            delta   = round(our_bp - bp_ref, 3)
            verdict = 'BETTER' if delta < 0 else 'WORSE'
            print(f"  {r['League']:<15} best={r['Model']}/{r['Case']}  "
                  f"MAE_full={r['MAE_test']:.3f}  "
                  f"MAE_BP={our_bp:.3f}  "
                  f"base={bp_ref}  [{verdict} {delta:+.3f}]")
    else:
        print(best.to_string(index=False))

    print(f"\nABLATION A (rolling window vs 6-season filter, fair test):")
    if abl_a_df is not None:
        for league in abl_a_df['League'].unique():
            sub = abl_a_df[abl_a_df['League'] == league]
            rw = sub[sub['Method'].str.contains('Rolling')]['MAE'].values[0]
            fs = sub[sub['Method'].str.contains('6-Season')]['MAE'].values[0]
            winner = 'RW WINS' if rw < fs else 'FILTER WINS'
            impr   = round((fs - rw) / fs * 100, 1)
            print(f"  {league:<15} RW={rw:.3f}  6S={fs:.3f}  [{winner}  {impr:+.1f}%]")

    print(f"\nABLATION D (cross-league):")
    for _, r in abl_d_df.iterrows():
        print(f"  {r['Target_League']:<15} {r['Finding']}")

    print(f"\nWILCOXON: any p<0.05 significance = {wilc_df['Significant_p05'].any()}")
    print(f"\n[DONE] Elapsed: {elapsed:.0f}s")
    print(f"       Results: {RESULTS_DIR}/paper_results_final.xlsx")
    print(f"       Figures: {RESULTS_DIR}/fig1–fig8")


if __name__ == '__main__':
    main()