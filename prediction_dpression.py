

import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, precision_recall_curve,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

SEED = 42
ID_COL = 'id'
TARGET_COL = 'Depression'
N_SPLITS = 5
HOLDOUT_SIZE = 0.15
N_JOBS = -1
FBETA = 1.0
EDA_PLOT_SAMPLE = 20000
HPO_SAMPLE_SIZE = 60000
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed()

# Couleurs / palettes (meme esprit que le script addicted_comptetion_kaggle.py)
yellow, cyan_g, purple = "#F7C53E", "#0CF7AF", "#D826F8"
PALETTE_2 = [cyan_g, purple]
PALETTE_3 = [yellow, cyan_g, purple]

sns.set_style("whitegrid")
plt.rcParams["figure.facecolor"] = "#f8fafc"
pd.set_option("display.float_format", "{:.4f}".format)

DATA_DIR = r'C:\Users\pc\Downloads\playground-series-s4e11'
OUT_DIR = r'C:\Users\pc\Downloads\playground-series-s4e11\outputs'
os.makedirs(OUT_DIR, exist_ok=True)

train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')

print(f'Train shape: {train.shape}')
print(f'Test shape: {test.shape}')

assert ID_COL in train.columns and TARGET_COL in train.columns
assert ID_COL in test.columns

print(f'\nDistribution de la cible ({TARGET_COL}):')
print(train[TARGET_COL].value_counts())
print(train[TARGET_COL].value_counts(normalize=True) * 100)

CLASS_ORDER = [0, 1]
TARGET_COLOR_MAP = {0: PALETTE_3[1], 1: PALETTE_3[2]}

# =========================================================
# EDA enrichie
# =========================================================
def build_eda_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        series = df[column]
        not_null = series.dropna()
        rows.append({
            'features': column,
            'dtype': str(series.dtype),
            'missing_count': int(series.isna().sum()),
            'missing_pct': float(series.isna().mean() * 100.0),
            'nunique': int(series.nunique()),
            'sample_values': ','.join(not_null.astype(str).unique()[:4]),
        })
    return pd.DataFrame(rows)


eda_cols = [c for c in train.columns if c != TARGET_COL]
print("\n--- Resume EDA ---")
eda_summary = build_eda_summary(train[eda_cols])
print(eda_summary)
print(train[eda_cols].describe(include='all').T)

# --- Distribution de la cible (barres + camembert) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
counts = train[TARGET_COL].value_counts().reindex(CLASS_ORDER)
sns.barplot(x=['Non (0)', 'Oui (1)'], y=counts.values,
            palette=[TARGET_COLOR_MAP[c] for c in CLASS_ORDER], ax=axes[0])
axes[0].set_title(f'Distribution - {TARGET_COL}', fontsize=11, fontweight='bold')
axes[0].set_ylabel('nombre')
for i, v in enumerate(counts.values):
    axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=10)
pct = counts / counts.sum() * 100
axes[1].pie(pct.values, labels=['Non (0)', 'Oui (1)'], autopct='%1.1f%%', startangle=90,
            colors=[TARGET_COLOR_MAP[c] for c in CLASS_ORDER],
            wedgeprops={"edgecolor": "white", "linewidth": 1.5})
axes[1].set_title(f"Distribution - {TARGET_COL} (%)", fontsize=13, fontweight="bold")
plt.savefig(f'{OUT_DIR}/eda_target_distribution.png', dpi=120)
plt.close()

# --- % de valeurs manquantes (train vs test) ---
missing_train = train.isna().mean().sort_values(ascending=False) * 100
missing_test = test.isna().mean().reindex(missing_train.index) * 100
missing_cols = [c for c in missing_train.index if missing_train[c] > 0]

if missing_cols:
    fig_width = max(8, 0.7 * len(missing_cols))
    fig, ax = plt.subplots(figsize=(fig_width, 6), constrained_layout=True)
    x = np.arange(len(missing_cols))
    width = 0.38
    ax.bar(x - width / 2, missing_train[missing_cols], width, label="Train", color=PALETTE_2[0])
    ax.bar(x + width / 2, missing_test[missing_cols].values, width, label="Test", color=PALETTE_2[1])
    ax.set_xticks(x)
    ax.set_xticklabels(missing_cols, rotation=40, ha='right')
    ax.set_ylabel("missing %")
    ax.set_title('% de NaN par colonne', fontsize=13, fontweight='bold')
    ax.legend()
    plt.savefig(f'{OUT_DIR}/eda_missing_values.png', dpi=120)
    plt.close()

# --- Variables numeriques vs cible (kde + boxplot) ---
NUMERICAL_COL = [c for c in [
    'Age', 'Academic Pressure', 'Work Pressure', 'CGPA',
    'Study Satisfaction', 'Job Satisfaction', 'Work/Study Hours',
    'Financial Stress',
] if c in train.columns]

eda_sample = train.sample(min(len(train), EDA_PLOT_SAMPLE), random_state=SEED)

n_cols = 2
n_rows = int(np.ceil(len(NUMERICAL_COL) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.2 * n_rows), constrained_layout=True)
axes = axes.flatten()
for i, col in enumerate(NUMERICAL_COL):
    sns.kdeplot(data=eda_sample, x=col, hue=TARGET_COL, hue_order=CLASS_ORDER,
                palette=TARGET_COLOR_MAP, fill=True, alpha=0.35, common_norm=False, ax=axes[i])
    axes[i].set_title(f'Influence de {col} sur {TARGET_COL}', fontsize=11)
for j in range(len(NUMERICAL_COL), len(axes)):
    axes[j].axis('off')
plt.savefig(f'{OUT_DIR}/eda_numerical_kde.png', dpi=120)
plt.close()

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.2 * n_rows), constrained_layout=True)
axes = axes.flatten()
for i, col in enumerate(NUMERICAL_COL):
    sns.boxplot(data=eda_sample, x=TARGET_COL, y=col, order=CLASS_ORDER,
                hue=TARGET_COL, palette=TARGET_COLOR_MAP, legend=False,
                ax=axes[i], showfliers=False)
    axes[i].set_title(f"{col} par {TARGET_COL}", fontsize=11)
for j in range(len(NUMERICAL_COL), len(axes)):
    axes[j].axis("off")
plt.savefig(f'{OUT_DIR}/eda_numerical_boxplot.png', dpi=120)
plt.close()


# --- Variables categorielles vs cible (countplot + proportion empilee) ---
def plot_categorical_vs_target(df, columns, target, class_order, color_map, out_dir):
    for col in columns:
        if col not in df.columns:
            continue
        n_categories = df[col].nunique()
        fig_width = max(11, 1.1 * n_categories + 6)
        fig, axes = plt.subplots(1, 2, figsize=(fig_width, 4.5), constrained_layout=True)
        order = df[col].value_counts().index
        sns.countplot(data=df, x=col, order=order, hue=col, palette=PALETTE_3,
                       legend=False, ax=axes[0])
        axes[0].set_title(col, fontsize=11)
        axes[0].tick_params(axis="x", rotation=25)
        prop = pd.crosstab(df[col], df[target], normalize='index').reindex(columns=class_order).loc[order]
        prop.plot(kind="bar", stacked=True, color=[color_map[c] for c in class_order], ax=axes[1], width=0.75)
        axes[1].set_title(f"{col} vs {target} (proportion)", fontsize=11)
        axes[1].set_ylabel("proportion")
        axes[1].tick_params(axis="x", rotation=25)
        axes[1].legend(title=target, bbox_to_anchor=(1.02, 1), loc="upper left")
        safe_name = col.lower().replace(' ', '_').replace('/', '_')
        plt.savefig(f'{out_dir}/eda_cat_{safe_name}.png', dpi=120)
        plt.close()


CAT_COL = [c for c in [
    'Gender', 'Sleep Duration', 'Dietary Habits',
    'Working Professional or Student',
    'Have you ever had suicidal thoughts ?',
    'Family History of Mental Illness',
] if c in train.columns]

plot_categorical_vs_target(train, CAT_COL, TARGET_COL, CLASS_ORDER, TARGET_COLOR_MAP, OUT_DIR)

VALID_SLEEP = ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours']
VALID_DIET = ['Healthy', 'Moderate', 'Unhealthy']


def clean_noise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Sleep Duration'] = df['Sleep Duration'].where(
        df['Sleep Duration'].isin(VALID_SLEEP), other=np.nan
    )
    df['Dietary Habits'] = df['Dietary Habits'].where(
        df['Dietary Habits'].isin(VALID_DIET), other=np.nan
    )
    return df


def add_features(df: pd.DataFrame, freq_maps: dict = None):
    df = df.copy()

    df['is_student'] = (df['Working Professional or Student'] == 'Student').astype(int)

    df['pressure'] = df['Academic Pressure'].fillna(df['Work Pressure'])
    df['satisfaction'] = df['Study Satisfaction'].fillna(df['Job Satisfaction'])

    df['has_cgpa'] = df['CGPA'].notna().astype(int)
    df['pressure_missing'] = df['pressure'].isna().astype(int)
    df['satisfaction_missing'] = df['satisfaction'].isna().astype(int)

    df['pressure'] = df['pressure'].fillna(df['pressure'].median())
    df['satisfaction'] = df['satisfaction'].fillna(df['satisfaction'].median())
    df['CGPA'] = df['CGPA'].fillna(0)

    df['suicidal_thoughts'] = (df['Have you ever had suicidal thoughts ?'] == 'Yes').astype(int)
    df['family_history'] = (df['Family History of Mental Illness'] == 'Yes').astype(int)

    df['stress_load'] = df['pressure'] + df['Financial Stress'].fillna(df['Financial Stress'].median())
    df['work_study_x_pressure'] = df['Work/Study Hours'] * df['pressure']

    high_card_cols = ['City', 'Profession', 'Degree']
    if freq_maps is None:
        freq_maps = {c: df[c].value_counts(normalize=True) for c in high_card_cols}
    for c in high_card_cols:
        df[f'{c}_freq'] = df[c].map(freq_maps[c]).fillna(0)

    return df, freq_maps


train_c = clean_noise(train)
test_c = clean_noise(test)

train_fe, freq_maps = add_features(train_c)
test_fe, _ = add_features(test_c, freq_maps=freq_maps)

cat_low_card = ['Gender', 'Sleep Duration', 'Dietary Habits']

drop_cols = [
    ID_COL, TARGET_COL, 'Name', 'City', 'Profession', 'Degree',
    'Working Professional or Student', 'Have you ever had suicidal thoughts ?',
    'Family History of Mental Illness', 'Academic Pressure', 'Work Pressure',
    'Study Satisfaction', 'Job Satisfaction',
]

feature_cols = [c for c in train_fe.columns if c not in drop_cols]

n_train = len(train_fe)
all_data = pd.concat([train_fe[feature_cols], test_fe[feature_cols]], axis=0)
all_data = pd.get_dummies(all_data, columns=[c for c in cat_low_card if c in all_data.columns])
all_data = all_data.fillna(all_data.median(numeric_only=True))

X = all_data.iloc[:n_train].reset_index(drop=True)
X_test = all_data.iloc[n_train:].reset_index(drop=True)
y = train_fe[TARGET_COL].values

print(f'\nX_train shape: {X.shape}')
print(f'X_test shape: {X_test.shape}')

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=HOLDOUT_SIZE, random_state=SEED, stratify=y
)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)


grid_specs = {
    'RandomForest': (
        RandomForestClassifier(random_state=SEED, n_jobs=N_JOBS, class_weight='balanced'),
        {
            'n_estimators': [300, 500],
            'max_depth': [10, 14],
            'min_samples_leaf': [1, 3],
        },
    ),
    'XGBoost': (
        XGBClassifier(random_state=SEED, n_jobs=N_JOBS, eval_metric='logloss', tree_method='hist'),
        {
            'n_estimators': [300, 500],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1],
        },
    ),
    'LightGBM': (
        LGBMClassifier(random_state=SEED, n_jobs=N_JOBS, verbose=-1),
        {
            'n_estimators': [300, 500],
            'max_depth': [-1, 6],
            'learning_rate': [0.05, 0.1],
        },
    ),
    'CatBoost': (
        CatBoostClassifier(random_state=SEED, silent=True, thread_count=N_JOBS),
        {
            'iterations': [300, 500],
            'depth': [4, 6],
            'learning_rate': [0.05, 0.1],
        },
    ),
}

best_estimators = {}
print("\n=== Optimisation par GridSearchCV (scoring = f1) ===")
for name, (estimator, param_grid) in grid_specs.items():
    print(f"\n--- {name} ---")
    grid = GridSearchCV(
        estimator, param_grid, scoring='f1', cv=skf, n_jobs=N_JOBS, refit=True,
    )
    grid.fit(X, y)

    print(f"Meilleurs parametres : {grid.best_params_}")
    print(f"Meilleur F1 (CV)     : {grid.best_score_:.4f}")
    best_estimators[name] = grid.best_estimator_

def evaluate_all_metrics(y_true, y_pred, y_proba=None, label=""):
    results = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
    }
    if y_proba is not None:
        results['roc_auc'] = roc_auc_score(y_true, y_proba)
        results['pr_auc'] = average_precision_score(y_true, y_proba)

    print(f"=== Resume des metriques {label} ===")
    for name, value in results.items():
        print(f"{name:>10}: {value:.4f}")
    print("\n=== Classification report ===")
    print(classification_report(y_true, y_pred, target_names=['Non', 'Oui']))
    print("=== Matrice de confusion ===")
    print(confusion_matrix(y_true, y_pred))
    return results


val_probas = {}
print("\n=== RESULTATS FINAUX PAR MODELE (hold-out, seuil = 0.5) ===")
for name, model in best_estimators.items():
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    val_probas[name] = y_proba
    evaluate_all_metrics(y_val, y_pred, y_proba, label=f"- {name}")

    cm = confusion_matrix(y_val, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4), constrained_layout=True)
    sns.heatmap(cm, annot=True, fmt='d', cmap='mako', cbar=False,
                xticklabels=['Non', 'Oui'], yticklabels=['Non', 'Oui'], ax=ax)
    ax.set_xlabel('Predit')
    ax.set_ylabel('Reel')
    ax.set_title(f'Matrice de confusion - {name}', fontsize=11)
    plt.savefig(f'{OUT_DIR}/confusion_matrix_{name.lower()}.png', dpi=120)
    plt.close()

# --- Importance des variables (RandomForest, pour rester coherent avec le script de base) ---
importances = pd.Series(
    best_estimators['RandomForest'].feature_importances_, index=X.columns
).sort_values(ascending=False)
print("\n=== Top 15 variables les plus importantes (RandomForest) ===")
print(importances.head(15))

fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
importances.head(15).sort_values().plot(kind='barh', ax=ax, color=cyan_g)
ax.set_title('Importance des variables - RandomForest', fontsize=12, fontweight='bold')
plt.savefig(f'{OUT_DIR}/feature_importance.png', dpi=120)
plt.close()

def find_best_blend_weights(y_true, proba_dict, n_random=3000, seed=SEED):

    names = list(proba_dict.keys())
    probas = np.column_stack([proba_dict[n] for n in names])
    rng = np.random.default_rng(seed)

    best_w, best_score = None, -1.0
    for _ in range(n_random):
        w = rng.dirichlet(np.ones(len(names)))
        score = roc_auc_score(y_true, probas @ w)
        if score > best_score:
            best_score, best_w = score, w

    step = 0.05
    while step > 1e-3:
        improved = False
        for i in range(len(names)):
            for delta in (step, -step):
                w_try = best_w.copy()
                w_try[i] = max(0.0, w_try[i] + delta)
                if w_try.sum() == 0:
                    continue
                w_try = w_try / w_try.sum()
                score = roc_auc_score(y_true, probas @ w_try)
                if score > best_score:
                    best_score, best_w = score, w_try
                    improved = True
        if not improved:
            step /= 2

    return dict(zip(names, best_w)), best_score


print("\n=== Recherche des poids de blending (maximisation ROC-AUC) ===")
for name, proba in val_probas.items():
    print(f"{name:>20}: ROC-AUC seul = {roc_auc_score(y_val, proba):.4f}")

blend_weights, blend_roc_auc = find_best_blend_weights(y_val, val_probas)
print("\nMeilleurs poids trouves:")
for name, w in blend_weights.items():
    print(f"  {name:>20}: {w:.3f}")
print(f"ROC-AUC du blend: {blend_roc_auc:.4f}")

blended_val_proba = sum(blend_weights[n] * val_probas[n] for n in val_probas)

def tune_threshold_fbeta(y_true, proba, beta=FBETA):
    precision, recall, thresh = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]
    denom = (beta ** 2 * precision) + recall
    fbeta = np.divide((1 + beta ** 2) * precision * recall, denom,
                       out=np.zeros_like(denom), where=denom > 0)
    best_idx = np.argmax(fbeta)
    return thresh[best_idx], fbeta[best_idx]


def plot_threshold_metrics(y_true, proba, out_path, beta=FBETA, n_points=99):
    thresholds = np.linspace(0.01, 0.99, n_points)
    precisions, recalls, f1s, accs = [], [], [], []
    for t in thresholds:
        preds = (proba >= t).astype(int)
        precisions.append(precision_score(y_true, preds, zero_division=0))
        recalls.append(recall_score(y_true, preds, zero_division=0))
        f1s.append(f1_score(y_true, preds, zero_division=0))
        accs.append(accuracy_score(y_true, preds))

    best_t, best_fbeta = tune_threshold_fbeta(y_true, proba, beta)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(thresholds, precisions, label='Precision')
    ax.plot(thresholds, recalls, label='Recall')
    ax.plot(thresholds, f1s, label='F1')
    ax.plot(thresholds, accs, label='Accuracy', linestyle='--', alpha=0.6)
    ax.axvline(best_t, color='red', linestyle=':', label=f'Seuil optimal (F{beta}={best_fbeta:.3f})')
    ax.set_xlabel('Seuil de decision (sur P(Oui))')
    ax.set_ylabel('Score')
    ax.set_title(f'Evolution des metriques selon le seuil - blend (F{beta})')
    ax.legend()
    plt.savefig(out_path, dpi=120)
    plt.close()

    return best_t


best_threshold = plot_threshold_metrics(
    y_val, blended_val_proba, f'{OUT_DIR}/threshold_metrics_blend.png', beta=FBETA
)
print(f"\nSeuil optimal trouve (F{FBETA}): {best_threshold:.4f}")

final_blend_preds = (blended_val_proba >= best_threshold).astype(int)
print("\n=== RESULTATS FINAUX DU BLEND (seuil ajuste) ===")
evaluate_all_metrics(y_val, final_blend_preds, blended_val_proba, label="- BLEND")


print("\n=== Entrainement final sur l'ensemble du train ===")
test_probas = {}
for name, model in best_estimators.items():
    final_model = model.__class__(**model.get_params())
    final_model.fit(X, y)
    test_probas[name] = final_model.predict_proba(X_test)[:, 1]

blended_test_proba = sum(blend_weights[n] * test_probas[n] for n in test_probas)
test_preds = (blended_test_proba >= best_threshold).astype(int)

submission = pd.DataFrame({
    ID_COL: test_fe[ID_COL],
    TARGET_COL: test_preds,
})
submission.to_csv(f'{OUT_DIR}/submission.csv', index=False)
print(f"\nFichier de soumission cree: {OUT_DIR}/submission.csv")
print(submission[TARGET_COL].value_counts())
