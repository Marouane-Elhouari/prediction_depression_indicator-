import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    )
from xgboost import XGBClassifier 
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import random
from sklearn.ensemble import RandomForestClassifier
import optuna 

SEED = 42
ID_COL = 'id'
TARGET_COL = 'Depression'
PRIMARY_METRIC = 'f1_macro'
N_SPLITS = 3
HOLDOUT_SIZE = 0.15
PAIRPLOT_SAMPLE = 4000
EDA_PLOT_SAMPLE = 20000
HPO_SAMPLE_SIZE = 60000
N_JOBS = -1
USE_CLASS_WEIGHTS = True  # donne plus de poids a la classe minoritaire (yes/no) pendant l'entrainement
N_TRIALS = 4
def set_seed(seed : int = SEED) -> None : 
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed()
# Couleurs et Palettes
yellow, cyan_g, cyan_dark = "#F7C53E", "#0CF7AF", "#11AB7C"
purple, purple_dark, purple_light = "#D826F8", "#9309AB", "#b683d6"
blue, red, orange, green = "#0C97FA", "#FA1D19", "#FA9F19", "#0CFA58"
light_blue, soft_blue, dark_blue = "#01FADC", "#81c9e6", "#394be6"

PALETTE_2 = [cyan_g, purple]
PALETTE_3 = [yellow, cyan_g, purple]
PALETTE_7 = [purple_dark, purple_light, purple, blue, light_blue, dark_blue, soft_blue]
sns.set_style("whitegrid")
sns.set_palette(PALETTE_7)
plt.rcParams["figure.facecolor"] = "#f8fafc"
pd.set_option("display.float_format", "{:.4f}".format)


DATA_DIR = r"C:\Users\pc\Downloads\playground-series-s4e11"
train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')
sample_submission = pd.read_csv(f'{DATA_DIR}/sample_submission.csv')

train[TARGET_COL] = train[TARGET_COL].map({1: 'yes', 0: 'no', '1': 'yes', '0': 'no'})
sample_submission[TARGET_COL] = sample_submission[TARGET_COL].map({1: 'yes', 0: 'no', '1': 'yes', '0': 'no'})

print(f'Train shape: {train.shape}')
print(f'Test shape: {test.shape}')
print(f'Sample submission shape: {sample_submission.shape}')
assert ID_COL in train.columns and TARGET_COL in train.columns
assert ID_COL in test.columns 
assert len (sample_submission) == len(test)
print(f'Target values (train): {train[TARGET_COL].unique()}')
print(f'Target encoding successful: {set(train[TARGET_COL].unique()).issubset({"yes", "no"})}')
print("\n--- Train Head ---")
print(train.head())
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

eda_cols = [c for c in train.columns if c  != TARGET_COL]
eda_summary = build_eda_summary(train[eda_cols])
print(eda_summary)
print(train[eda_cols].describe().T)
print(f'\nTarget distribution:\n{train[TARGET_COL].value_counts()}')

CLASS_ORDER = sorted(train[TARGET_COL].unique())
TARGET_COLOR_MAP = dict(zip(CLASS_ORDER , PALETTE_3))

fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
counts = train[TARGET_COL].value_counts().reindex(CLASS_ORDER)
sns.barplot(x=counts.index, y=counts.values, palette=[TARGET_COLOR_MAP[c] for c in CLASS_ORDER], ax=axes[0])
axes[0].set_title(f'Distribution - {TARGET_COL}', fontsize=11, fontweight='bold')
axes[0].set_ylabel('nombre')
for i, v in enumerate(counts.values):
    axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=10)
pct = counts / counts.sum() * 100
axes[1].pie(pct.values, labels=pct.index, autopct='%1.1f%%', startangle=90,
            colors=[TARGET_COLOR_MAP[c] for c in CLASS_ORDER],
            wedgeprops={"edgecolor": "white", "linewidth": 1.5})
axes[1].set_title(f"Distribution - {TARGET_COL} (%)", fontsize=13, fontweight="bold")
plt.show()
missing_train = train.isna().mean().sort_values(ascending=False) * 100
missing_test = test.isna().mean().sort_values(ascending=False) * 100
missing_cols = [c for c in missing_train.index if missing_train[c] > 0]

if missing_cols:
    fig_width = max(8, 0.7 * len(missing_cols))
    fig, ax = plt.subplots(figsize=(fig_width, 6), constrained_layout=True)
    x = np.arange(len(missing_cols))
    width = 0.38
    ax.bar(x - width / 2, missing_train[missing_cols], width, label="Train", color=PALETTE_2[0])
    ax.bar(x + width / 2, missing_test.reindex(missing_cols).values, width, label="Test", color=PALETTE_2[1])
    ax.set_xticks(x)
    ax.set_xticklabels(missing_cols, rotation=40, ha='right')
    ax.set_ylabel("missing %")
    ax.set_title('% de NaN par colonne', fontsize=13, fontweight='bold')
    ax.legend()
    plt.show()

NUMERICAL_COL = ['Age' , 'Academic Pressure' , 'Work Pressure' , 'CGPA' ,'Study Satisfaction'  , 'Work/Study Hours' , 'Financial Stress']
CAT_COL = ['Name' , 'Gender' , 'City' , 'Working Professional or Student' , 'Profession' , 'Dietary Habits' , 'Degree' , 'Have you ever had suicidal thoughts ?' , 'Family History of Mental Illness']  

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
plt.show()


fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.2 * n_rows), constrained_layout=True)
axes = axes.flatten()
for i, col in enumerate(NUMERICAL_COL):
    sns.boxplot(data=eda_sample, x=TARGET_COL, y=col, order=CLASS_ORDER,
                palette=TARGET_COLOR_MAP, ax=axes[i], showfliers=False)
    axes[i].set_title(f"{col} par {TARGET_COL}", fontsize=11)
for j in range(len(NUMERICAL_COL), len(axes)):
    axes[j].axis("off")
plt.show()


def plot_categorical_vs_target(df, column, target, class_order, color_map):
    for col in column:
        n_categories = df[col].nunique()
        fig_width = max(11, 1.1 * n_categories + 6)
        fig, axes = plt.subplots(1, 2, figsize=(fig_width, 4.5), constrained_layout=True)
        order = df[col].value_counts().index
        sns.countplot(data=df, x=col, order=order, hue=col, palette=PALETTE_7, legend=False, ax=axes[0])
        axes[0].set_title(col, fontsize=11)
        axes[0].tick_params(axis="x", rotation=25)
        prop = pd.crosstab(df[col], df[target], normalize='index').reindex(columns=class_order).loc[order]
        prop.plot(kind="bar", stacked=True, color=[color_map[c] for c in class_order], ax=axes[1], width=0.75)
        axes[1].set_title(f"{col} vs {target} (proportion)", fontsize=11)
        axes[1].set_ylabel("proportion")
        axes[1].tick_params(axis="x", rotation=25)
        axes[1].legend(title=target, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.show()


plot_categorical_vs_target(train, CAT_COL, TARGET_COL, CLASS_ORDER, TARGET_COLOR_MAP)

pair_sample = train.sample(min(len(train), PAIRPLOT_SAMPLE), random_state=SEED)
g = sns.pairplot(pair_sample, vars=NUMERICAL_COL[:5], hue=TARGET_COL,
                  hue_order=CLASS_ORDER, palette=TARGET_COLOR_MAP, corner=True,
                  height=1.8, plot_kws={"alpha": 0.45, "s": 16})
sns.move_legend(g, "upper right", bbox_to_anchor=(0.82, 0.85), frameon=False)
g.fig.set_size_inches(12, 12)
g.fig.suptitle('Pairplot des variables numériques selon addicted_label', y=1.02)
plt.show()



