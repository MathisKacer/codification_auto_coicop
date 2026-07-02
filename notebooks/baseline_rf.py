# %% [markdown]
# # Modele binaire : la baseline est-elle correcte ?
#
# Objectif : predire si la baseline (vote majoritaire, TTC en cas d'egalite)
# donne le bon code COICOP au niveau 4. Une bonne prediction de ce modele
# permettrait de savoir *quand* faire confiance a la baseline et quand
# escalader vers un traitement plus couteux (ex : LLM).

# %% Imports
import sys, os
sys.path.append(os.path.abspath(".."))

import pandas as pd

from data.load_data import charger_donnees
from src.baseline import baseline_majorite_ttc, preparer_donnees, entrainer_evaluer

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# %% Chargement des donnees
chemin_s3 = "s3://projet-budget-famille/data/workflow_runs/2026-06-29/codif-vvkv9/decide-coicop/predictions.parquet"
df = charger_donnees(chemin_s3)
df.head()

# %% Colonnes
cols_base = ["lcs_code", "rag_code", "ragann_code", "ttc_code_1"]
col_ttc = "ttc_code_1"
col_vrai = "code"

# %% Baseline : on la recalcule pour construire la cible et la feature associee
y_pred_baseline = baseline_majorite_ttc(df, cols_base, col_ttc, niveau=4)

# %% Preparation des donnees
X, y = preparer_donnees(df, y_pred_baseline, col_vrai, niveau=4)
X.head()

# %% Entrainement + evaluation
res = entrainer_evaluer(X, y, test_size=0.2, random_state=42)
# %%
