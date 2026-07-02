# %% [markdown]
# # Baseline pour la codification COICOP
#
# Objectif : établir une baseline simple avant de construire un modèle plus
# élaboré (Random Forest binaire "TTC a-t-il raison ?").
#
# Règle de la baseline :
# - vote majoritaire parmi les 4 classifieurs de base (LCS, RAG, RAG-ANN, TTC)
# - en cas d'égalité, on suit TTC
# - le code choisi est comparé au vrai code au niveau 4

# %% Imports
import sys
import os

sys.path.append(os.path.abspath(".."))  # accès aux modules src/ et data/

import pandas as pd
import numpy as np
from src.correspondance_hierarchique import tronquer_a_niveau
from data.load_data import charger_donnees
from src.baseline_ttc import baseline_majorite_ttc, evaluer_baseline

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# %% Chargement des données
chemin_s3 = "s3://projet-budget-famille/data/workflow_runs/2026-06-29/codif-vvkv9/decide-coicop/predictions.parquet"
df = charger_donnees(chemin_s3)
df.head()

# %% Définition des colonnes
cols_base = ["lcs_code", "rag_code", "ragann_code", "ttc_code_1"]
col_ttc = "ttc_code_1"
col_vrai = "code"
cols_votants = cols_base   # les 4 base, TTC inclus

# %% [markdown]
# ## Application de la baseline

# %% Prédiction
y_pred_baseline = baseline_majorite_ttc(df, cols_votants, col_ttc, niveau=4)
y_pred_baseline.head(10)

# %% Évaluation
acc_baseline = evaluer_baseline(df, y_pred_baseline, col_vrai, niveau=4)

# %%
# ## Test en ne prenant que le code de TTC

# %% Accuracy si on suit TTC seul (baseline dégradée de référence)
y_pred_ttc = df[col_ttc].map(lambda x: tronquer_a_niveau(x, niveau=4))
acc_ttc = evaluer_baseline(df, y_pred_ttc, col_vrai, niveau=4)

print(f"\nComparaison :")
print(f"  TTC seul       : {acc_ttc:.3f}")
print(f"  Vote majoritaire (TTC arbitre) : {acc_baseline:.3f}")
print(f"  Gain           : {(acc_baseline - acc_ttc)*100:+.2f} points")

# %%
