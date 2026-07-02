# %% [markdown]
# # Stats descriptives — accord des classifieurs et faux positifs
#
# Analyse exploratoire des prédictions de codification automatique COICOP :
#
# 1. **Accord unanime des 4 classifieurs de base** (LCS, RAG, RAG-ANN, TTC)
#    et taux de faux positifs associés
# 2. **Dissociation selon le LLM-judge** : suit-il le consensus ou pas, et
#    qui a raison dans chaque cas ?
# 3. **Cas où un seul classifieur a raison** contre tous les autres
#    (focus sur la valeur ajoutée du LLM-judge)
# 4. **Évolution selon le niveau de troncature COICOP** (niveaux 1 à 4)

# %% Imports
import sys
import os

sys.path.append(os.path.abspath(".."))  # accès aux modules src/ et data/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data.load_data import charger_donnees
from src.stats_accord import (
    # Niveau unique
    stats_accord,
    analyse_faux_positifs,
    stats_accord_avec_llm,
    stats_classifieur_seul_correct,
    analyse_classifieur_seul,
    # Multi-niveaux
    stats_accord_multi_niveaux,
    stats_seul_multi_niveaux,
    rapport_complet_multi_niveaux,
    recap_multi_niveaux,
    plot_recap_multi_niveaux,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# %% Chargement des données
chemin_s3 = "s3://projet-budget-famille/data/workflow_runs/2026-06-29/codif-vvkv9/decide-coicop/predictions.parquet"
df = charger_donnees(chemin_s3)
df.head()

# %% Définition des colonnes
cols_base = ["lcs_code", "rag_code", "ragann_code", "ttc_code_1"]
col_llm = "llm_code"   # ← à confirmer / corriger
col_vrai = "code"
cols_tous = cols_base + [col_llm]   # pour les analyses avec LLM inclus

# %% [markdown]
# ## 1. Stats au niveau 4 (granularité max)
#
# Vue détaillée sur le niveau le plus fin : accord global des 4 base,
# analyse des FP, puis dissociation selon le LLM.

# %% 1a) Stats globales d'accord unanime des 4 classifieurs de base
df_stats = stats_accord(df, cols_base, col_vrai, niveau=4)

# %% 1b) Analyse fine des FP unanimes
df_fp = analyse_faux_positifs(df_stats, niveau=4, top_n=15)

# %% 1c) Dissociation selon que le LLM suit ou non le consensus
df_acc = stats_accord_avec_llm(df, cols_base, col_llm, col_vrai, niveau=4)

# %% 1d) Sous-tableaux pour exploration
cols_a_voir = [
    "l_pr_product", "code", "vrai_tronq",
    "code_consensus",
    *[f"{c}_tronq" for c in cols_base],
]

# FP unanimes des 4 base, enrichis de l'info LLM
df_fp = df_fp.copy()
df_fp["llm_tronq"] = df_fp[col_llm].map(lambda x: __import__("src.stats_accord", fromlist=["tronquer_niveau"]).tronquer_niveau(x, niveau=4))
df_fp["llm_suit"] = df_fp["llm_tronq"] == df_fp["code_consensus"]
df_fp["llm_correct"] = df_fp["llm_tronq"] == df_fp["vrai_tronq"]

fp_llm_suit = df_fp[df_fp["llm_suit"]]          # FP 5/5
fp_llm_diverge = df_fp[~df_fp["llm_suit"]]      # 4 base faux, LLM diverge

print(f"FP unanimes des 4 base       : {len(df_fp)}")
print(f"  ├─ LLM suit (FP 5/5)        : {len(fp_llm_suit)}")
print(f"  └─ LLM diverge              : {len(fp_llm_diverge)}")
print(f"       ├─ LLM correct (sauve) : {(~df_fp['llm_suit'] & df_fp['llm_correct']).sum()}")
print(f"       └─ LLM faux aussi      : {(~df_fp['llm_suit'] & ~df_fp['llm_correct']).sum()}")

# %% [markdown]
# ## 2. Cas où un seul classifieur a raison (niveau 4)
#
# Pour chaque classifieur, on regarde les cas où il est le seul à avoir
# raison parmi les 5 (4 base + LLM). Les cas où le **LLM** est seul correct
# représentent le plafond de verre d'un méta-classifieur sans LLM.

# %% 2a) Stats globales
df_seul = stats_classifieur_seul_correct(df, cols_tous, col_vrai, niveau=4)

# %% 2b) Détail par classifieur sauveur
for c in cols_tous:
    analyse_classifieur_seul(df_seul, c, top_n=10)

# %% 2c) Vue des lignes concernées
cols_seul_a_voir = [
    "l_pr_product", "code", "vrai_tronq",
    "classifieur_seul",
    *[f"{c}_tronq" for c in cols_tous],
]
df_seul[df_seul["seul_correct"]][cols_seul_a_voir].head(30)

# Focus LLM seul correct
df_seul[df_seul["classifieur_seul"] == col_llm][cols_seul_a_voir].head(20)

# %% [markdown]
# ## 3. Évolution selon le niveau de troncature COICOP
#
# On reproduit les analyses précédentes pour les 4 niveaux de troncature
# (division → sous-classe) afin de voir comment évoluent les taux d'accord
# et de faux positifs avec la granularité demandée.

# %% 3a) Tableau récap synthétique
recap = recap_multi_niveaux(df, cols_base, col_llm, col_vrai, niveaux=(1, 2, 3, 4))

# %% 3b) Visualisation : décomposition empilée par niveau
plot_recap_multi_niveaux(recap, n_total=len(df))

# %% 3c) Rapport détaillé pour chaque niveau (accord + FP + dissociation LLM)
res_complet = rapport_complet_multi_niveaux(
    df, cols_base, col_llm, col_vrai,
    niveaux=(1, 2, 3, 4), top_n=10,
)

# %% 3d) Cas "un seul correct" pour chaque niveau
res_seul = stats_seul_multi_niveaux(df, cols_tous, col_vrai, niveaux=(1, 2, 3, 4))

# %% [markdown]
# ## 4. Récupération des dataframes par niveau
#
# Les dictionnaires `res_complet` et `res_seul` permettent de récupérer
# les dataframes complets pour creuser un niveau particulier.


recap   # tableau récap synthétique
# %% Exemples de récupération
df_stats_n3 = res_complet[3]["df_stats"]
df_fp_n3 = res_complet[3]["df_fp"]
df_acc_n3 = res_complet[3]["df_acc"]
df_seul_n2 = res_seul[2]
# %%
res_seul[4]
# %%
# %% Export du rapport HTML complet
from src.stats_accord import rapport_html

rapport_html(
    df, cols_base, col_llm, col_vrai, cols_tous,
    niveaux=(1, 2, 3, 4),
    top_n=10,
    col_libelle="l_pr_product",
    chemin_sortie="outputs/rapport_stats_accord.html",
)

# %%
