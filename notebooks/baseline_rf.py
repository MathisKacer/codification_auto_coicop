# %% [markdown]
# # Modele binaire : la baseline est-elle correcte ?
#
# Objectif : predire si la baseline (vote majoritaire, TTC en cas d'egalite)
# donne le bon code COICOP au niveau 4. Une bonne prediction de ce modele
# permettrait de savoir *quand* faire confiance a la baseline et quand
# envoyer au LLM (traitement plus couteux).

# %% Imports
import sys, os
sys.path.append(os.path.abspath(".."))

import numpy as np
import pandas as pd

from data.load_data import charger_donnees
from src.baseline import (
    baseline_majorite_ttc, preparer_donnees, entrainer_evaluer, entrainer_evaluer_cv,
    courbe_precision_rappel,
)

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

# %% Entrainement + evaluation (split unique, pour les importances de features)
res = entrainer_evaluer(X, y, test_size=0.2, random_state=42)

# %% Validation croisee (estimation plus robuste des metriques)
res_cv = entrainer_evaluer_cv(X, y, n_splits=5, random_state=42)

# %% Compromis precision/rappel sur la detection des erreurs de baseline
df_seuils = courbe_precision_rappel(res["y_test"], res["y_proba"])

# %% Analyse des erreurs manquees par le modele (faux negatifs, seuil 0.5)
# Parmi les vraies erreurs de baseline du test, lesquelles la RF laisse-t-elle
# passer a tort (P(baseline correcte) > 0.5) ?
X_test, y_test = res["X_test"], res["y_test"]
y_pred_rf = pd.Series(res["y_pred"], index=y_test.index)
y_proba_rf = pd.Series(res["y_proba"], index=y_test.index)

erreurs_baseline = X_test[y_test == 0].copy()
erreurs_baseline["statut"] = np.where(
    y_pred_rf[y_test == 0] == 1, "Manquee (RF dit correcte)", "Detectee par la RF",
)
erreurs_baseline["proba_correcte"] = y_proba_rf[y_test == 0]
print(erreurs_baseline["statut"].value_counts())

# %% Comparaison des features : erreurs manquees vs detectees
print(erreurs_baseline.groupby("statut").agg(
    n=("nb_accords_max", "count"),
    nb_accords_max_median=("nb_accords_max", "median"),
    ttc_conf_1_median=("ttc_conf_1", "median"),
    rag_confidence_median=("rag_confidence", "median"),
    ragann_confidence_median=("ragann_confidence", "median"),
    budget_median=("budget", "median"),
    proba_correcte_median=("proba_correcte", "median"),
).round(3).T.to_string())

# %% Repartition par division COICOP
division_erreurs = df.loc[erreurs_baseline.index, col_vrai].astype(str).str[:2]
tab_division = erreurs_baseline.assign(division=division_erreurs).groupby(
    ["division", "statut"]
).size().unstack(fill_value=0)
tab_division["taux_manquee (%)"] = (
    tab_division.get("Manquee (RF dit correcte)", 0) / tab_division.sum(axis=1) * 100
).round(1)
print(tab_division.sort_values("taux_manquee (%)", ascending=False).to_string())

# %% Detail des erreurs manquees, triees par proba_rf decroissante
manquees_idx = erreurs_baseline[
    erreurs_baseline["statut"] == "Manquee (RF dit correcte)"
].index

table_manquees = pd.DataFrame({
    "produit": df.loc[manquees_idx, "l_pr_product"],
    "enseigne": df.loc[manquees_idx, "shop_type_name"],
    "code_vrai": df.loc[manquees_idx, col_vrai],
    "code_baseline": y_pred_baseline.loc[manquees_idx],
    "lcs_code": df.loc[manquees_idx, "lcs_code"],
    "rag_code": df.loc[manquees_idx, "rag_code"],
    "ragann_code": df.loc[manquees_idx, "ragann_code"],
    "ttc_code_1": df.loc[manquees_idx, "ttc_code_1"],
    "ttc_conf_1": df.loc[manquees_idx, "ttc_conf_1"],
    "nb_accords_max": erreurs_baseline.loc[manquees_idx, "nb_accords_max"],
    "proba_rf": erreurs_baseline.loc[manquees_idx, "proba_correcte"],
}).sort_values("proba_rf", ascending=False)
print(table_manquees.round(3).to_string())
# %%
