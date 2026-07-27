# %% [markdown]
# # Généralisation à un nouveau run — résultats d'accuracy
#
# Le modèle RF de `modelisation.qmd` (prédit si la baseline — vote
# majoritaire, TTC arbitre — a trouvé le bon code) est entraîné et évalué
# sur le run `codif-vvkv9` (2026-06-29). Ici, on l'entraîne EXACTEMENT comme
# `modelisation.qmd` — sur 80% de l'ANCIEN run seulement (même split,
# `test_size=0.2`, `random_state=42`) — et on l'évalue en plus sur le
# NOUVEAU run dans son intégralité, sans jamais l'entraîner dessus.
#
# Uniquement les résultats d'accuracy ici. Pour l'investigation des causes
# d'un éventuel écart (fuite train/test, comparaison des classifieurs,
# profondeur des codes...), voir `recherche_ecart_performance.py`.
#
# Ce notebook ne touche pas à `data/load_data.py` (donc ne change rien pour
# le site tant qu'on ne décide pas de publier) : le chemin du nouveau run
# est défini ici, en local.
#
# NB run choisi : le 2026-07-23 contient 3 sous-runs (`codif-5vrcw` 29 Ko,
# `codif-nzxdw` 24 Ko, `codif-8m8jn` 1.1 Mo). On prend `codif-8m8jn` — le
# plus tardif et de loin le plus volumineux (5834 lignes, contre 5825 pour
# l'ancien run). À corriger ci-dessous si ce n'est pas le bon.

# %% Imports
import sys, os
sys.path.append(os.path.abspath(".."))

import io
from contextlib import redirect_stdout

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from data.load_data import charger_donnees, CHEMIN_S3_MODELISATION
from src.baseline import baseline_majorite_ttc, evaluer_baseline, preparer_donnees, entrainer_evaluer

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# %% Chemins des deux runs
CHEMIN_ANCIEN = CHEMIN_S3_MODELISATION  # codif-vvkv9, 2026-06-29 — run du site actuel
CHEMIN_NOUVEAU = (
    "s3://projet-budget-famille/data/workflow_runs/2026-07-23/codif-8m8jn"
    "/decide-coicop/predictions.parquet"
)

cols_base = ["lcs_code", "rag_code", "ragann_code", "ttc_code_1"]
col_ttc = "ttc_code_1"
col_vrai = "code"

# %% Chargement des deux runs
with redirect_stdout(io.StringIO()):
    df_ancien = charger_donnees(CHEMIN_ANCIEN)
    df_nouveau = charger_donnees(CHEMIN_NOUVEAU)

print(f"Ancien run  : {len(df_ancien)} lignes")
print(f"Nouveau run : {len(df_nouveau)} lignes")

# %% [markdown]
# ## 1. Baseline (vote majoritaire, TTC arbitre)

# %%
def accuracy_baseline(df):
    """Accuracy silencieuse de la baseline (masque les print() internes)."""
    with redirect_stdout(io.StringIO()):
        y_pred = baseline_majorite_ttc(df, cols_base, col_ttc, niveau=4)
        return y_pred, evaluer_baseline(df, y_pred, col_vrai, niveau=4)

y_pred_baseline_ancien, acc_baseline_ancien = accuracy_baseline(df_ancien)
y_pred_baseline_nouveau, acc_baseline_nouveau = accuracy_baseline(df_nouveau)

# %% [markdown]
# ## 2. Modèle RF (entraîné sur 80% de l'ancien run, même split que le site)
#
# `entrainer_evaluer` fait un split 80/20 stratifié de l'ancien run,
# entraîne le pipeline sur les 80% (train), et l'évalue sur les 20% restants
# (test) — c'est le modèle `res` publié dans `modelisation.qmd`. On réutilise
# ici son pipeline déjà entraîné, sans le réentraîner sur davantage de
# données, puis on l'évalue sur le nouveau run (jamais vu à l'entraînement).

# %%
with redirect_stdout(io.StringIO()):
    X_ancien, y_ancien = preparer_donnees(df_ancien, y_pred_baseline_ancien, col_vrai, niveau=4)
    X_nouveau, y_nouveau = preparer_donnees(df_nouveau, y_pred_baseline_nouveau, col_vrai, niveau=4)
    res = entrainer_evaluer(X_ancien, y_ancien, test_size=0.2, random_state=42)

pipeline = res["pipeline"]
y_pred_nouveau = pipeline.predict(X_nouveau)
y_proba_nouveau = pipeline.predict_proba(X_nouveau)[:, 1]

acc_rf_ancien, auc_rf_ancien = res["accuracy"], res["auc"]
acc_rf_nouveau = accuracy_score(y_nouveau, y_pred_nouveau)
auc_rf_nouveau = roc_auc_score(y_nouveau, y_proba_nouveau)

# %% [markdown]
# ## 3. Récapitulatif

# %%
recap = pd.DataFrame({
    "Run": ["Ancien (test 20%)", "Nouveau (intégral)"],
    "Accuracy baseline": [acc_baseline_ancien, acc_baseline_nouveau],
    "Accuracy RF": [acc_rf_ancien, acc_rf_nouveau],
    "ROC AUC RF": [auc_rf_ancien, auc_rf_nouveau],
})
recap["Accuracy baseline"] = recap["Accuracy baseline"].map(lambda x: f"{x:.1%}")
recap["Accuracy RF"] = recap["Accuracy RF"].map(lambda x: f"{x:.1%}")
recap["ROC AUC RF"] = recap["ROC AUC RF"].round(3)
recap

# %%
print(f"Écart accuracy baseline : {(acc_baseline_nouveau - acc_baseline_ancien) * 100:+.1f} points")
print(f"Écart accuracy RF       : {(acc_rf_nouveau - acc_rf_ancien) * 100:+.1f} points")
print(f"Écart ROC AUC RF        : {auc_rf_nouveau - auc_rf_ancien:+.3f}")

# %% [markdown]
# ## 4. Détail par division COICOP (nouveau run)

# %%
correct_rf_nouveau = pd.Series(y_pred_nouveau == y_nouveau.values, index=y_nouveau.index)
division_nouveau = df_nouveau[col_vrai].astype(str).str[:2]

recap_division = pd.DataFrame({
    "n": division_nouveau.value_counts(),
    "baseline_correcte (%)": (y_nouveau.groupby(division_nouveau).mean() * 100).round(1),
    "rf_accuracy (%)": (correct_rf_nouveau.groupby(division_nouveau).mean() * 100).round(1),
}).sort_index()
recap_division

# %% [markdown]
# ## 5. Réentraînement direct sur le nouveau run (référence in-distribution)
#
# Le modèle de la section 2 est entraîné sur l'ANCIEN run et n'a jamais vu le
# nouveau — sa perte de perf peut donc venir soit d'un problème de
# généralisation du modèle, soit du fait que le nouveau run est
# intrinsèquement plus difficile à prédire (features moins informatives,
# cf. dégradation de RAG-ANN dans `recherche_ecart_performance.py`).
#
# Pour trancher : on entraîne et évalue un modèle *directement* sur le
# nouveau run (même split 80/20, `random_state=42`) — sans aucun lien avec
# l'ancien run. Si cette accuracy "in-distribution" remonte près des 86% de
# l'ancien run, le problème est bien la généralisation (il suffira de
# réentraîner avant de publier). Si elle reste basse, le nouveau run est
# intrinsèquement plus dur à prédire, peu importe sur quoi le modèle est
# entraîné.

# %%
with redirect_stdout(io.StringIO()):
    res_nouveau = entrainer_evaluer(X_nouveau, y_nouveau, test_size=0.2, random_state=42)

acc_rf_nouveau_reentraine, auc_rf_nouveau_reentraine = res_nouveau["accuracy"], res_nouveau["auc"]

recap_reentraine = pd.DataFrame({
    "Modèle": [
        "Entraîné ancien, testé ancien (test 20%)",
        "Entraîné ancien, testé nouveau (intégral)",
        "Entraîné nouveau, testé nouveau (test 20%)",
    ],
    "Accuracy RF": [acc_rf_ancien, acc_rf_nouveau, acc_rf_nouveau_reentraine],
    "ROC AUC RF": [auc_rf_ancien, auc_rf_nouveau, auc_rf_nouveau_reentraine],
})
recap_reentraine["Accuracy RF"] = recap_reentraine["Accuracy RF"].map(lambda x: f"{x:.1%}")
recap_reentraine["ROC AUC RF"] = recap_reentraine["ROC AUC RF"].round(3)
recap_reentraine

# %%
print(f"Réentraîner sur le nouveau run récupère : {(acc_rf_nouveau_reentraine - acc_rf_nouveau) * 100:+.1f} points d'accuracy")
print(f"Écart restant vs l'ancien run (in-distribution vs in-distribution) : {(acc_rf_nouveau_reentraine - acc_rf_ancien) * 100:+.1f} points")

# %%
