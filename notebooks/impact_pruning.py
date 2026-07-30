# %% [markdown]
# # Impact du pruning du `.0` terminal dans `tronquer_niveau`
#
# `tronquer_niveau` (`src/coicop.py`) retire depuis peu le `.0` terminal en
# cascade (convention COICOP "pas de sous-classe/groupe plus précise", ex.
# `13.9.0 == 13.9`) — aussi bien sur les codes prédits que sur le vrai code,
# puisque c'est la même fonction qui tronque les deux.
#
# Ce notebook compare, AVANT / APRÈS ce pruning, sur les deux runs
# disponibles :
# 1. l'accuracy globale de chaque classifieur (niveau 4) ;
# 2. l'accuracy de la baseline (vote majoritaire, TTC arbitre) ;
# 3. la modélisation RF complète (`baseline_correcte`).
#
# `AVANT` est reconstitué localement ci-dessous (ancienne version de
# `tronquer_niveau`, sans le strip) — le code de `src/` n'est pas modifié,
# seule sa fonction `tronquer_niveau` est temporairement remplacée en
# mémoire pour la comparaison, puis restaurée à la fin.

# %% Imports
import sys, os
sys.path.append(os.path.abspath(".."))

import io
from contextlib import redirect_stdout

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

import src.coicop as coicop_mod
import src.baseline.ttc as ttc_mod
import src.baseline.preprocessing as preprocessing_mod
import src.stats_accord as stats_accord_mod

from data.load_data import charger_donnees, CHEMIN_S3_MODELISATION
from src.baseline import (
    baseline_majorite_ttc, evaluer_baseline, preparer_donnees, entrainer_evaluer,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# %% Les deux versions de tronquer_niveau
APRES_PRUNING = coicop_mod.tronquer_niveau  # version actuelle de src/coicop.py (avec le strip)


def AVANT_PRUNING(code, niveau=4):
    """Reproduction de l'ancienne `tronquer_niveau`, sans le strip du `.0` terminal."""
    if pd.isna(code):
        return code
    s = str(code)
    if s in ("AUCUNE_SUGGESTION", "NON_CODABLE"):
        return s
    n_chiffres_cible = niveau + 1
    chiffres = 0
    out = []
    for c in s:
        if c.isdigit():
            if chiffres == n_chiffres_cible:
                break
            chiffres += 1
            out.append(c)
        else:
            if chiffres < n_chiffres_cible:
                out.append(c)
    return "".join(out)


def basculer_pruning(fn):
    """Remplace `tronquer_niveau` dans tous les modules du projet qui l'utilisent
    (import direct par valeur, donc `src.coicop.tronquer_niveau = fn` seul ne suffit
    pas partout)."""
    coicop_mod.tronquer_niveau = fn
    ttc_mod.tronquer_niveau = fn
    preprocessing_mod.tronquer_niveau = fn
    stats_accord_mod.tronquer_niveau = fn


# %% Chargement des deux runs
CHEMIN_ANCIEN = CHEMIN_S3_MODELISATION  # codif-vvkv9, 2026-06-29
CHEMIN_NOUVEAU = (
    "s3://projet-budget-famille/data/workflow_runs/2026-07-23/codif-8m8jn"
    "/decide-coicop/predictions.parquet"
)

cols_classifieurs = ["lcs_code", "rag_code", "ragann_code", "ttc_code_1", "llm_code"]
cols_base = ["lcs_code", "rag_code", "ragann_code", "ttc_code_1"]
col_ttc = "ttc_code_1"
col_vrai = "code"

with redirect_stdout(io.StringIO()):
    df_ancien = charger_donnees(CHEMIN_ANCIEN)
    df_nouveau = charger_donnees(CHEMIN_NOUVEAU)

print(f"Ancien run  (29/06, codif-vvkv9) : {len(df_ancien)} lignes")
print(f"Nouveau run (23/07, codif-8m8jn) : {len(df_nouveau)} lignes")

# %% [markdown]
# ## 1. Accuracy globale par classifieur (niveau 4)
#
# `pred_tronqué == vrai_tronqué` — exactement le calcul utilisé par
# `accuracy_multi_classifieurs`/`stats_accord` (donc par les pages
# stats-descriptives du site).

# %%
def accuracy_par_classifieur(df, fn, niveau=4):
    vrai_tronq = df[col_vrai].map(lambda x: fn(x, niveau))
    return {
        c: (df[c].map(lambda x: fn(x, niveau)) == vrai_tronq).mean()
        for c in cols_classifieurs
    }


lignes = []
for nom_run, df in [("Ancien (29/06)", df_ancien), ("Nouveau (23/07)", df_nouveau)]:
    avant = accuracy_par_classifieur(df, AVANT_PRUNING)
    apres = accuracy_par_classifieur(df, APRES_PRUNING)
    for c in cols_classifieurs:
        lignes.append({
            "run": nom_run, "classifieur": c,
            "avant": avant[c], "apres": apres[c],
            "ecart_pts": (apres[c] - avant[c]) * 100,
        })

recap_classifieurs = pd.DataFrame(lignes).set_index(["run", "classifieur"])
recap_classifieurs["avant"] = recap_classifieurs["avant"].map(lambda x: f"{x:.1%}")
recap_classifieurs["apres"] = recap_classifieurs["apres"].map(lambda x: f"{x:.1%}")
recap_classifieurs["ecart_pts"] = recap_classifieurs["ecart_pts"].round(1)
print(recap_classifieurs.to_string())
recap_classifieurs

# %% [markdown]
# ## 2. Accuracy de la baseline (vote majoritaire, TTC arbitre)

# %%
def accuracy_baseline(df, fn):
    basculer_pruning(fn)
    with redirect_stdout(io.StringIO()):
        y_pred = baseline_majorite_ttc(df, cols_base, col_ttc, niveau=4)
        return evaluer_baseline(df, y_pred, col_vrai, niveau=4)


lignes_baseline = []
for nom_run, df in [("Ancien (29/06)", df_ancien), ("Nouveau (23/07)", df_nouveau)]:
    acc_avant = accuracy_baseline(df, AVANT_PRUNING)
    acc_apres = accuracy_baseline(df, APRES_PRUNING)
    lignes_baseline.append({
        "run": nom_run, "avant": acc_avant, "apres": acc_apres,
        "ecart_pts": (acc_apres - acc_avant) * 100,
    })

recap_baseline = pd.DataFrame(lignes_baseline).set_index("run")
recap_baseline["avant"] = recap_baseline["avant"].map(lambda x: f"{x:.1%}")
recap_baseline["apres"] = recap_baseline["apres"].map(lambda x: f"{x:.1%}")
recap_baseline["ecart_pts"] = recap_baseline["ecart_pts"].round(1)
print(recap_baseline.to_string())
recap_baseline

# %% [markdown]
# ## 3. Modélisation RF complète (`baseline_correcte`)
#
# Même protocole que `generalisation_nouveau_run.py` : le pipeline est
# entraîné sur 80% de l'ancien run (`test_size=0.2`, `random_state=42`),
# évalué sur les 20% restants, puis appliqué tel quel (sans réentraînement)
# à l'intégralité du nouveau run.

# %%
def modelisation(fn):
    basculer_pruning(fn)
    with redirect_stdout(io.StringIO()):
        y_pred_baseline_ancien = baseline_majorite_ttc(df_ancien, cols_base, col_ttc, niveau=4)
        y_pred_baseline_nouveau = baseline_majorite_ttc(df_nouveau, cols_base, col_ttc, niveau=4)
        X_ancien, y_ancien = preparer_donnees(
            df_ancien, y_pred_baseline_ancien, col_vrai, niveau=4,
        )
        X_nouveau, y_nouveau = preparer_donnees(
            df_nouveau, y_pred_baseline_nouveau, col_vrai, niveau=4,
        )
        res = entrainer_evaluer(X_ancien, y_ancien, test_size=0.2, random_state=42)

    pipeline = res["pipeline"]
    y_pred_nouveau = pipeline.predict(X_nouveau)
    y_proba_nouveau = pipeline.predict_proba(X_nouveau)[:, 1]

    return {
        "acc_ancien": res["accuracy"],
        "auc_ancien": res["auc"],
        "acc_nouveau": accuracy_score(y_nouveau, y_pred_nouveau),
        "auc_nouveau": roc_auc_score(y_nouveau, y_proba_nouveau),
    }


modele_avant = modelisation(AVANT_PRUNING)
modele_apres = modelisation(APRES_PRUNING)

recap_modelisation = pd.DataFrame({
    "Ancien (test 20%) - accuracy": [modele_avant["acc_ancien"], modele_apres["acc_ancien"]],
    "Ancien (test 20%) - ROC AUC": [modele_avant["auc_ancien"], modele_apres["auc_ancien"]],
    "Nouveau (intégral) - accuracy": [modele_avant["acc_nouveau"], modele_apres["acc_nouveau"]],
    "Nouveau (intégral) - ROC AUC": [modele_avant["auc_nouveau"], modele_apres["auc_nouveau"]],
}, index=["Avant pruning", "Après pruning"])
print(recap_modelisation.to_string())
recap_modelisation

# %%
ecart_ancien = (modele_apres["acc_ancien"] - modele_avant["acc_ancien"]) * 100
ecart_nouveau = (modele_apres["acc_nouveau"] - modele_avant["acc_nouveau"]) * 100
print(f"RF - Ancien  : {ecart_ancien:+.1f} pts d'accuracy")
print(f"RF - Nouveau : {ecart_nouveau:+.1f} pts d'accuracy")

# %% Restauration de l'etat normal du module (post-fix, celui utilise par le reste du projet)
basculer_pruning(APRES_PRUNING)
