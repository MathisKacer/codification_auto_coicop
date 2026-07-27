# %% [markdown]
# # Recherche : écart de performance ancien run (codif-vvkv9) vs nouveau run (codif-8m8jn)

# %% Imports
import sys, os
sys.path.append(os.path.abspath(".."))

import pandas as pd
from sklearn.metrics import accuracy_score

from data.load_data import charger_donnees, CHEMIN_S3_MODELISATION
from src.coicop import tronquer_niveau, niveau_atteint, a_raison_jusqu_a_niveau
from src.baseline import baseline_majorite_ttc, preparer_donnees, entrainer_evaluer

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# %% Chemins des deux runs
CHEMIN_ANCIEN = CHEMIN_S3_MODELISATION
CHEMIN_NOUVEAU = (
    "s3://projet-budget-famille/data/workflow_runs/2026-07-23/codif-8m8jn"
    "/decide-coicop/predictions.parquet"
)

cols_base = ["lcs_code", "rag_code", "ragann_code", "ttc_code_1"]
col_ttc = "ttc_code_1"
col_vrai = "code"

# %% Chargement
df_ancien = charger_donnees(CHEMIN_ANCIEN)
df_nouveau = charger_donnees(CHEMIN_NOUVEAU)

# %% [markdown]
# ## 1. Fuite train/test (ancien run)

# %%
y_pred_baseline_ancien = baseline_majorite_ttc(df_ancien, cols_base, col_ttc, niveau=4)
X_ancien, y_ancien = preparer_donnees(df_ancien, y_pred_baseline_ancien, col_vrai, niveau=4)
res = entrainer_evaluer(X_ancien, y_ancien, test_size=0.2, random_state=42)

# %%
raw_product = df_ancien["raw_product"]
rp_train = set(raw_product.loc[res["X_train"].index])
fuite_test = raw_product.loc[res["X_test"].index].isin(rp_train).values

acc_fuite = accuracy_score(res["y_test"].values[fuite_test], res["y_pred"][fuite_test])
acc_sans_fuite = accuracy_score(res["y_test"].values[~fuite_test], res["y_pred"][~fuite_test])

recap_fuite = pd.DataFrame({
    "n": [raw_product.duplicated().sum(), fuite_test.sum(), (~fuite_test).sum()],
    "accuracy": [None, acc_fuite, acc_sans_fuite],
}, index=["raw_product dupliqués (ancien run)", "test — raw_product vu en train", "test — raw_product inédit"])
recap_fuite

# %% [markdown]
# ## 2. Composition des deux runs

# %%
pd.concat({
    "ancien": df_ancien["source"].value_counts(normalize=True) * 100,
    "nouveau": df_nouveau["source"].value_counts(normalize=True) * 100,
}, axis=1).round(1)

# %%
div_a = df_ancien[col_vrai].astype(str).str[:2].value_counts(normalize=True) * 100
div_n = df_nouveau[col_vrai].astype(str).str[:2].value_counts(normalize=True) * 100
pd.concat({"ancien": div_a, "nouveau": div_n}, axis=1).round(1).sort_index()

# %%
pd.concat({"ancien": df_ancien["budget"].describe(), "nouveau": df_nouveau["budget"].describe()}, axis=1).round(2)

# %% [markdown]
# ## 3. Accuracy par classifieur de base

# %%
lignes = []
for c in cols_base:
    acc_a = (df_ancien[c].map(lambda x: tronquer_niveau(x, niveau=4)) == df_ancien[col_vrai].map(lambda x: tronquer_niveau(x, niveau=4))).mean()
    acc_n = (df_nouveau[c].map(lambda x: tronquer_niveau(x, niveau=4)) == df_nouveau[col_vrai].map(lambda x: tronquer_niveau(x, niveau=4))).mean()
    lignes.append({"classifieur": c, "accuracy_ancien": acc_a, "accuracy_nouveau": acc_n, "ecart_pts": (acc_n - acc_a) * 100})
recap_classifieurs = pd.DataFrame(lignes).round(3)
recap_classifieurs

# %%
pd.DataFrame({
    "ancien": [(~df_ancien["ragann_codable"]).mean()],
    "nouveau": [(~df_nouveau["ragann_codable"]).mean()],
}, index=["ragann_non_codable"]).round(3)

# %% [markdown]
# ## 4. Profondeur des codes

# %%
cols_codes = ["code", "lcs_code", "rag_code", "ragann_code", "ttc_code_1", "ttc_code_2", "ttc_code_3", "llm_code"]

lignes_prof = []
for name, df in [("ancien", df_ancien), ("nouveau", df_nouveau)]:
    for c in cols_codes:
        if c not in df.columns:
            continue
        non_na = df[c].dropna()
        codes_reels = non_na[~non_na.isin(["AUCUNE_SUGGESTION", "NON_CODABLE"])]
        prof = codes_reels.map(niveau_atteint)
        lignes_prof.append({
            "run": name, "colonne": c,
            "pct_profondeur_sous_4": round((prof < 4).mean() * 100, 1),
            "pct_profondeur_sup_4": round((prof > 4).mean() * 100, 1),
            "n_total": len(codes_reels),
        })

recap_profondeur = pd.DataFrame(lignes_prof).set_index(["colonne", "run"]).reindex(cols_codes, level="colonne")
recap_profondeur

# %%
lignes_plafond = []
for name, df in [("ancien", df_ancien), ("nouveau", df_nouveau)]:
    for c in cols_base:
        vrai_tronq = df[col_vrai].map(lambda x: tronquer_niveau(x, niveau=4))
        pred_tronq = df[c].map(lambda x: tronquer_niveau(x, niveau=4))
        acc_naive = (pred_tronq == vrai_tronq).mean()
        acc_plafonne = df.apply(lambda row, c=c: a_raison_jusqu_a_niveau(row[c], row[col_vrai], niveau_max=4), axis=1).mean()
        lignes_plafond.append({
            "run": name, "classifieur": c,
            "accuracy_naive": round(acc_naive, 3), "accuracy_plafonnee": round(acc_plafonne, 3),
            "ecart_pts": round((acc_plafonne - acc_naive) * 100, 1),
        })
pd.DataFrame(lignes_plafond)

# %% [markdown]
# ## 5. Convention COICOP : "0" en 4e position = pas de sous-classe plus précise
#
# Un classifieur qui s'arrête avant le niveau 4 alors que le 4e segment de la
# vérité vaut "0" (ex. `01.2.1` vs vérité `01.2.1.0.1`, tronquée à
# `01.2.1.0`) n'est pas réellement en tort : "0" signifie "pas de sous-classe
# plus précise", donc la comparaison doit se faire au niveau 3 dans ce cas
# précis (le classifieur n'a rien affirmé de faux, il n'a juste pas ajouté un
# 4e segment qui n'aurait de toute façon rien apporté).

# %%
def a_raison_convention_zero(code_classifieur, code_verite, niveau_max=4):
    if pd.isna(code_classifieur) or pd.isna(code_verite):
        return False
    vrai_tr = tronquer_niveau(code_verite, niveau_max)
    pred_tr = tronquer_niveau(code_classifieur, niveau_max)
    if pred_tr == vrai_tr:
        return True
    segments_vrai = vrai_tr.split(".")
    if (
        len(segments_vrai) >= niveau_max
        and segments_vrai[niveau_max - 1] == "0"
        and niveau_atteint(code_classifieur) < niveau_max
    ):
        return tronquer_niveau(code_classifieur, niveau_max - 1) == tronquer_niveau(code_verite, niveau_max - 1)
    return False


lignes_zero = []
for name, df in [("ancien", df_ancien), ("nouveau", df_nouveau)]:
    for c in cols_base:
        vrai_tronq = df[col_vrai].map(lambda x: tronquer_niveau(x, niveau=4))
        pred_tronq = df[c].map(lambda x: tronquer_niveau(x, niveau=4))
        acc_naive = (pred_tronq == vrai_tronq).mean()
        correct_zero = df.apply(lambda row, c=c: a_raison_convention_zero(row[c], row[col_vrai], niveau_max=4), axis=1)
        acc_zero = correct_zero.mean()
        n_recupere = int((correct_zero & (pred_tronq != vrai_tronq)).sum())
        lignes_zero.append({
            "run": name, "classifieur": c,
            "accuracy_naive": round(acc_naive, 3), "accuracy_convention_zero": round(acc_zero, 3),
            "ecart_pts": round((acc_zero - acc_naive) * 100, 1), "n_recupere": n_recupere,
        })
recap_zero = pd.DataFrame(lignes_zero).pivot(index="classifieur", columns="run", values=["accuracy_naive", "accuracy_convention_zero"])
recap_zero["ecart_naive_pts"] = (recap_zero[("accuracy_naive", "nouveau")] - recap_zero[("accuracy_naive", "ancien")]) * 100
recap_zero["ecart_convention_zero_pts"] = (recap_zero[("accuracy_convention_zero", "nouveau")] - recap_zero[("accuracy_convention_zero", "ancien")]) * 100
recap_zero

# %%
