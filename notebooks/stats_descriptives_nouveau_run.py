# quarto render notebooks/stats_descriptives_nouveau_run.py --to html --output-dir ../outputs

# %% [markdown]
# ---
# title: "Stats descriptives — nouveau run (2026-07-30, codif-x98xl)"
# subtitle: "Même rapport que `rapport_stats_accord.py`, + validation de `code_lvl4`"
# date: today
# format:
#   html:
#     toc: true
#     toc-depth: 3
#     embed-resources: true
#     theme: flatly
#     code-fold: true
#     code-summary: "Voir le code"
#     df-print: default
# jupyter: python3
# execute:
#   warning: false
# ---
#
# Reprend exactement le rapport de `rapport_stats_accord.py` (accords
# unanimes, 3 contre 1, seul correct, accuracy par division...), mais sur le
# nouveau run du 2026-07-30 (`codif-x98xl`). Ce run ne touche pas à
# `data/load_data.py` (donc rien ne change pour le site tant qu'on ne décide
# pas de publier) : le chemin est défini ici, en local.
#
# Ce run introduit une nouvelle colonne, `code_lvl4` : le vrai code
# apparemment déjà tronqué/élagué au niveau 4 en amont (donc censé être
# l'équivalent de `tronquer_niveau(code, niveau=4)` après notre fix côté
# `src/coicop.py`). Une section de validation en tout début et en toute fin
# de notebook vérifie que les deux approches — `code` passé à notre
# `tronquer_niveau`, vs. `code_lvl4` utilisé tel quel — donnent bien les
# mêmes résultats.

# %%
import sys
import os

sys.path.append(os.path.abspath(".."))  # accès aux modules src/ et data/

import pandas as pd
from IPython.display import display, Markdown, HTML

from data.load_data import charger_donnees
from src.coicop import tronquer_niveau, niveau_atteint
from src.stats_accord import (
    stats_accord,
    analyse_faux_positifs,
    stats_accord_avec_llm,
    stats_classifieur_seul_correct,
    stats_majorite_3_1,
    recap_multi_niveaux,
    stats_seul_par_division,
    accuracy_par_division,
    accuracy_multi_classifieurs,
    precision_par_division_llm,
)

pd.set_option("display.max_columns", None)

# %% Chemin du nouveau run (en local, ne touche pas data/load_data.py)
CHEMIN_S3 = (
    "s3://projet-budget-famille/data/workflow_runs/2026-07-30/codif-x98xl"
    "/decide-coicop/predictions.parquet"
)
df = charger_donnees(CHEMIN_S3)

cols_base = ["lcs_code", "rag_code", "ragann_code", "ttc_code_1"]
col_llm = "llm_code"
col_vrai = "code"
col_libelle = "l_pr_product"
cols_tous = cols_base + [col_llm]
niveaux = (1, 2, 3, 4)

display(Markdown(
    f"**{len(df)} observations** — classifieurs de base : `{', '.join(cols_base)}` — "
    f"LLM-judge : `{col_llm}`"
))

LIBELLES_CLASSIFIEURS = {
    "lcs_code": "LCS", "rag_code": "RAG", "ragann_code": "RAG-ANN",
    "ttc_code_1": "TTC", "llm_code": "LLM-judge",
}
LIBELLES_COLONNES = {
    "niveau": "Niveau", "n_total": "Total", "n_accord": "Accord unanime",
    "pct_accord": "% accord", "n_correct": "Corrects",
    "pct_correct": "% corrects (accords)",
    "pct_correct_des_accords": "% corrects (accords)",
    "n_fp": "Faux positifs", "n_fp_base": "Faux positifs",
    "pct_fp_des_accords": "% FP (accords)", "pct_fp_du_total": "% FP (total)",
    "n_fp_5_5": "FP partagés (+LLM)", "n_llm_sauve": "LLM rattrape",
    "n_seul_correct": "Un seul correct", "pct_seul_correct": "% un seul correct",
    "classifieur": "Classifieur", "classifieur_seul": "Classifieur seul correct",
    "n_3_1": "Cas 3 contre 1", "pct_3_1": "% cas 3 contre 1",
    "classifieur_dissident": "Classifieur dissident",
    "code_majorite": "Code majorité", "code_minoritaire": "Code dissident",
    "majorite_correcte": "Majorité correcte", "minorite_correcte": "Dissident correct",
    "n_majorite_correcte": "Majorité correcte", "pct_majorite_correcte": "% majorité correcte",
    "n_minorite_correcte": "Dissident correct", "pct_minorite_correcte": "% dissident correct",
    "n_aucun_correct": "Personne correct", "pct_aucun_correct": "% personne correct",
    "n": "Effectif", "pct": "%", "code": "Code",
    "vrai_tronq": "Code vrai", "code_consensus": "Consensus",
    "vrai_div": "Division vraie", "pred_div": "Division prédite",
    "division": "Division", "cas": "Cas",
    "l_pr_product": "Libellé produit",
    "accuracy": "Accuracy", "accuracy_hors_exclus": "Accuracy (hors 98/99)",
    "n_exclus": "Exclus (98/99)", "n_hors_exclus": "Total (hors 98/99)",
    "n_correct_hors_exclus": "Corrects (hors 98/99)",
    "n_erreurs": "Erreurs", "precision": "Précision",
    "pred_division": "Division prédite (LLM)",
    "pct_profondeur_1": "% profondeur 1", "pct_profondeur_2": "% profondeur 2",
    "pct_profondeur_3": "% profondeur 3", "pct_profondeur_4": "% profondeur 4",
    "pct_profondeur_5_plus": "% profondeur 5+",
    **LIBELLES_CLASSIFIEURS,
    **{f"{k}_tronq": v for k, v in LIBELLES_CLASSIFIEURS.items()},
    **{f"accuracy_{k}": f"Accuracy {v}" for k, v in LIBELLES_CLASSIFIEURS.items()},
}


def joli(tableau):
    """Renomme colonnes/classifieurs et formate les % pour l'affichage."""
    out = tableau.copy()
    for c in out.columns:
        if str(c).startswith(("pct", "accuracy", "precision")) and pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].map(lambda x: f"{x:.1%}" if pd.notna(x) else x)
    for c in out.select_dtypes(include="object").columns:
        out[c] = out[c].replace(LIBELLES_CLASSIFIEURS)
    out = out.rename(columns=LIBELLES_COLONNES)
    if out.index.name in LIBELLES_COLONNES:
        out.index.name = LIBELLES_COLONNES[out.index.name]
    if out.columns.name in LIBELLES_COLONNES:
        out.columns.name = LIBELLES_COLONNES[out.columns.name]
    return out


def titre(texte):
    """Sous-titre (gras) précédant un tableau, avec un espace plus
    au-dessus pour le détacher du tableau précédent."""
    display(HTML(f'<p style="margin-top:2em; margin-bottom:0.4em; font-weight:bold;">{texte}</p>'))


def sous_titre(texte):
    """Variante italique de `titre`, pour les libellés secondaires."""
    display(HTML(f'<p style="margin-top:1.3em; margin-bottom:0.3em; font-style:italic;">{texte}</p>'))


# %% [markdown]
# ## 0. Validation de `code_lvl4` vs `tronquer_niveau(code, niveau=4)`
#
# Le run fournit désormais `code_lvl4`, censé être le vrai code déjà tronqué
# et élagué au niveau 4 par le pipeline en amont. On vérifie qu'il coïncide
# avec notre propre `tronquer_niveau(code, niveau=4)` (qui inclut le fix de
# pruning du `.0` terminal) avant de s'appuyer dessus.

# %%
notre_tronq = df[col_vrai].map(lambda x: tronquer_niveau(x, niveau=4))
match_code_lvl4 = (df["code_lvl4"].astype(str) == notre_tronq.astype(str))

display(Markdown(
    f"**Concordance `code_lvl4` == `tronquer_niveau(code, 4)`** : "
    f"{match_code_lvl4.mean():.1%} ({int(match_code_lvl4.sum())} / {len(df)})"
))

if not match_code_lvl4.all():
    titre(f"Désaccords — {(~match_code_lvl4).sum()} cas")
    display(df.loc[~match_code_lvl4, [col_vrai, "code_lvl4"]].assign(notre_tronq=notre_tronq[~match_code_lvl4]))
else:
    display(Markdown("*Aucun désaccord : les deux approches sont strictement équivalentes sur ce run.*"))

# %% [markdown]
# La suite du rapport utilise `col_vrai = "code"` (donc notre
# `tronquer_niveau`, déjà pruné) — la section de validation finale (tout en
# bas) rejoue les chiffres-clés avec `col_vrai = "code_lvl4"` pour confirmer
# qu'ils sont identiques.

# %% [markdown]
# ## 1. Vue synthétique multi-niveaux
#
# Décomposition, pour chaque niveau de troncature COICOP, de la part des
# observations en accord correct unanime, faux positif partagé par tous les
# classifieurs (dont le LLM), faux positif rattrapé par le LLM, ou désaccord.

# %%
recap = recap_multi_niveaux(df, cols_base, col_llm, col_vrai, niveaux=niveaux, verbose=False)
joli(recap)

# %% [markdown]
# ## 1bis. Accuracy du LLM-judge vs vérité terrain, par division (niveau 1)
#
# Accuracy de `llm_code` contre `code` au niveau 4 (granularité max), ventilée
# par division COICOP (niveau 1 du vrai code). En regard, l'accuracy obtenue
# en excluant les lignes où le LLM classe en division "98" (indéterminé /
# illisible) ou "99" (hors COICOP : dons, impôts, opérations bancaires...).

# %%
recap_accuracy_llm = accuracy_par_division(
    df, col_pred=col_llm, col_vrai=col_vrai, niveau=4, codes_exclus=("98", "99"), verbose=False,
)
joli(recap_accuracy_llm)

# %% [markdown]
# ## 1ter. Précision du LLM-judge par division PRÉDITE

# %%
sous_titre("Correction jugée au niveau 1 (division)")
recap_precision_llm_n1 = precision_par_division_llm(
    df, col_pred=col_llm, col_vrai=col_vrai, niveau=1, codes_exclus=("98", "99"), verbose=False,
)
display(joli(recap_precision_llm_n1))

sous_titre("Correction jugée au niveau 4 (granularité max)")
recap_precision_llm_n4 = precision_par_division_llm(
    df, col_pred=col_llm, col_vrai=col_vrai, niveau=4, codes_exclus=("98", "99"), verbose=False,
)
display(joli(recap_precision_llm_n4))

# %% [markdown]
# ## 1quater. Accuracy comparée de tous les classifieurs, par division

# %%
sous_titre("Correction jugée au niveau 1 (division)")
recap_accuracy_tous_n1 = accuracy_multi_classifieurs(df, cols_tous, col_vrai, niveau=1, verbose=False)
display(joli(recap_accuracy_tous_n1))

sous_titre("Correction jugée au niveau 4 (granularité max)")
recap_accuracy_tous_n4 = accuracy_multi_classifieurs(df, cols_tous, col_vrai, niveau=4, verbose=False)
display(joli(recap_accuracy_tous_n4))

# %% [markdown]
# ## 2. Rapport sur les accords unanimes des 4 classifieurs
#
# ::: {.panel-tabset}


# %%
def afficher_detail_niveau(n, top_n=10):
    df_stats = stats_accord(df, cols_base, col_vrai, niveau=n, verbose=False)
    titre("Accord unanime des classifieurs de base")
    display(joli(df_stats.attrs["resume"]))

    df_fp = analyse_faux_positifs(df_stats, niveau=n, top_n=top_n, verbose=False)
    if len(df_fp):
        titre(f"Faux positifs unanimes — {len(df_fp)} cas")
        sous_titre("Top vrais codes concernés :")
        display(joli(df_fp.attrs["top_vrais"]))
        sous_titre("Top codes prédits à tort :")
        display(joli(df_fp.attrs["top_predits"]))
        sous_titre("Top confusions (vrai → prédit) :")
        display(joli(df_fp.attrs["confusions"]))
        if "confusions_division" in df_fp.attrs:
            sous_titre("Confusions agrégées au niveau 1 (division) :")
            display(joli(df_fp.attrs["confusions_division"]))
    else:
        display(Markdown("*Aucun faux positif unanime à ce niveau.*"))

    df_acc = stats_accord_avec_llm(df, cols_base, col_llm, col_vrai, niveau=n, verbose=False)
    titre("Dissociation selon le comportement du LLM-judge")
    display(joli(df_acc.attrs["recap"]))
    return df_stats, df_fp, df_acc


# %% [markdown]
# ### Niveau 4 — sous-classe

# %%
df_stats_4, df_fp_4, df_acc_4 = afficher_detail_niveau(4)

# %% [markdown]
# ### Niveau 3 — classe

# %%
df_stats_3, df_fp_3, df_acc_3 = afficher_detail_niveau(3)

# %% [markdown]
# ### Niveau 2 — groupe

# %%
df_stats_2, df_fp_2, df_acc_2 = afficher_detail_niveau(2)

# %% [markdown]
# ### Niveau 1 — division

# %%
df_stats_1, df_fp_1, df_acc_1 = afficher_detail_niveau(1)

# %% [markdown]
# :::

# %% [markdown]
# ## 3. Cas où 3 classifieurs de base sont d'accord contre 1 dissident
#
# ::: {.panel-tabset}


# %%
def afficher_3_1_niveau(n):
    df_31 = stats_majorite_3_1(df, cols_base, col_vrai, niveau=n, verbose=False)
    display(joli(df_31.attrs["resume"]))
    titre("Répartition par classifieur dissident")
    display(joli(df_31.attrs["repart_dissident"]))
    return df_31


# %% [markdown]
# ### Niveau 4 — sous-classe

# %%
df_31_4 = afficher_3_1_niveau(4)

# %% [markdown]
# ### Niveau 3 — classe

# %%
df_31_3 = afficher_3_1_niveau(3)

# %% [markdown]
# ### Niveau 2 — groupe

# %%
df_31_2 = afficher_3_1_niveau(2)

# %% [markdown]
# ### Niveau 1 — division

# %%
df_31_1 = afficher_3_1_niveau(1)

# %% [markdown]
# :::

# %% [markdown]
# ## 4. Cas où un seul classifieur a raison
#
# ::: {.panel-tabset}


# %%
def afficher_seul_niveau(n):
    df_seul = stats_classifieur_seul_correct(df, cols_tous, col_vrai, niveau=n, verbose=False)
    display(joli(df_seul.attrs["resume"]))
    titre("Répartition par classifieur sauveur")
    display(joli(df_seul.attrs["repart"]))
    return df_seul


# %% [markdown]
# ### Niveau 4 — sous-classe

# %%
df_seul_4 = afficher_seul_niveau(4)

# %% [markdown]
# ### Niveau 3 — classe

# %%
df_seul_3 = afficher_seul_niveau(3)

# %% [markdown]
# ### Niveau 2 — groupe

# %%
df_seul_2 = afficher_seul_niveau(2)

# %% [markdown]
# ### Niveau 1 — division

# %%
df_seul_1 = afficher_seul_niveau(1)

# %% [markdown]
# :::

# %% [markdown]
# ## 5. Ventilation par division COICOP — cas "un seul correct" (niveau 4)

# %%
cross = stats_seul_par_division(df_seul_4, cols_tous, niveau_analyse=4, verbose=False)
titre("Effectifs (divisions × classifieur sauveur)")
display(joli(cross))
titre("Part de chaque division dans les sauvetages de chaque classifieur (%)")
display(joli(cross.attrs["parts_col"]))
titre("Classifieur le plus souvent sauveur, par division (%)")
display(joli(cross.attrs["parts_lig"]))

# %% [markdown]
# ## 6. Profondeur réelle des codes, par classifieur
#
# Complément à la section 0 : répartition de la profondeur effective de
# chaque code (avant toute troncature), en part du total des observations.

# %%
cols_codes = [col_vrai, "code_lvl4", *cols_base, "ttc_code_2", "ttc_code_3", col_llm]
cols_codes = [c for c in cols_codes if c in df.columns]


def stats_profondeur(df, colonnes):
    n_total = len(df)
    lignes = []
    for c in colonnes:
        non_na = df[c].dropna()
        sentinelles = non_na.isin(["AUCUNE_SUGGESTION", "NON_CODABLE"])
        codes_reels = non_na[~sentinelles]
        prof = codes_reels.map(niveau_atteint)
        lignes.append({
            "classifieur": c,
            "pct_absent": df[c].isna().sum() / n_total,
            "pct_profondeur_1": (prof == 1).sum() / n_total,
            "pct_profondeur_2": (prof == 2).sum() / n_total,
            "pct_profondeur_3": (prof == 3).sum() / n_total,
            "pct_profondeur_4": (prof == 4).sum() / n_total,
            "pct_profondeur_5_plus": (prof >= 5).sum() / n_total,
        })
    out = pd.DataFrame(lignes).set_index("classifieur")
    out.index = out.index.map(lambda c: LIBELLES_CLASSIFIEURS.get(c, c))
    out.index.name = "classifieur"
    return out


joli(stats_profondeur(df, cols_codes).round(4))

# %% [markdown]
# ## 7. Validation finale : `col_vrai = "code_lvl4"` vs `col_vrai = "code"`
#
# On rejoue les deux tableaux les plus synthétiques (section 1 et 1quater)
# en utilisant directement `code_lvl4` comme vérité terrain, sans passer par
# notre `tronquer_niveau` — pour confirmer qu'ils sont bien identiques à ce
# qui précède.

# %%
col_vrai_lvl4 = "code_lvl4"

recap_lvl4 = recap_multi_niveaux(df, cols_base, col_llm, col_vrai_lvl4, niveaux=niveaux, verbose=False)
titre("recap_multi_niveaux — avec code_lvl4")
display(joli(recap_lvl4))

identique_recap = recap.equals(recap_lvl4)
display(Markdown(f"**Identique à la section 1 (`col_vrai=\"code\"`)** : {identique_recap}"))

# %%
recap_accuracy_tous_n4_lvl4 = accuracy_multi_classifieurs(
    df, cols_tous, col_vrai_lvl4, niveau=4, verbose=False,
)
titre("accuracy_multi_classifieurs (niveau 4) — avec code_lvl4")
display(joli(recap_accuracy_tous_n4_lvl4))

identique_accuracy = recap_accuracy_tous_n4.equals(recap_accuracy_tous_n4_lvl4)
display(Markdown(f"**Identique à la section 1quater (`col_vrai=\"code\"`)** : {identique_accuracy}"))