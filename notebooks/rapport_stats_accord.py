# %% [markdown]
# ---
# title: "Rapport stats accord — codification COICOP"
# subtitle: "Accord des classifieurs de base, faux positifs et valeur ajoutée du LLM-judge"
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
# Ce document est généré avec [Quarto](https://quarto.org) à partir de
# `notebooks/rapport_stats_accord.py` (format "percent"). Les fonctions de
# calcul vivent dans `src/stats_accord.py` et sont appelées ici en mode
# silencieux (`verbose=False`) : chaque section affiche directement les
# tableaux produits, sans dépendre de sorties texte capturées depuis la
# console Python. Les sections par niveau sont présentées sous onglets
# (niveau 4 affiché par défaut).
#
# Pour régénérer le rapport :
#
# ```bash
# quarto render notebooks/rapport_stats_accord.py --to html --output-dir ../outputs
# ```

# %% [markdown]
# ## Chargement des données

# %%
import sys
import os

sys.path.append(os.path.abspath(".."))  # accès aux modules src/ et data/

import pandas as pd
from IPython.display import display, Markdown, HTML

from data.load_data import charger_donnees
from src.stats_accord import (
    stats_accord,
    analyse_faux_positifs,
    stats_accord_avec_llm,
    stats_classifieur_seul_correct,
    stats_seul_multi_niveaux,
    recap_multi_niveaux,
    stats_seul_par_division,
)

pd.set_option("display.max_columns", None)

CHEMIN_S3 = "s3://projet-budget-famille/data/workflow_runs/2026-06-29/codif-vvkv9/decide-coicop/predictions.parquet"
df = charger_donnees(CHEMIN_S3)

cols_base = ["lcs_code", "rag_code", "ragann_code", "ttc_code_1"]
col_llm = "llm_code"
col_vrai = "code"
col_libelle = "l_pr_product"
cols_tous = cols_base + [col_llm]
niveaux = (1, 2, 3, 4)

display(Markdown(f"**{len(df)} observations** — classifieurs de base : `{', '.join(cols_base)}` — LLM-judge : `{col_llm}`"))

# %% [markdown]
# ### Libellés d'affichage
#
# Les fonctions de `src/stats_accord.py` gardent des noms techniques
# (`pct_fp_du_total`, `lcs_code_tronq`, ...) pour rester stables côté code.
# `joli()` ne fait que reformater l'affichage dans ce rapport : renommage des
# colonnes/classifieurs et pourcentages lisibles.

# %%
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
    "n": "Effectif", "pct": "%", "code": "Code",
    "vrai_tronq": "Code vrai", "code_consensus": "Consensus",
    "vrai_div": "Division vraie", "pred_div": "Division prédite",
    "division": "Division", "cas": "Cas",
    "l_pr_product": "Libellé produit",
    **LIBELLES_CLASSIFIEURS,
    **{f"{k}_tronq": v for k, v in LIBELLES_CLASSIFIEURS.items()},
}


def joli(tableau):
    """Renomme colonnes/classifieurs et formate les % pour l'affichage."""
    out = tableau.copy()
    for c in out.columns:
        if str(c).startswith("pct") and pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].map(lambda x: f"{x:.1%}" if pd.notna(x) else x)
    for c in out.select_dtypes(include="object").columns:
        out[c] = out[c].replace(LIBELLES_CLASSIFIEURS)
    out = out.rename(columns=LIBELLES_COLONNES)
    if out.index.name in LIBELLES_COLONNES:
        out.index.name = LIBELLES_COLONNES[out.index.name]
    if out.columns.name in LIBELLES_COLONNES:
        out.columns.name = LIBELLES_COLONNES[out.columns.name]
    return out

# %% [markdown]
# ## 1. Vue synthétique multi-niveaux
#
# Décomposition, pour chaque niveau de troncature COICOP, de la part des
# observations en accord correct, faux positif partagé par tous les
# classifieurs (dont le LLM), faux positif rattrapé par le LLM, ou désaccord.

# %%
recap = recap_multi_niveaux(df, cols_base, col_llm, col_vrai, niveaux=niveaux, verbose=False)
joli(recap)

# %% [markdown]
# ## 2. Rapport détaillé par niveau
#
# Pour chaque niveau : accord unanime des classifieurs de base, faux positifs
# associés, et dissociation selon que le LLM-judge suive ou non ce consensus.
# Le niveau 4 (le plus fin) est affiché par défaut.
#
# ::: {.panel-tabset}


# %%
def afficher_detail_niveau(n, top_n=10):
    df_stats = stats_accord(df, cols_base, col_vrai, niveau=n, verbose=False)
    display(Markdown("**Accord unanime des classifieurs de base**"))
    display(joli(df_stats.attrs["resume"]))

    df_fp = analyse_faux_positifs(df_stats, niveau=n, top_n=top_n, verbose=False)
    if len(df_fp):
        display(Markdown(f"**Faux positifs unanimes — {len(df_fp)} cas**"))
        display(Markdown("_Top vrais codes concernés :_"))
        display(joli(df_fp.attrs["top_vrais"]))
        display(Markdown("_Top codes prédits à tort :_"))
        display(joli(df_fp.attrs["top_predits"]))
        display(Markdown("_Top confusions (vrai → prédit) :_"))
        display(joli(df_fp.attrs["confusions"]))
        if "confusions_division" in df_fp.attrs:
            display(Markdown("_Confusions agrégées au niveau 1 (division) :_"))
            display(joli(df_fp.attrs["confusions_division"]))
    else:
        display(Markdown("*Aucun faux positif unanime à ce niveau.*"))

    df_acc = stats_accord_avec_llm(df, cols_base, col_llm, col_vrai, niveau=n, verbose=False)
    display(Markdown("**Dissociation selon le comportement du LLM-judge**"))
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
# ## 3. Cas où un seul classifieur a raison
#
# Pour chaque niveau, cas où exactement un classifieur (parmi les 4 de base
# et le LLM) donne le bon code alors que tous les autres se trompent — le
# "plafond de verre" que rattrape chaque classifieur. Le niveau 4 est affiché
# par défaut.
#
# ::: {.panel-tabset}


# %%
def afficher_seul_niveau(n):
    df_seul = stats_classifieur_seul_correct(df, cols_tous, col_vrai, niveau=n, verbose=False)
    display(joli(df_seul.attrs["resume"]))
    display(Markdown("**Répartition par classifieur sauveur**"))
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
# ## 4. Zoom niveau 4 : lignes concernées
#
# Détail ligne à ligne, pour le niveau de granularité maximale.

# %% [markdown]
# ### Faux positifs unanimes des 4 classifieurs de base

# %%
cols_fp = [c for c in [col_libelle, "code", "vrai_tronq", "code_consensus",
                       *[f"{c}_tronq" for c in cols_base]] if c in df_fp_4.columns]
display(Markdown(f"{len(df_fp_4)} lignes."))
display(HTML(
    f'<div style="max-height:500px;overflow-y:auto;border:1px solid #ddd;border-radius:6px;">'
    f'{joli(df_fp_4[cols_fp]).to_html(classes="table table-sm table-striped", index=False)}'
    f'</div>'
))

# %% [markdown]
# ### Cas où un seul classifieur a raison

# %%
df_seul_only_4 = df_seul_4[df_seul_4["seul_correct"]]
cols_seul = [c for c in [col_libelle, "code", "vrai_tronq", "classifieur_seul",
                         *[f"{c}_tronq" for c in cols_tous]] if c in df_seul_only_4.columns]
display(Markdown(f"{len(df_seul_only_4)} lignes."))
display(HTML(
    f'<div style="max-height:500px;overflow-y:auto;border:1px solid #ddd;border-radius:6px;">'
    f'{joli(df_seul_only_4[cols_seul]).to_html(classes="table table-sm table-striped", index=False)}'
    f'</div>'
))

# %% [markdown]
# ## 5. Ventilation par division COICOP — cas "un seul correct" (niveau 4)
#
# Pour chaque classifieur, dans quelles divisions COICOP se concentrent ses
# "sauvetages" (cas où lui seul trouve le bon code) ?

# %%
cross = stats_seul_par_division(df_seul_4, cols_tous, niveau_analyse=4, verbose=False)
display(Markdown("**Effectifs (divisions × classifieur sauveur)**"))
display(joli(cross))
display(Markdown("**Part de chaque division dans les sauvetages de chaque classifieur (%)**"))
display(joli(cross.attrs["parts_col"]))
display(Markdown("**Classifieur le plus souvent sauveur, par division (%)**"))
display(joli(cross.attrs["parts_lig"]))

# %%
for c in cols_tous:
    detail_c = cross.attrs["detail_par_classifieur"][c]
    display(Markdown(f"**{LIBELLES_CLASSIFIEURS[c]}** — {len(detail_c)} division(s) concernée(s)"))
    if len(detail_c):
        display(joli(detail_c))
