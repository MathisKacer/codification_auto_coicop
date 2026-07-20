# quarto render notebooks/rapport_stats_accord.py --to html --output-dir ../outputs

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
# Rapport donnant des statistiques descriptives jusqu'au niveau 4 sur les
# résultats de la classification automatique de la COICOP. Il permet
# notamment d'établir des situations précises et de voir les performances de
# chaque classifieur dans celles-ci. Il est reproductible pour chaque
# nouveau run.

# %%
import sys
import os

sys.path.append(os.path.abspath(".."))  # accès aux modules src/ et data/

import pandas as pd
from IPython.display import display, Markdown, HTML

from data.load_data import charger_donnees, CHEMIN_S3_STATS_DESCRIPTIVES
from src.stats_accord import (
    stats_accord,
    analyse_faux_positifs,
    stats_accord_avec_llm,
    stats_classifieur_seul_correct,
    stats_seul_multi_niveaux,
    stats_majorite_3_1,
    recap_multi_niveaux,
    recap_3_1_multi_niveaux,
    stats_seul_par_division,
    accuracy_par_division,
    accuracy_multi_classifieurs,
    precision_par_division_llm,
)

pd.set_option("display.max_columns", None)

CHEMIN_S3 = CHEMIN_S3_STATS_DESCRIPTIVES
df = charger_donnees(CHEMIN_S3)

cols_base = ["lcs_code", "rag_code", "ragann_code", "ttc_code_1"]
col_llm = "llm_code"
col_vrai = "code"
col_libelle = "l_pr_product"
cols_tous = cols_base + [col_llm]
niveaux = (1, 2, 3, 4)

display(Markdown(f"**{len(df)} observations** — classifieurs de base : `{', '.join(cols_base)}` — LLM-judge : `{col_llm}`"))

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
#
# NB : pour les divisions vraies 98 et 99 elles-mêmes, l'accuracy "hors
# exclusion" tombe mécaniquement à 0 % — on exclut alors précisément les
# prédictions qui pourraient être correctes. Ces deux lignes ne sont donc pas
# interprétables comme une performance ; seule la ligne TOTAL et les
# divisions standards 01-13 le sont.

# %%
recap_accuracy_llm = accuracy_par_division(
    df, col_pred=col_llm, col_vrai=col_vrai, niveau=4, codes_exclus=("98", "99"), verbose=False,
)
joli(recap_accuracy_llm)

# %% [markdown]
# ## 1ter. Précision du LLM-judge par division PRÉDITE
#
# Vue complémentaire de la section précédente : au lieu de partir de la
# division vraie (rappel), on part ici de la division que le LLM a prédite,
# et on regarde quelle part de ces prédictions est effectivement correcte
# (précision). Répond à la question « quand le LLM annonce telle division,
# a-t-il raison ? ». Le regroupement par ligne (division annoncée par le LLM)
# est le même dans les deux tableaux ci-dessous ; seul le niveau auquel on
# juge une prédiction "correcte" change.
#
# - **Niveau 1** : la division vraie correspond à la division prédite.
# - **Niveau 4** : le code est exactement correct à la granularité maximale.
#   Permet de chiffrer, pour chaque division annoncée par le LLM, le volume
#   `n_erreurs` réellement à reprendre — utile pour prioriser une relecture
#   manuelle par division prédite (le plus d'erreurs captées pour le moins
#   d'effectifs à revoir).

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
#
# Même lecture que la section 1bis (accuracy par division COICOP vraie), mais
# indépendamment du classifieur : les 4 classifieurs de base et le LLM-judge
# sont présentés côte à côte pour comparaison directe, division par division.

# %%
sous_titre("Correction jugée au niveau 1 (division)")
recap_accuracy_tous_n1 = accuracy_multi_classifieurs(
    df, cols_tous, col_vrai, niveau=1, verbose=False,
)
display(joli(recap_accuracy_tous_n1))

sous_titre("Correction jugée au niveau 4 (granularité max)")
recap_accuracy_tous_n4 = accuracy_multi_classifieurs(
    df, cols_tous, col_vrai, niveau=4, verbose=False,
)
display(joli(recap_accuracy_tous_n4))

# %% [markdown]
# ## 2. Rapport sur les accords unanimes des 4 classifieurs
#
# Pour chaque niveau : accord unanime des classifieurs de base, faux positifs
# associés, et dissocie les cas où le LLM-judge suit ou non le consensus.
# Un tableau avec tous les cas est donné pour le niveau 4.
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

# %%
cols_fp = [c for c in [col_libelle, "code", "vrai_tronq", "code_consensus",
                       *[f"{c}_tronq" for c in cols_base]] if c in df_fp_4.columns]
titre(f"Lignes concernées — {len(df_fp_4)} faux positifs unanimes")
display(HTML(
    f'<div style="max-height:500px;overflow-y:auto;border:1px solid #ddd;border-radius:6px;">'
    f'{joli(df_fp_4[cols_fp]).to_html(classes="table table-sm table-striped", index=False)}'
    f'</div>'
))

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
# Parmi les 4 classifieurs de base, cas où 3 votent le même code et le 4e
# diverge : répartition du classifieur dissident, et fréquence à laquelle
# la majorité (les 3 d'accord) a raison plutôt que le dissident.
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

# %%
df_31_only_4 = df_31_4[df_31_4["cas_3_1"]]
cols_31 = [c for c in [col_libelle, "code", "vrai_tronq", "classifieur_dissident",
                       "code_majorite", "code_minoritaire",
                       "majorite_correcte", "minorite_correcte",
                       *[f"{c}_tronq" for c in cols_base]] if c in df_31_only_4.columns]
titre(f"Lignes concernées — {len(df_31_only_4)} cas")
display(HTML(
    f'<div style="max-height:500px;overflow-y:auto;border:1px solid #ddd;border-radius:6px;">'
    f'{joli(df_31_only_4[cols_31]).to_html(classes="table table-sm table-striped", index=False)}'
    f'</div>'
))

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
# Pour chaque niveau, cas où exactement un classifieur (parmi les 4 de base
# et le LLM) donne le bon code alors que tous les autres se trompent.
# Un tableau avec tous les cas est donné pour le niveau 4.
# ::: {.panel-tabset}


# %%
def afficher_seul_niveau(n):
    df_seul = stats_classifieur_seul_correct(df, cols_tous, col_vrai, niveau=n, verbose=False)
    display(joli(df_seul.attrs["resume"]))
    titre("Répartition par classifieur sauveur")
    display(joli(df_seul.attrs["repart"]))
    return df_seul


def afficher_detail_classifieur(df_seul_only, classifieur):
    """Lignes où `classifieur` est seul correct, avec la division (niveau 1) du vrai code."""
    sub = df_seul_only[df_seul_only["classifieur_seul"] == classifieur].copy()
    sub["division"] = sub["vrai_tronq"].str[:2]
    cols = [c for c in [col_libelle, "code", "division", "vrai_tronq"] if c in sub.columns]
    titre(f"{LIBELLES_CLASSIFIEURS[classifieur]} — {len(sub)} cas")
    display(HTML(
        f'<div style="max-height:400px;overflow-y:auto;border:1px solid #ddd;border-radius:6px;">'
        f'{joli(sub[cols]).to_html(classes="table table-sm table-striped", index=False)}'
        f'</div>'
    ))


# %% [markdown]
# ### Niveau 4 — sous-classe

# %%
df_seul_4 = afficher_seul_niveau(4)

# %%
df_seul_only_4 = df_seul_4[df_seul_4["seul_correct"]]
cols_seul = [c for c in [col_libelle, "code", "vrai_tronq", "classifieur_seul",
                         *[f"{c}_tronq" for c in cols_tous]] if c in df_seul_only_4.columns]
titre(f"Lignes concernées — {len(df_seul_only_4)} cas")
display(HTML(
    f'<div style="max-height:500px;overflow-y:auto;border:1px solid #ddd;border-radius:6px;">'
    f'{joli(df_seul_only_4[cols_seul]).to_html(classes="table table-sm table-striped", index=False)}'
    f'</div>'
))

# %% [markdown]
# **Détail par classifieur sauveur** (niveau 4), avec la division COICOP (niveau 1) du vrai code :
#
# ::: {.panel-tabset}

# %% [markdown]
# #### LCS

# %%
afficher_detail_classifieur(df_seul_only_4, "lcs_code")

# %% [markdown]
# #### RAG

# %%
afficher_detail_classifieur(df_seul_only_4, "rag_code")

# %% [markdown]
# #### RAG-ANN

# %%
afficher_detail_classifieur(df_seul_only_4, "ragann_code")

# %% [markdown]
# #### TTC

# %%
afficher_detail_classifieur(df_seul_only_4, "ttc_code_1")

# %% [markdown]
# #### LLM-judge

# %%
afficher_detail_classifieur(df_seul_only_4, "llm_code")

# %% [markdown]
# :::

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
#
# Pour chaque classifieur, dans quelles divisions COICOP se concentrent ses
# "sauvetages" (cas où lui seul trouve le bon code) ?

# %%
cross = stats_seul_par_division(df_seul_4, cols_tous, niveau_analyse=4, verbose=False)
titre("Effectifs (divisions × classifieur sauveur)")
display(joli(cross))
titre("Part de chaque division dans les sauvetages de chaque classifieur (%)")
display(joli(cross.attrs["parts_col"]))
titre("Classifieur le plus souvent sauveur, par division (%)")
display(joli(cross.attrs["parts_lig"]))

# %%
for c in cols_tous:
    detail_c = cross.attrs["detail_par_classifieur"][c]
    titre(f"{LIBELLES_CLASSIFIEURS[c]} — {len(detail_c)} division(s) concernée(s)")
    if len(detail_c):
        display(joli(detail_c))
