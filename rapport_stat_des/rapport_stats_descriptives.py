# quarto render rapport_stat_des/rapport_stats_descriptives.py --to html --output-dir ../outputs
#
# Pour rejouer ce rapport sur un autre run (nouveau run de l'enquête, sans
# toucher au run par défaut utilisé par le site) :
#
#   CHEMIN_S3_STATS_DESCRIPTIVES="s3://projet-budget-famille/data/workflow_runs/.../predictions.parquet" \
#     quarto render rapport_stat_des/rapport_stats_descriptives.py --to html --output-dir ../outputs
#
# Aucune modification de fichier n'est nécessaire : c'est le seul levier
# prévu pour changer de run, ce qui rend ce notebook directement rejouable
# comme étape automatisée d'un pipeline (Argo, cron...) le jour où ce
# rapport y sera intégré.

# %% [markdown]
# ---
# title: "Stats descriptives — codification COICOP"
# subtitle: "Fiabilité des classifieurs de codification COICOP sur un run donné"
# date: today
# format:
#   html:
#     toc: true
#     toc-depth: 3
#     embed-resources: true
#     theme: flatly
#     df-print: default
# jupyter: python3
# execute:
#   echo: false
#   warning: false
# ---
#
# Ce rapport dresse un état des lieux de la fiabilité des classifieurs de
# codification COICOP sur un run de l'enquête Budget de Famille : accord
# entre classifieurs, effet du prix et du mode de collecte, performance du
# classifieur TTC, qualité des codes produits, et calibration des scores de
# confiance.

# %%
import os

import pandas as pd
from IPython.display import display, Markdown, HTML

from load_data import charger_donnees, CHEMIN_S3_STATS_DESCRIPTIVES
from coicop import libelle_division, tronquer_niveau, niveau_atteint
from stats_accord import (
    stats_accord,
    analyse_faux_positifs,
    stats_accord_avec_llm,
    stats_classifieur_seul_correct,
    stats_majorite_3_1,
    recap_multi_niveaux,
    stats_seul_par_division,
    accuracy_par_division,
    accuracy_multi_classifieurs,
    accuracy_multi_classifieurs_par_groupe,
    precision_par_division_llm,
)
from rapport_utils import LIBELLES_CLASSIFIEURS, joli, titre, sous_titre

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

# %%
CHEMIN_S3 = os.environ.get("CHEMIN_S3_STATS_DESCRIPTIVES", CHEMIN_S3_STATS_DESCRIPTIVES)
df = charger_donnees(CHEMIN_S3)
df["ligne"] = df.index  # identifiant de ligne pour retrouver un cas sans publier le libellé brut

cols_base = ["lcs_code", "rag_code", "ragann_code", "ttc_code_1"]
col_llm = "llm_code"
col_vrai = "code"
col_ligne = "ligne"
cols_tous = cols_base + [col_llm]
niveaux = (1, 2, 3, 4)

display(Markdown(
    f"**{len(df)} observations** — run : `{CHEMIN_S3}` — "
    f"classifieurs de base : `{', '.join(cols_base)}` — LLM-judge : `{col_llm}`"
))

# %% [markdown]
# # 1. Aperçu des données

# %% [markdown]
# ## Jeu de données

# %%
display(Markdown(f"**{len(df)} observations** — classifieurs de base : `{', '.join(cols_base)}` — LLM-judge : `{col_llm}`"))
display(Markdown(f"Source : `{CHEMIN_S3}`"))

# %% [markdown]
# ## Aperçu des premières lignes

# %%
df[[col_ligne, col_vrai, *cols_base, col_llm]].head().set_index(col_ligne)

# %% [markdown]
# ## Répartition par division COICOP (niveau 1, vrai code)

# %%
repartition = (
    df[col_vrai].astype(str).str[:2]
    .value_counts().sort_index()
    .rename_axis("division").reset_index(name="n")
)
repartition["%"] = (repartition["n"] / len(df) * 100).round(1)
joli(repartition)

# %% [markdown]
# # 2. Vue synthétique et accuracy multi-niveaux

# %% [markdown]
# ## Vue synthétique multi-niveaux
#
# Décomposition, pour chaque niveau de troncature COICOP, de la part des
# observations en accord correct unanime, faux positif partagé par tous les
# classifieurs (dont le LLM), faux positif rattrapé par le LLM, ou désaccord.

# %%
recap = recap_multi_niveaux(df, cols_base, col_llm, col_vrai, niveaux=niveaux, verbose=False)
joli(recap)

# %% [markdown]
# ## Accuracy par classifieur, par division (vérité terrain)
#
# Accuracy de chacun des 5 classifieurs (les 4 classifieurs de base et le
# LLM-judge) contre `code`, ventilée par division COICOP (niveau 1 du vrai
# code), pour chaque niveau de troncature de comparaison. La ligne TOTAL
# donne l'accuracy globale de chaque classifieur ; les autres lignes la
# détaillent par division. Les divisions 98 et 99 sont des codes à part
# entière, que les classifieurs doivent aussi savoir détecter, elles sont
# donc traitées ici comme n'importe quelle autre division, sans exclusion.

# %% [markdown]
# ::: {.panel-tabset}
#
# ### Niveau 4 — sous-classe

# %%
display(joli(accuracy_multi_classifieurs(
    df, cols_pred=cols_tous, col_vrai=col_vrai, niveau=4, codes_exclus=(), verbose=False,
)))

# %% [markdown]
# ### Niveau 3 — classe

# %%
display(joli(accuracy_multi_classifieurs(
    df, cols_pred=cols_tous, col_vrai=col_vrai, niveau=3, codes_exclus=(), verbose=False,
)))

# %% [markdown]
# ### Niveau 2 — groupe

# %%
display(joli(accuracy_multi_classifieurs(
    df, cols_pred=cols_tous, col_vrai=col_vrai, niveau=2, codes_exclus=(), verbose=False,
)))

# %% [markdown]
# ### Niveau 1 — division

# %%
display(joli(accuracy_multi_classifieurs(
    df, cols_pred=cols_tous, col_vrai=col_vrai, niveau=1, codes_exclus=(), verbose=False,
)))

# %% [markdown]
# :::

# %% [markdown]
# ## Accuracy du LLM-judge par division (vérité terrain)
#
# Même lecture, mais focalisée sur le LLM-judge seul (indépendamment des
# autres classifieurs) : accuracy de `llm_code` contre `code`, ventilée par
# division COICOP vraie (rappel), pour chaque niveau de troncature.


# %%
def accuracy_llm_par_division(niveau):
    recap_niveau = accuracy_par_division(
        df, col_pred=col_llm, col_vrai=col_vrai, niveau=niveau, codes_exclus=(), verbose=False,
    )[["n", "n_correct", "accuracy"]].copy()
    recap_niveau["n_erreurs"] = recap_niveau["n"] - recap_niveau["n_correct"]
    return recap_niveau[["n", "n_correct", "n_erreurs", "accuracy"]]


# %% [markdown]
# ::: {.panel-tabset}
#
# ### Niveau 4 — sous-classe

# %%
display(joli(accuracy_llm_par_division(4)))

# %% [markdown]
# ### Niveau 3 — classe

# %%
display(joli(accuracy_llm_par_division(3)))

# %% [markdown]
# ### Niveau 2 — groupe

# %%
display(joli(accuracy_llm_par_division(2)))

# %% [markdown]
# ### Niveau 1 — division

# %%
display(joli(accuracy_llm_par_division(1)))

# %% [markdown]
# :::

# %% [markdown]
# ## Précision du LLM-judge par division prédite
#
# Vue complémentaire de la section précédente : au lieu de partir de la
# division vraie (rappel), on part ici de la division que le LLM a prédite,
# et on regarde quelle part de ces prédictions est effectivement correcte
# (précision).

# %% [markdown]
# ::: {.panel-tabset}
#
# ### Niveau 4 — sous-classe

# %%
display(joli(precision_par_division_llm(
    df, col_pred=col_llm, col_vrai=col_vrai, niveau=4, verbose=False,
)))

# %% [markdown]
# ### Niveau 3 — classe

# %%
display(joli(precision_par_division_llm(
    df, col_pred=col_llm, col_vrai=col_vrai, niveau=3, verbose=False,
)))

# %% [markdown]
# ### Niveau 2 — groupe

# %%
display(joli(precision_par_division_llm(
    df, col_pred=col_llm, col_vrai=col_vrai, niveau=2, verbose=False,
)))

# %% [markdown]
# ### Niveau 1 — division

# %%
display(joli(precision_par_division_llm(
    df, col_pred=col_llm, col_vrai=col_vrai, niveau=1, verbose=False,
)))

# %% [markdown]
# :::

# %% [markdown]
# ## Confusion par division COICOP
#
# Matrice de confusion (division vraie × division prédite par le LLM) : une
# ligne redonne l'accuracy par division vraie, une colonne redonne la
# précision par division prédite. La diagonale concentre les accords ; les
# cellules hors diagonale montrent où se concentrent les confusions.

# %%
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

vrai_div = df[col_vrai].astype(str).str[:2]
pred_div = df[col_llm].map(lambda x: x[:2] if pd.notna(x) else "N/A")

categories = sorted(set(vrai_div) | set(pred_div) - {"N/A"}) + (
    ["N/A"] if "N/A" in set(pred_div) else []
)
confusion = pd.crosstab(vrai_div, pred_div).reindex(
    index=categories, columns=categories, fill_value=0,
)
totaux = confusion.sum(axis=1)
confusion_pct = confusion.div(totaux, axis=0).fillna(0) * 100

seq_bleu = LinearSegmentedColormap.from_list("seq_bleu", [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
])

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(confusion_pct.values, cmap=seq_bleu, vmin=0, vmax=100)

ax.set_xticks(range(len(categories)))
ax.set_yticks(range(len(categories)))
ax.set_xticklabels(categories)
ax.set_yticklabels([f"{c} (n={totaux[c]})" for c in categories])
ax.set_xlabel("Division prédite (LLM)")
ax.set_ylabel("Division vraie (n = effectif de la division)")
ax.set_title("Confusion par division COICOP — % de la division vraie")

for i in range(len(categories)):
    for j in range(len(categories)):
        val = confusion_pct.values[i, j]
        if val > 0.5:
            couleur_texte = "white" if val > 55 else "#1a1a1a"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    color=couleur_texte, fontsize=7)

fig.colorbar(im, ax=ax, shrink=0.8, label="% de la division vraie")
plt.tight_layout()
plt.show()

# %% [markdown]
# # 3. Accuracy par prix et mode de collecte
#
# Deux variables disponibles indépendamment du contenu du libellé peuvent
# influencer la difficulté de codification : le **montant de la dépense**
# (`budget`) et son **mode de collecte** (`source` — ticket de caisse
# scanné via l'appli, ou saisie manuelle dans un carnet/l'appli).

# %%
LIBELLES_SOURCE = {
    "receipts_from_app": "Ticket de caisse (appli)",
    "manual_from_book": "Saisie manuelle (carnet)",
    "manual_from_app": "Saisie manuelle (appli)",
}
ORDRE_SOURCE = [*LIBELLES_SOURCE.values(), "TOTAL"]

groupe_source = df["source"].map(LIBELLES_SOURCE).fillna("INCONNU")
groupe_source.name = "source"

BORNES_BUDGET = [0, 2, 5, 10, 20, 50, 100, float("inf")]
LIBELLES_BUDGET = ["< 2 €", "2-5 €", "5-10 €", "10-20 €", "20-50 €", "50-100 €", "100 € et +"]
ORDRE_BUDGET = [*LIBELLES_BUDGET, "Budget inconnu", "TOTAL"]

tranche_budget = pd.cut(
    df["budget"], bins=BORNES_BUDGET, labels=LIBELLES_BUDGET, include_lowest=True,
)
groupe_budget = tranche_budget.astype(str).replace("nan", "Budget inconnu")
groupe_budget.name = "budget"

# %% [markdown]
# ## Accuracy par mode de collecte
#
# `source` distingue les dépenses issues d'un ticket de caisse scanné dans
# l'application (`receipts_from_app`) de celles saisies manuellement, que ce
# soit dans le carnet papier (`manual_from_book`) ou directement dans
# l'application (`manual_from_app`). Un ticket scanné donne un libellé plus
# proche du texte de caisse d'origine ; une saisie manuelle dépend de la
# reformulation du déclarant — deux régimes de difficulté potentiellement
# différents pour les classifieurs.

# %% [markdown]
# ::: {.panel-tabset}
#
# ### Niveau 4 — sous-classe

# %%
display(joli(accuracy_multi_classifieurs_par_groupe(
    df, cols_pred=cols_tous, col_vrai=col_vrai, groupe=groupe_source, niveau=4, verbose=False,
).reindex(ORDRE_SOURCE)))

# %% [markdown]
# ### Niveau 3 — classe

# %%
display(joli(accuracy_multi_classifieurs_par_groupe(
    df, cols_pred=cols_tous, col_vrai=col_vrai, groupe=groupe_source, niveau=3, verbose=False,
).reindex(ORDRE_SOURCE)))

# %% [markdown]
# ### Niveau 2 — groupe

# %%
display(joli(accuracy_multi_classifieurs_par_groupe(
    df, cols_pred=cols_tous, col_vrai=col_vrai, groupe=groupe_source, niveau=2, verbose=False,
).reindex(ORDRE_SOURCE)))

# %% [markdown]
# ### Niveau 1 — division

# %%
display(joli(accuracy_multi_classifieurs_par_groupe(
    df, cols_pred=cols_tous, col_vrai=col_vrai, groupe=groupe_source, niveau=1, verbose=False,
).reindex(ORDRE_SOURCE)))

# %% [markdown]
# :::

# %% [markdown]
# ## Accuracy par tranche de budget
#
# Montant de la dépense (`budget`), découpé en tranches fixes plutôt qu'en
# quantiles pour rester lisible en euros. Les dépenses sans montant
# renseigné sont regroupées à part ("Budget inconnu") plutôt qu'exclues.

# %%
display(Markdown(
    f"**{int(df['budget'].isna().sum())}** dépenses sans montant renseigné "
    f"sur **{len(df)}** ({df['budget'].isna().mean():.1%})."
))

# %% [markdown]
# ::: {.panel-tabset}
#
# ### Niveau 4 — sous-classe

# %%
display(joli(accuracy_multi_classifieurs_par_groupe(
    df, cols_pred=cols_tous, col_vrai=col_vrai, groupe=groupe_budget, niveau=4, verbose=False,
).reindex(ORDRE_BUDGET)))

# %% [markdown]
# ### Niveau 3 — classe

# %%
display(joli(accuracy_multi_classifieurs_par_groupe(
    df, cols_pred=cols_tous, col_vrai=col_vrai, groupe=groupe_budget, niveau=3, verbose=False,
).reindex(ORDRE_BUDGET)))

# %% [markdown]
# ### Niveau 2 — groupe

# %%
display(joli(accuracy_multi_classifieurs_par_groupe(
    df, cols_pred=cols_tous, col_vrai=col_vrai, groupe=groupe_budget, niveau=2, verbose=False,
).reindex(ORDRE_BUDGET)))

# %% [markdown]
# ### Niveau 1 — division

# %%
display(joli(accuracy_multi_classifieurs_par_groupe(
    df, cols_pred=cols_tous, col_vrai=col_vrai, groupe=groupe_budget, niveau=1, verbose=False,
).reindex(ORDRE_BUDGET)))

# %% [markdown]
# :::

# %% [markdown]
# # 4. Accords unanimes des 4 classifieurs
#
# Pour chaque niveau : accord unanime des classifieurs de base, faux positifs
# associés, et dissociation des cas où le LLM-judge suit ou non le consensus.


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
# ::: {.panel-tabset}
#
# ## Niveau 4 — sous-classe

# %%
df_stats_4, df_fp_4, df_acc_4 = afficher_detail_niveau(4)

# %%
cols_fp = [c for c in [col_ligne, "code", "vrai_tronq", "code_consensus",
                       *[f"{c}_tronq" for c in cols_base]] if c in df_fp_4.columns]
titre(f"Lignes concernées — {len(df_fp_4)} faux positifs unanimes")
display(HTML(
    f'<div style="max-height:500px;overflow-y:auto;border:1px solid #ddd;border-radius:6px;">'
    f'{joli(df_fp_4[cols_fp]).to_html(classes="table table-sm table-striped", index=False)}'
    f'</div>'
))

# %% [markdown]
# ## Niveau 3 — classe

# %%
df_stats_3, df_fp_3, df_acc_3 = afficher_detail_niveau(3)

# %% [markdown]
# ## Niveau 2 — groupe

# %%
df_stats_2, df_fp_2, df_acc_2 = afficher_detail_niveau(2)

# %% [markdown]
# ## Niveau 1 — division

# %%
df_stats_1, df_fp_1, df_acc_1 = afficher_detail_niveau(1)

# %% [markdown]
# :::

# %% [markdown]
# # 5. Cas où 3 classifieurs de base sont d'accord contre 1 dissident
#
# Parmi les 4 classifieurs de base, cas où 3 votent le même code et le 4e
# diverge : répartition du classifieur dissident, et fréquence à laquelle la
# majorité (les 3 d'accord) a raison plutôt que le dissident.


# %%
def afficher_3_1_niveau(n):
    df_31 = stats_majorite_3_1(df, cols_base, col_vrai, niveau=n, verbose=False)
    titre("Vue d'ensemble")
    display(joli(df_31.attrs["resume"]))
    titre("Répartition par classifieur dissident")
    display(joli(df_31.attrs["repart_dissident"]))
    return df_31


# %% [markdown]
# ::: {.panel-tabset}
#
# ## Niveau 4 — sous-classe

# %%
df_31_4 = afficher_3_1_niveau(4)

# %%
df_31_only_4 = df_31_4[df_31_4["cas_3_1"]]
cols_31 = [c for c in [col_ligne, "code", "vrai_tronq", "classifieur_dissident",
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
# ## Niveau 3 — classe

# %%
df_31_3 = afficher_3_1_niveau(3)

# %% [markdown]
# ## Niveau 2 — groupe

# %%
df_31_2 = afficher_3_1_niveau(2)

# %% [markdown]
# ## Niveau 1 — division

# %%
df_31_1 = afficher_3_1_niveau(1)

# %% [markdown]
# :::

# %% [markdown]
# # 6. Un seul classifieur a raison
#
# Pour chaque niveau, cas où exactement un classifieur (parmi les 4 de base
# et le LLM) donne le bon code alors que tous les autres se trompent.


# %%
def afficher_seul_niveau(n):
    df_seul = stats_classifieur_seul_correct(df, cols_tous, col_vrai, niveau=n, verbose=False)
    titre("Vue d'ensemble")
    display(joli(df_seul.attrs["resume"]))
    titre("Répartition par classifieur sauveur")
    display(joli(df_seul.attrs["repart"]))
    return df_seul


def afficher_detail_classifieur(df_seul_only, classifieur):
    """Lignes où `classifieur` est seul correct, avec la division (niveau 1) du vrai code."""
    sub = df_seul_only[df_seul_only["classifieur_seul"] == classifieur].copy()
    sub["division"] = sub["vrai_tronq"].str[:2]
    cols = [c for c in [col_ligne, "code", "division", "vrai_tronq"] if c in sub.columns]
    titre(f"{LIBELLES_CLASSIFIEURS[classifieur]} — {len(sub)} cas")
    display(HTML(
        f'<div style="max-height:400px;overflow-y:auto;border:1px solid #ddd;border-radius:6px;">'
        f'{joli(sub[cols]).to_html(classes="table table-sm table-striped", index=False)}'
        f'</div>'
    ))


# %% [markdown]
# :::: {.panel-tabset}
#
# ## Niveau 4 — sous-classe

# %%
df_seul_4 = afficher_seul_niveau(4)

# %%
df_seul_only_4 = df_seul_4[df_seul_4["seul_correct"]]
cols_seul = [c for c in [col_ligne, "code", "vrai_tronq", "classifieur_seul",
                         *[f"{c}_tronq" for c in cols_tous]] if c in df_seul_only_4.columns]
titre(f"Lignes concernées — {len(df_seul_only_4)} cas")
display(HTML(
    f'<div style="max-height:500px;overflow-y:auto;border:1px solid #ddd;border-radius:6px;">'
    f'{joli(df_seul_only_4[cols_seul]).to_html(classes="table table-sm table-striped", index=False)}'
    f'</div>'
))

# %% [markdown]
# **Détail par classifieur sauveur** (niveau 4), avec la division COICOP
# (niveau 1) du vrai code :
#
# ::: {.panel-tabset}
#
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
# ## Niveau 3 — classe

# %%
df_seul_3 = afficher_seul_niveau(3)

# %% [markdown]
# ## Niveau 2 — groupe

# %%
df_seul_2 = afficher_seul_niveau(2)

# %% [markdown]
# ## Niveau 1 — division

# %%
df_seul_1 = afficher_seul_niveau(1)

# %% [markdown]
# ::::

# %% [markdown]
# ## Ventilation par division COICOP — cas "un seul correct" (niveau 4)
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


def afficher_ventilation_classifieur(classifieur):
    detail_c = cross.attrs["detail_par_classifieur"][classifieur]
    titre(f"{LIBELLES_CLASSIFIEURS[classifieur]} — {len(detail_c)} division(s) concernée(s)")
    if len(detail_c):
        display(joli(detail_c))


# %% [markdown]
# **Détail par classifieur sauveur**, au choix :
#
# ::: {.panel-tabset}
#
# ### LCS

# %%
afficher_ventilation_classifieur("lcs_code")

# %% [markdown]
# ### RAG

# %%
afficher_ventilation_classifieur("rag_code")

# %% [markdown]
# ### RAG-ANN

# %%
afficher_ventilation_classifieur("ragann_code")

# %% [markdown]
# ### TTC

# %%
afficher_ventilation_classifieur("ttc_code_1")

# %% [markdown]
# ### LLM-judge

# %%
afficher_ventilation_classifieur("llm_code")

# %% [markdown]
# :::

# %% [markdown]
# # 7. Le top-3 de TTC
#
# TTC (TorchTextClassifier) ne fournit pas un seul code mais un classement de
# ses trois meilleures propositions (`ttc_code_1`, `ttc_code_2`,
# `ttc_code_3`), chacune avec une confiance. Les autres analyses de ce
# notebook n'utilisent que le rang 1 ; cette section regarde ce qu'apportent
# les rangs 2 et 3, et isole les cas où TTC est le seul à trouver le bon
# code. Toutes les comparaisons sont faites au niveau 4 (sous-classe).


# %%
def tr4(col):
    return df[col].map(lambda x: tronquer_niveau(x, niveau=4))


def pct(x):
    return f"{x:.1%}"


vrai = tr4(col_vrai)
ttc1 = tr4("ttc_code_1")
ttc2 = tr4("ttc_code_2")
ttc3 = tr4("ttc_code_3")

c1 = (ttc1 == vrai)
c2 = (ttc2 == vrai)
c3 = (ttc3 == vrai)
top3 = c1 | c2 | c3
n = len(df)

# %% [markdown]
# ## Justesse de chaque rang TTC
#
# Part des produits pour lesquels le code d'un rang donné correspond au vrai
# code. La dernière ligne — le vrai code présent quelque part dans le top-3 —
# est le plafond que l'on atteindrait en sachant toujours choisir le bon
# rang.

# %%
recap_ttc = pd.DataFrame({
    "Cas": [
        "Rang 1 correct",
        "Rang 2 correct",
        "Rang 3 correct",
        "Vrai code présent dans le top-3",
    ],
    "Effectif": [int(c1.sum()), int(c2.sum()), int(c3.sum()), int(top3.sum())],
    "% des produits": [pct(c1.mean()), pct(c2.mean()), pct(c3.mean()), pct(top3.mean())],
})
recap_ttc

# %%
display(Markdown(
    f"Le rang 1 seul est correct dans **{pct(c1.mean())}** des cas, mais le vrai "
    f"code figure dans le top-3 dans **{pct(top3.mean())}** des cas — soit "
    f"**{(top3.mean() - c1.mean()) * 100:.1f} points** de couverture "
    f"supplémentaire dormant dans les rangs 2 et 3. "
    f"(TTC fournit toujours un rang 2 et un rang 3 : présents dans "
    f"{pct(ttc2.notna().mean())} et {pct(ttc3.notna().mean())} des cas.)"
))

# %% [markdown]
# ## Valeur ajoutée des rangs 2 et 3
#
# La question opérationnelle : quand le rang 1 se trompe, à quelle fréquence
# le vrai code est-il quand même récupérable en rang 2 ou 3 ?

# %%
faux1 = ~c1
n_faux = int(faux1.sum())
r2 = int((faux1 & c2).sum())
r3 = int((faux1 & c3).sum())
r23 = int((faux1 & (c2 | c3)).sum())

resc = pd.DataFrame({
    "Cas (parmi les produits où le rang 1 a tort)": [
        "Rang 2 rattrape",
        "Rang 3 rattrape",
        "Rang 2 ou 3 rattrape",
    ],
    "Effectif": [r2, r3, r23],
    "% des cas où rang 1 a tort": [pct(r2 / n_faux), pct(r3 / n_faux), pct(r23 / n_faux)],
    "% de tous les produits": [pct(r2 / n), pct(r3 / n), pct(r23 / n)],
})
resc

# %%
display(Markdown(
    f"Sur les **{n_faux}** produits où le rang 1 de TTC se trompe, le vrai code "
    f"est malgré tout présent en rang 2 ou 3 dans **{r23}** cas "
    f"(**{pct(r23 / n_faux)}**). C'est le gain potentiel qu'on récupérerait en "
    f"exploitant le top-3 plutôt que le seul rang 1."
))

# %% [markdown]
# **La confiance discrimine-t-elle les rangs 2/3 corrects ?** Si oui,
# `ttc_conf_2` et `ttc_conf_3` sont des variables utiles pour départager.
# Confiance médiane selon que le rang est correct ou non :

# %%
conf_tab = pd.DataFrame({
    "Rang": ["Rang 2", "Rang 3"],
    "Conf. médiane si correct": [
        round(df.loc[c2, "ttc_conf_2"].median(), 3),
        round(df.loc[c3, "ttc_conf_3"].median(), 3),
    ],
    "Conf. médiane si incorrect": [
        round(df.loc[~c2 & ttc2.notna(), "ttc_conf_2"].median(), 3),
        round(df.loc[~c3 & ttc3.notna(), "ttc_conf_3"].median(), 3),
    ],
})
conf_tab

# %% [markdown]
# ## TTC a-t-il raison seul ?
#
# Cas où TTC trouve le bon code alors qu'aucun des trois autres classifieurs
# de base (LCS, RAG, RAG-ANN) ne l'a — la contribution propre de TTC, qu'on
# perdrait en le retirant.

# %%
lcs = tr4("lcs_code")
rag = tr4("rag_code")
ragann = tr4("ragann_code")
autres_bon = (lcs == vrai) | (rag == vrai) | (ragann == vrai)

ttc1_seul = c1 & ~autres_bon
ttc_top3_seul = top3 & ~autres_bon

seul_ttc = pd.DataFrame({
    "Cas": [
        "Rang 1 correct, aucun autre classifieur correct",
        "Vrai code dans le top-3, aucun autre classifieur correct",
    ],
    "Effectif": [int(ttc1_seul.sum()), int(ttc_top3_seul.sum())],
    "% des produits": [pct(ttc1_seul.mean()), pct(ttc_top3_seul.mean())],
})
seul_ttc

# %%
display(Markdown(
    f"TTC est seul à trouver le bon code (rang 1) dans **{int(ttc1_seul.sum())}** "
    f"cas (**{pct(ttc1_seul.mean())}**). En comptant son top-3, il apporte le vrai "
    f"code qu'aucun autre classifieur n'a proposé dans **{int(ttc_top3_seul.sum())}** "
    f"cas (**{pct(ttc_top3_seul.mean())}**)."
))

# %% [markdown]
# **Détail des cas où TTC (rang 1) est seul correct**, avec la division
# COICOP (niveau 1) du vrai code.

# %%
det = pd.DataFrame({
    "Ligne": df.loc[ttc1_seul, col_ligne].values,
    "Code vrai": vrai[ttc1_seul].values,
    "Division": vrai[ttc1_seul].str[:2].map(libelle_division).values,
})
display(HTML(
    f'<div style="max-height:400px;overflow-y:auto;border:1px solid #ddd;border-radius:6px;">'
    f'{det.to_html(classes="table table-sm table-striped", index=False)}'
    f'</div>'
))

# %% [markdown]
# # 8. Profondeur et qualité des codes
#
# Un code COICOP est hiérarchique (convention du projet : niveau N = N+1
# chiffres significatifs, format `XX.X.X.X` au niveau 4). Les analyses de ce
# notebook comparent toujours les classifieurs à la vérité terrain **au
# niveau 4** (`tronquer_niveau(..., niveau=4)`), qui ne fait que raccourcir un
# code, jamais le rallonger. Un classifieur peut donc être pénalisé de deux
# façons différentes d'une simple erreur de contenu :
#
# - il répond avec **moins** de segments que le niveau 4 (code incomplet) —
#   la comparaison stricte le compte comme faux même s'il n'a rien affirmé
#   d'incorrect sur les segments fournis ;
# - il répond avec **plus** de segments (sur-précis) — sans incidence sur la
#   comparaison (le surplus est tronqué), mais ça peut signaler un
#   changement de comportement du classifieur.


# %%
cols_codes = [col_vrai, *cols_base, "ttc_code_2", "ttc_code_3", col_llm]
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
            "n": n_total,
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


recap_profondeur = stats_profondeur(df, cols_codes)

# %% [markdown]
# ## Profondeur réelle des codes, par classifieur
#
# Répartition de la profondeur effective de chaque code (avant toute
# troncature), en part du total des observations. `code` est la vérité
# terrain ; les autres colonnes sont les classifieurs (`ttc_code_2` et
# `ttc_code_3` sont les rangs 2 et 3 de TTC, cf. section 6).

# %%
joli(recap_profondeur.round(4))

# %% [markdown]
# ## Codes incomplets, complets ou sur-précis par rapport au niveau 4

# %%
recap_synthese = pd.DataFrame({
    "pct_absent": recap_profondeur["pct_absent"],
    "pct_incomplet": recap_profondeur[["pct_profondeur_1", "pct_profondeur_2", "pct_profondeur_3"]].sum(axis=1),
    "pct_complet_niveau_4": recap_profondeur["pct_profondeur_4"],
    "pct_sur_precis": recap_profondeur["pct_profondeur_5_plus"],
})
joli(recap_synthese.round(4))

# %%
display(Markdown(
    "Un code **incomplet** (moins de 4 segments) n'est pas forcément une "
    "erreur de fond : si le 4e segment de la vérité vaut `0` (convention "
    "COICOP \"pas de sous-classe plus précise\"), le classifieur n'a rien "
    "affirmé d'incorrect en s'arrêtant avant. `tronquer_niveau()` "
    "(`src/coicop.py`), utilisée partout dans ce notebook, retire ces \"0\" "
    "terminaux en cascade pour ne pas pénaliser ce cas."
))

# %% [markdown]
# ## Codes absents et indicateurs de codabilité
#
# Au-delà de la profondeur, un classifieur peut ne renvoyer aucun code
# (`NaN`), ou exposer son propre indicateur booléen de codabilité.

# %%
titre("Codes absents (NaN), par classifieur")
joli(recap_profondeur[["pct_absent"]].round(4))

# %%
LIBELLES_CODABLE = {"codable": "Codable (indicateur générique)", "ragann_codable": "RAG-ANN"}

cols_codable = [c for c in ["codable", "ragann_codable"] if c in df.columns]
if cols_codable:
    recap_codable = pd.DataFrame({
        "pct_non_codable": [(~df[c]).sum() / len(df) for c in cols_codable],
    }, index=[LIBELLES_CODABLE[c] for c in cols_codable])
    recap_codable.index.name = "classifieur"
    titre("Indicateurs de codabilité dédiés")
    display(joli(recap_codable.round(4)))

# %% [markdown]
# ## Divisions spéciales (98 / 99) par classifieur
#
# Rappel : la division `98` signale un produit indéterminé/illisible, `99`
# un cas hors COICOP (dons, impôts, opérations bancaires...). Part des
# prédictions de chaque classifieur tombant dans l'une de ces deux
# divisions, comparée à la vérité terrain.

# %%
def pct_divisions_speciales(df, colonnes):
    lignes = []
    for c in colonnes:
        div = df[c].astype(str).str[:2]
        lignes.append({
            "classifieur": c,
            "pct_div_98": (div == "98").sum() / len(df),
            "pct_div_99": (div == "99").sum() / len(df),
        })
    out = pd.DataFrame(lignes).set_index("classifieur")
    out.index = out.index.map(lambda c: LIBELLES_CLASSIFIEURS.get(c, c))
    out.index.name = "classifieur"
    return out


joli(pct_divisions_speciales(df, cols_codes).round(4))

# %% [markdown]
# # 9. Calibration des scores de confiance
#
# Chaque classifieur expose son propre score de confiance (`lcs_distance`,
# `rag_confidence`, `ragann_confidence`, `ttc_conf_1`, `llm_confiance`).
# Cette section vérifie s'il est bien calibré : est-ce qu'une confiance plus
# élevée correspond réellement à une part plus grande de bonnes réponses ?
# C'est la question centrale avant d'utiliser un seuil de confiance pour
# décider quand faire confiance à un classifieur et quand escalader (reprise
# manuelle ou LLM-judge).


# %%
import plotly.graph_objects as go


def stats_calibration(df, col_confiance, col_pred, col_vrai, bornes, niveau=4):
    """Part de predictions correctes (niveau 4), par tranche de confiance."""
    vrai_tronq = df[col_vrai].map(lambda x: tronquer_niveau(x, niveau=niveau))
    pred_tronq = df[col_pred].map(lambda x: tronquer_niveau(x, niveau=niveau))
    correct = pred_tronq == vrai_tronq
    tranche = pd.cut(df[col_confiance], bins=bornes, include_lowest=True)
    recap_calib = pd.DataFrame({"correct": correct, "tranche": tranche}).groupby(
        "tranche", observed=True
    ).agg(n=("correct", "count"), pct_correct_confiance=("correct", "mean"))
    return recap_calib


def graphique_calibration(recap_calib, titre_graphique):
    fig = go.Figure()
    fig.add_bar(
        x=[str(i) for i in recap_calib.index], y=recap_calib["pct_correct_confiance"],
        text=[f"n={n}" for n in recap_calib["n"]], textposition="outside",
        marker_color="#2a78d6",
    )
    fig.update_layout(
        title=titre_graphique,
        yaxis_title="% correct (niveau 4)", yaxis_tickformat=".0%", yaxis_range=[0, 1.05],
        xaxis_title="Tranche de confiance",
    )
    fig.show()


# %% [markdown]
# ::: {.panel-tabset}
#
# ## LCS
#
# `lcs_distance` est une distance : **plus petit = plus proche, donc plus
# confiant**. Les tranches sont donc lues dans l'ordre inverse des autres
# classifieurs.

# %%
recap_lcs = stats_calibration(
    df, "lcs_distance", "lcs_code", col_vrai,
    bornes=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
)
display(joli(recap_lcs.round(3)))
graphique_calibration(recap_lcs, "LCS : % correct par tranche de distance (plus petit = plus proche)")

# %% [markdown]
# ## RAG

# %%
recap_rag = stats_calibration(
    df, "rag_confidence", "rag_code", col_vrai,
    bornes=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
)
display(joli(recap_rag.round(3)))
graphique_calibration(recap_rag, "RAG : % correct par tranche de confiance")

# %% [markdown]
# ## RAG-ANN

# %%
recap_ragann = stats_calibration(
    df, "ragann_confidence", "ragann_code", col_vrai,
    bornes=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
)
display(joli(recap_ragann.round(3)))
graphique_calibration(recap_ragann, "RAG-ANN : % correct par tranche de confiance")

# %% [markdown]
# ## TTC
#
# Rang 1 uniquement — voir la section 6 pour les rangs 2 et 3.

# %%
recap_ttc_calib = stats_calibration(
    df, "ttc_conf_1", "ttc_code_1", col_vrai,
    bornes=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
)
display(joli(recap_ttc_calib.round(3)))
graphique_calibration(recap_ttc_calib, "TTC (rang 1) : % correct par tranche de confiance")

# %% [markdown]
# ## LLM-judge
#
# `llm_confiance` est une échelle discrète 1 à 5 (pas une probabilité
# continue).

# %%
recap_llm_calib = stats_calibration(
    df, "llm_confiance", col_llm, col_vrai,
    bornes=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
)
display(joli(recap_llm_calib.round(3)))
graphique_calibration(recap_llm_calib, "LLM-judge : % correct par tranche de confiance (échelle 1-5)")

# %% [markdown]
# :::
