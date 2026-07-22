#Notebook de recherche qui étudie les 35 erreurs de la RF

#%%
import sys, os
sys.path.append(os.path.abspath(".."))

import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from data.load_data import charger_donnees, CHEMIN_S3_MODELISATION
from src.coicop import tronquer_niveau
from src.baseline import baseline_majorite_ttc, preparer_donnees, entrainer_evaluer

df = charger_donnees(CHEMIN_S3_MODELISATION)

cols_base = ["lcs_code", "rag_code", "ragann_code", "ttc_code_1"]
col_ttc = "ttc_code_1"
col_vrai = "code"

#%%
# Reproduit le pipeline RF de modelisation.qmd (meme random_state=42 par defaut
# dans entrainer_evaluer) => memes 35 erreurs manquees que sur le site.
y_pred_baseline = baseline_majorite_ttc(df, cols_base, col_ttc, niveau=4)
X, y = preparer_donnees(df, y_pred_baseline, col_vrai, niveau=4)
res = entrainer_evaluer(X, y)

y_test = res["y_test"]
y_pred_rf = pd.Series(res["y_pred"], index=y_test.index)
y_proba_rf = pd.Series(res["y_proba"], index=y_test.index)

manquees_idx = y_test[(y_test == 0) & (y_pred_rf == 1)].index
print(f"{len(manquees_idx)} erreurs manquees")

#%%
# Tableau complet des erreurs manquees, AVEC le libelle produit brut : ce notebook
# est un notebook de recherche prive, pas publie sur le site (contrairement a
# modelisation.qmd qui anonymise et n'expose que "ligne").
table_manquees = pd.DataFrame({
    "libelle": df.loc[manquees_idx, "raw_product"],
    "enseigne": df.loc[manquees_idx, "shop_type_name"],
    "code_vrai": df.loc[manquees_idx, col_vrai].map(lambda x: tronquer_niveau(x, niveau=4)),
    "code_baseline": y_pred_baseline.loc[manquees_idx],
    "ttc_code_1": df.loc[manquees_idx, "ttc_code_1"].map(lambda x: tronquer_niveau(x, niveau=4)),
    "ttc_conf_1": df.loc[manquees_idx, "ttc_conf_1"],
    "proba_rf": y_proba_rf.loc[manquees_idx],
}).sort_values("proba_rf", ascending=False)

table_manquees

#%%
# Reannotation manuelle des 35 erreurs manquees (Mathis). "douteux" = la baseline a
# faux mais la reponse est litigieuse (presque acceptable) ; "probleme_troncature" =
# la baseline a en realite bon, mais la comparaison brute au niveau 4 ne le detecte
# pas a cause d'un souci de troncature du code.
annotation_mathis = pd.DataFrame(
    [
        (5505, "05.5.2.1", False, False),
        (5534, "01.1.2.3", True, False),
        (2262, "04.3.1.1", False, False),
        (5336, "01.1.2.3", True, False),
        (3990, "05.2.1.2", False, False),
        (5360, "01.1.3.3", False, False),
        (3086, "09.2.2.2", False, False),
        (525, "07.4.1", True, False),
        (4565, "05.3.2.9", False, False),
        (973, "11.1.1", False, False),
        (592, "01.1.3.4", False, False),
        (2307, "01.1.3.4", False, False),
        (4965, "01", False, False),
        (4718, "98.1", True, False),
        (3496, "01.2.9", False, True),
        (2740, "98.1", False, False),
        (5117, "01.1.9.1", True, False),
        (755, "01.2.5", False, True),
        (1836, "98.1", False, False),
        (5157, "01", False, False),
        (3147, "01.1.7.9", False, False),
        (1266, "99.1", False, False),
        (1821, "99.1", True, False),
        (880, "01.1.9.1", False, False),
        (2543, "03.1.2.2", False, False),
        (465, "13.1.2", False, True),
        (2918, "01.1.9.1", True, False),
        (87, "01.1.1.3", False, False),
        (143, "09.3.2.2", False, False),
        (242, "01.1.3", False, False),
        (566, "01.1.7.9", False, False),
        (3065, "01.2.9", False, True),
        (3898, "01.1.2.2", False, False),
        (3326, "99.1", False, False),
        (2956, "06.4.2", False, True),
    ],
    columns=["ligne", "code_mathis", "ambigu", "probleme_troncature"],
).set_index("ligne")

# Recolle le libelle/les codes du pipeline pour retrouver le contexte de chaque cas
table_annotee = table_manquees.join(annotation_mathis)
table_annotee

#%%
# Categorie de verite terrain par ligne : les cas mis de cote par Mathis (ambigu /
# probleme de troncature) restent a part. Parmi le reste, code_mathis == code_baseline
# signifie que la relecture manuelle donne raison a la baseline : ce n'est pas une
# erreur de la baseline, mais un probleme d'etiquette d'origine (code_vrai faux dans
# le jeu de donnees) - distinct d'un probleme de troncature. Seul code_mathis !=
# code_baseline est une erreur de baseline confirmee.
def categoriser(row):
    if row["ambigu"]:
        return "Ambigu"
    if row["probleme_troncature"]:
        return "Probleme de troncature"
    if row["code_mathis"] == row["code_baseline"]:
        return "Erreur d'annotation"
    return "Erreur reelle"


table_annotee["categorie"] = table_annotee.apply(categoriser, axis=1)

#%%
# Repartition des 35 cas par tranche de proba_rf (confiance de la RF que la
# baseline est correcte, en tranches de 0.1) x categorie
bornes = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
table_annotee["tranche_confiance"] = pd.cut(
    table_annotee["proba_rf"], bins=bornes, include_lowest=True,
)

ordre_categories = ["Erreur reelle", "Ambigu", "Probleme de troncature", "Erreur d'annotation"]
repartition = (
    table_annotee.groupby(["tranche_confiance", "categorie"], observed=True)
    .size()
    .unstack(fill_value=0)
    .reindex(columns=ordre_categories, fill_value=0)
)
repartition

#%%
couleurs = {
    "Erreur reelle": "#2a78d6",
    "Ambigu": "#eb6834",
    "Probleme de troncature": "#1baf7a",
    "Erreur d'annotation": "#eda100",
}

fig, ax = plt.subplots(figsize=(7, 5))
bas = pd.Series(0, index=repartition.index)
for cat in ordre_categories:
    ax.bar(
        repartition.index.astype(str), repartition[cat], bottom=bas,
        label=cat, color=couleurs[cat], width=0.6,
        edgecolor="white", linewidth=2,
    )
    bas = bas + repartition[cat]

for i, total in enumerate(bas):
    ax.text(i, total + 0.3, str(int(total)), ha="center", color="#52514e")

ax.set_xlabel("Tranche de proba_rf (confiance de la RF en la baseline)")
ax.set_ylabel("Nombre de cas")
ax.set_title("Erreurs manquees par la RF, selon la confiance et la nature du cas")
ax.legend()
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.show()

#%%
# Meme graphique, en interactif (Plotly) : au survol de chaque segment, popup
# listant les cas qui le composent (ligne, code_vrai, code_baseline).
def formatter_cas(sous_table):
    lignes = [
        f"{ligne} : {row['code_vrai']} (baseline: {row['code_baseline']})"
        for ligne, row in sous_table.iterrows()
    ]
    return "<br>".join(lignes) if lignes else "aucun cas"


hover_par_cellule = table_annotee.groupby(
    ["tranche_confiance", "categorie"], observed=True
).apply(formatter_cas)

fig_interactif = go.Figure()
for cat in ordre_categories:
    hovertext = [
        hover_par_cellule.get((tranche, cat), "aucun cas") for tranche in repartition.index
    ]
    fig_interactif.add_bar(
        x=repartition.index.astype(str), y=repartition[cat],
        name=cat, marker_color=couleurs[cat],
        hovertext=hovertext, hoverinfo="text",
    )

fig_interactif.update_layout(
    barmode="stack",
    title="Erreurs manquees par la RF, selon la confiance et la nature du cas",
    xaxis_title="Tranche de proba_rf (confiance de la RF en la baseline)",
    yaxis_title="Nombre de cas",
    legend_title_text="Categorie",
)
fig_interactif.show()
# %%
