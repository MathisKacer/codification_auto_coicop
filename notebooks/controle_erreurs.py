#Notebook de recherche qui étudie les 35 erreurs de la RF

#%%
sys.path.append(os.path.abspath(".."))

import pandas as pd
from data.load_data import charger_donnees, CHEMIN_S3_MODELISATION
from src.coicop import tronquer_niveau
from src.baseline import baseline_majorite_ttc, preparer_donnees, entrainer_evaluer

df = charger_donnees(CHEMIN_S3_MODELISATION)

cols_base = ["lcs_code", "rag_code", "ragann_code", "ttc_code_1"]
col_ttc = "ttc_code_1"
col_vrai = "code"

#%%
# df.loc[5505] renvoie une Series (la ligne "couchée", peu lisible).
# Passer une liste a .loc renvoie un DataFrame a une seule ligne, affiche comme un tableau normal.
df.loc[[5505]]

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
        (3065, "01.2.9", True, False),
        (3898, "01.1.2.2", False, False),
        (3326, "99.1", False, False),
        (2956, "06.4.2", False, True),
    ],
    columns=["ligne", "code_mathis", "douteux", "probleme_troncature"],
).set_index("ligne")

# Recolle le libelle/les codes du pipeline pour retrouver le contexte de chaque cas
table_annotee = table_manquees.join(annotation_mathis)
table_annotee
# %%
