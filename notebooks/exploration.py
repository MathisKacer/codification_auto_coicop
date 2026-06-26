"""
Exploration interactive du dataset de codification automatique COICOP.

Ce script ne fait qu'appeler les fonctions définies dans src/ :
toute la logique réutilisable vit dans preprocessing.py et evaluation.py.
"""
# %%
from data.load_data import charger_base
from src.preprocessing import filtrer_lignes_valides, preparer_features, construire_X_y
from src.evaluation import (
    evaluer_llm_as_judge,
    diagnostiquer_categories_rares,
    calculer_classifieurs_corrects,
)

# %%
# Chargement des données
df = charger_base()

print("Dimensions :", df.shape)
print("\nColonnes :", df.columns.tolist())
print("\nTypes :\n", df.dtypes)
print("\nValeurs manquantes par colonne :\n", df.isna().sum())
print("\nDistribution de la categorie vraie :\n", df["code"].value_counts())
print("\nApercu :\n", df.head())

# %%
# Filtrage des lignes valides + accuracy de référence du LLM-as-judge
df_valide = filtrer_lignes_valides(df)
accuracy_llm = evaluer_llm_as_judge(df_valide)  # accuracy de 0.725

# %%
# Préparation des features (gestion des NA, recodage)
df_valide = preparer_features(df_valide)
X, y = construire_X_y(df_valide)

print(X.shape, y.shape)
print(X.isna().sum())

# %%
# Diagnostic des catégories rares
compte_par_categorie = diagnostiquer_categories_rares(y)

# %%
# Détermination des classifieurs ayant trouvé la bonne réponse
df_valide = calculer_classifieurs_corrects(df_valide)

# %%
