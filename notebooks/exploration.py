"""
Exploration interactive du dataset de codification automatique COICOP.
"""
# %%
import sys
sys.path.append("/home/onyxia/work/codification_auto_coicop")

from data.load_data import charger_base
from src.preprocessing import filtrer_lignes_valides, preparer_features, construire_X_y
from src.evaluation import (
    evaluer_llm_as_judge,
    diagnostiquer_categories_rares,
    calculer_classifieurs_corrects,
)
from src.cible_gagnant import ajouter_gagnant
from src.modeling_tree import split_train_test, construire_pipeline_arbre, entrainer_arbre, visualiser_arbre_supertree
from src.evaluation_arbre import evaluer_prediction_gagnant, evaluer_code_final
from src.baseline import evaluer_baseline_vote
from src.modeling_rf import construire_pipeline_rf, entrainer_rf, afficher_importance_features

# %%
# Chargement des donnees
df = charger_base()
print("Dimensions :", df.shape)
print("\nColonnes :", df.columns.tolist())
print("\nTypes :\n", df.dtypes)
print("\nValeurs manquantes par colonne :\n", df.isna().sum())
print("\nDistribution de la categorie vraie :\n", df["code"].value_counts())
print("\nApercu :\n", df.head())

# %%
# Filtrage des lignes valides + accuracy de reference du LLM-as-judge
df_valide = filtrer_lignes_valides(df)
accuracy_llm = evaluer_llm_as_judge(df_valide)

# %%
# Preparation des features (gestion des NA, recodage)
df_valide = preparer_features(df_valide)

# %%
# Diagnostic des categories rares (sur la verite terrain brute)
compte_par_categorie = diagnostiquer_categories_rares(df_valide["code"])

# %%
# Determination des classifieurs ayant trouve la bonne reponse (cree les *_raison)
df_valide = calculer_classifieurs_corrects(df_valide)

# %%
# Ajout de la cible "gagnant" (depend des *_raison)
df_valide = ajouter_gagnant(df_valide)

# %%
# Construction de X et y -- une seule fois, apres que toutes les colonnes necessaires existent
X, y = construire_X_y(df_valide)
print(X.shape, y.shape)
print(X.isna().sum())

# %%
# Split stratifie sur gagnant
X_train, X_test, y_train, y_test = split_train_test(X, y)

# %%
# Construction et entrainement de l'arbre
pipeline_arbre = construire_pipeline_arbre(max_depth=5, min_samples_leaf=20)
pipeline_arbre = entrainer_arbre(pipeline_arbre, X_train, y_train)

# %%
visualiser_arbre_supertree(pipeline_arbre, X_train, y_train)

# %%
# Evaluation niveau 1 : prediction du gagnant lui-meme
y_pred_gagnant = evaluer_prediction_gagnant(pipeline_arbre, X_test, y_test)

# %%
# Evaluation niveau 2 : code final reconstitue + comparaison au LLM sur le meme jeu de test
accuracy_arbre, accuracy_llm = evaluer_code_final(y_pred_gagnant, X_test, df_valide, niveau_max=4)

# %%
accuracy_baseline = evaluer_baseline_vote(X_test, df_valide, niveau_max=4)

# %%
# Construction et entrainement du Random Forest (memes X_train, y_train que l'arbre)
pipeline_rf = construire_pipeline_rf(n_estimators=300, max_depth=None, min_samples_leaf=5)
pipeline_rf = entrainer_rf(pipeline_rf, X_train, y_train)

# %%
# Evaluation (memes fonctions que pour l'arbre)
y_pred_gagnant_rf = evaluer_prediction_gagnant(pipeline_rf, X_test, y_test)
accuracy_rf, _ = evaluer_code_final(y_pred_gagnant_rf, X_test, df_valide, niveau_max=4)

# %%
# Importance des features (utile pour interpretation, en bonus du visuel d'arbre)
df_importance = afficher_importance_features(pipeline_rf, top_n=20)
# %%
