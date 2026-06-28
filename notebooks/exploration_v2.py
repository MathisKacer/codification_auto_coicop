"""
Exploration et modelisation du meta-classifieur COICOP (formulation B).

Pipeline complet :
1. Chargement S3 + filtrage des lignes valides
2. Preparation des features (NA, recodage)
3. Calcul de la correspondance hierarchique (niveau 4)
4. Construction de la cible "gagnant" (formulation B)
5. Baseline vote majoritaire (reference triviale)
6. Arbre de decision simple
7. Random Forest (sans feature engineering)
8. Random Forest avec feature engineering enrichi
9. Comparaison globale avec le LLM-as-judge
"""
# %%
import sys
sys.path.append("/home/onyxia/work/codification_auto_coicop")

%load_ext autoreload
%autoreload 2

from data.load_data import charger_base
from src.preprocessing import (
    filtrer_lignes_valides,
    preparer_features,
    construire_X_y,
    COLONNES_CATEGORIELLES,
)
from src.evaluation import (
    evaluer_llm_as_judge,
    diagnostiquer_categories_rares,
    calculer_classifieurs_corrects,
)
from src.cible_gagnant import ajouter_gagnant
from src.baseline import evaluer_baseline_vote
from src.modeling_tree import (
    split_train_test,
    construire_pipeline_arbre,
    entrainer_arbre,
    visualiser_arbre_supertree,
)
from src.modeling_rf import (
    construire_pipeline_rf,
    entrainer_rf,
    afficher_importance_features,
)
from src.evaluation_arbre import evaluer_prediction_gagnant, evaluer_code_final
from src.feature_engineering import (
    appliquer_feature_engineering,
    lister_nouvelles_colonnes,
)

# %% [markdown]
# ## 1. Chargement et exploration

# %%
df = charger_base()
print("Dimensions :", df.shape)
print("\nColonnes :", df.columns.tolist())
print("\nValeurs manquantes par colonne :\n", df.isna().sum())
print("\nApercu :\n", df.head())

# %% [markdown]
# ## 2. Filtrage des lignes valides + accuracy de reference du LLM

# %%
df_valide = filtrer_lignes_valides(df)
accuracy_llm = evaluer_llm_as_judge(df_valide, niveau_max=4)

# %% [markdown]
# ## 3. Preparation des features (gestion des NA, recodage)

# %%
df_valide = preparer_features(df_valide)

# %% [markdown]
# ## 4. Diagnostic des categories rares (descriptif)

# %%
compte_par_categorie = diagnostiquer_categories_rares(df_valide["code"])

# %% [markdown]
# ## 5. Determination des classifieurs corrects (correspondance hierarchique niveau 4)

# %%
df_valide = calculer_classifieurs_corrects(df_valide, niveau_max=4)

# %% [markdown]
# ## 6. Construction de la cible "gagnant" (formulation B)

# %%
df_valide = ajouter_gagnant(df_valide)

# %% [markdown]
# ## 7. Construction de X et y (sans feature engineering -- baseline)

# %%
X, y = construire_X_y(df_valide)
print(X.shape, y.shape)
print("\nValeurs manquantes dans X :\n", X.isna().sum())

# %% [markdown]
# ## 8. Split stratifie train/test

# %%
X_train, X_test, y_train, y_test = split_train_test(X, y)

# %% [markdown]
# ## 9. Baseline vote majoritaire (reference triviale)

# %%
accuracy_baseline = evaluer_baseline_vote(X_test, df_valide, niveau_max=4)

# %% [markdown]
# ## 10. Arbre de decision simple

# %%
pipeline_arbre = construire_pipeline_arbre(max_depth=5, min_samples_leaf=20)
pipeline_arbre = entrainer_arbre(pipeline_arbre, X_train, y_train)

# %%
y_pred_arbre = evaluer_prediction_gagnant(pipeline_arbre, X_test, y_test)
accuracy_arbre, _ = evaluer_code_final(y_pred_arbre, X_test, df_valide, niveau_max=4)

# %%
# Visualisation interactive de l'arbre
visualiser_arbre_supertree(
    pipeline_arbre, X_train, y_train,
    save_html_path="/home/onyxia/work/codification_auto_coicop/outputs/arbre_simple.html"
)

# %% [markdown]
# ## 11. Random Forest (sans feature engineering)

# %%
pipeline_rf = construire_pipeline_rf(
    n_estimators=300, max_depth=None, min_samples_leaf=5,
)
pipeline_rf = entrainer_rf(pipeline_rf, X_train, y_train)

# %%
y_pred_rf = evaluer_prediction_gagnant(pipeline_rf, X_test, y_test)
accuracy_rf, _ = evaluer_code_final(y_pred_rf, X_test, df_valide, niveau_max=4)

# %%
df_importance_rf = afficher_importance_features(pipeline_rf, top_n=20)

# %% [markdown]
# ## 12. Random Forest avec feature engineering enrichi
# Ajoute : codes eclates par niveau hierarchique, accords pairwise par niveau,
# statistiques agregees sur les scores.

# %%
df_valide_fe = appliquer_feature_engineering(df_valide, niveau_max=4)

cat_extra, num_extra = lister_nouvelles_colonnes(niveau_max=4)
print(f"Nouvelles features categorielles : {len(cat_extra)}")
print(f"Nouvelles features numeriques : {len(num_extra)}")

# %%
X_fe, y_fe = construire_X_y(
    df_valide_fe,
    colonnes_categorielles_extra=cat_extra,
    colonnes_numeriques_extra=num_extra,
)
print(X_fe.shape, y_fe.shape)

# Meme random_state pour conserver le meme split -> comparaison directe
X_train_fe, X_test_fe, y_train_fe, y_test_fe = split_train_test(X_fe, y_fe)

# %%
pipeline_rf_fe = construire_pipeline_rf(
    n_estimators=300, max_depth=None, min_samples_leaf=5,
    colonnes_categorielles=COLONNES_CATEGORIELLES + cat_extra,
)
pipeline_rf_fe = entrainer_rf(pipeline_rf_fe, X_train_fe, y_train_fe)

# %%
y_pred_rf_fe = evaluer_prediction_gagnant(pipeline_rf_fe, X_test_fe, y_test_fe)
accuracy_rf_fe, _ = evaluer_code_final(y_pred_rf_fe, X_test_fe, df_valide_fe, niveau_max=4)

# %%
df_importance_rf_fe = afficher_importance_features(pipeline_rf_fe, top_n=30)

# %% [markdown]
# ## 13. Comparaison globale

# %%
print("\n=== Recapitulatif des performances (niveau 4) ===")
print(f"  Baseline vote majoritaire     : {accuracy_baseline:.3f}")
print(f"  Arbre simple                  : {accuracy_arbre:.3f}")
print(f"  Random Forest (sans FE)       : {accuracy_rf:.3f}")
print(f"  Random Forest (avec FE)       : {accuracy_rf_fe:.3f}")
print(f"  LLM-as-judge (reference)      : {accuracy_llm:.3f}")
# %%
