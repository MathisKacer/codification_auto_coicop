"""
Exploration et modelisation du meta-classifieur COICOP (formulation B).

Pipeline complet :
1. Chargement S3 + filtrage des lignes valides
2. Preparation des features (NA, recodage)
3. Calcul de la correspondance hierarchique (niveau 4)
4. Construction de la cible "gagnant"
5. Baseline vote majoritaire (reference triviale)
6. Arbre de decision simple
7. Random Forest + OneHot
8. Random Forest + OrdinalEncoder (resout la dilution)
9. Random Forest + OrdinalEncoder + Feature Engineering (configuration la plus aboutie)

"""
# %%
import sys
sys.path.append("/home/onyxia/work/codification_auto_coicop")

%load_ext autoreload
%autoreload 2

from sklearn.metrics import accuracy_score

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
    construire_pipeline_rf_ordinal,
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

# %% [markdown]
# ## 2. Filtrage des lignes valides et accuracy de reference du LLM

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
# ## 6. Construction de la cible "gagnant"

# %%
df_valide = ajouter_gagnant(df_valide)

# %% [markdown]
# ## 7. Split train / test
# Le split est fait une fois et reutilise pour TOUS les modeles sans FE,
# pour que les comparaisons soient sur exactement le meme jeu de test.

# %%
X, y = construire_X_y(df_valide)
X_train, X_test, y_train, y_test = split_train_test(X, y)

# %% [markdown]
# ## 8. Baseline vote majoritaire (reference triviale)

# %%
accuracy_baseline = evaluer_baseline_vote(X_test, df_valide, niveau_max=4)

# %% [markdown]
# ## 9. Arbre de decision simple (OneHot)

# %%
pipeline_arbre = construire_pipeline_arbre(max_depth=5, min_samples_leaf=20)
pipeline_arbre = entrainer_arbre(pipeline_arbre, X_train, y_train)

y_pred_arbre = evaluer_prediction_gagnant(pipeline_arbre, X_test, y_test)
accuracy_arbre, _ = evaluer_code_final(y_pred_arbre, X_test, df_valide, niveau_max=4)

# %%
visualiser_arbre_supertree(
    pipeline_arbre, X_train, y_train
    )

# %% [markdown]
# ## 10. Random Forest + OneHot (sans FE)

# %%
pipeline_rf_oh = construire_pipeline_rf(
    n_estimators=300, max_depth=None, min_samples_leaf=5,
)
pipeline_rf_oh = entrainer_rf(pipeline_rf_oh, X_train, y_train)

X_t = pipeline_rf_oh.named_steps["prep"].transform(X_train)
print(f"Shape apres OneHot : {X_t.shape}")

acc_train_oh = accuracy_score(y_train, pipeline_rf_oh.predict(X_train))
acc_test_oh = accuracy_score(y_test, pipeline_rf_oh.predict(X_test))
print(f"Train: {acc_train_oh:.3f} | Test: {acc_test_oh:.3f}")

y_pred_rf_oh = evaluer_prediction_gagnant(pipeline_rf_oh, X_test, y_test)
accuracy_rf_oh, _ = evaluer_code_final(y_pred_rf_oh, X_test, df_valide, niveau_max=4)

# %% [markdown]
# ## 11. Random Forest + OrdinalEncoder (sans FE)
# Resout la dilution : ~15 colonnes denses au lieu de 1655 creuses.

# %%
pipeline_rf_ord = construire_pipeline_rf_ordinal(
    n_estimators=300, max_depth=None, min_samples_leaf=5,
)
pipeline_rf_ord = entrainer_rf(pipeline_rf_ord, X_train, y_train)

X_t = pipeline_rf_ord.named_steps["prep"].transform(X_train)
print(f"Shape apres OrdinalEncoder : {X_t.shape}")

acc_train_ord = accuracy_score(y_train, pipeline_rf_ord.predict(X_train))
acc_test_ord = accuracy_score(y_test, pipeline_rf_ord.predict(X_test))
print(f"Train: {acc_train_ord:.3f} | Test: {acc_test_ord:.3f}")

y_pred_rf_ord = evaluer_prediction_gagnant(pipeline_rf_ord, X_test, y_test)
accuracy_rf_ord, _ = evaluer_code_final(y_pred_rf_ord, X_test, df_valide, niveau_max=4)

# %%
df_importance_ord = afficher_importance_features(pipeline_rf_ord, top_n=20)

# %% [markdown]
# ## 12. Random Forest + OrdinalEncoder + Feature Engineering
# Le FE ajoute : codes par niveau hierarchique, accords pairwise par niveau,
# nombre d'accords agreges par niveau.

# %%
df_valide_fe = appliquer_feature_engineering(df_valide, niveau_max=4)
cat_extra, num_extra = lister_nouvelles_colonnes(niveau_max=4)
print(f"Nouvelles features categorielles : {len(cat_extra)}")
print(f"Nouvelles features numeriques : {len(num_extra)}")

X_fe, y_fe = construire_X_y(
    df_valide_fe,
    colonnes_categorielles_extra=cat_extra,
    colonnes_numeriques_extra=num_extra,
)
X_train_fe, X_test_fe, y_train_fe, y_test_fe = split_train_test(X_fe, y_fe)

# %%
pipeline_rf_ord_fe = construire_pipeline_rf_ordinal(
    n_estimators=300, max_depth=None, min_samples_leaf=5,
    colonnes_categorielles=COLONNES_CATEGORIELLES + cat_extra,
)
pipeline_rf_ord_fe = entrainer_rf(pipeline_rf_ord_fe, X_train_fe, y_train_fe)

X_t = pipeline_rf_ord_fe.named_steps["prep"].transform(X_train_fe)
print(f"Shape apres OrdinalEncoder + FE : {X_t.shape}")

acc_train_fe = accuracy_score(y_train_fe, pipeline_rf_ord_fe.predict(X_train_fe))
acc_test_fe = accuracy_score(y_test_fe, pipeline_rf_ord_fe.predict(X_test_fe))
print(f"Train: {acc_train_fe:.3f} | Test: {acc_test_fe:.3f}")

y_pred_rf_ord_fe = evaluer_prediction_gagnant(pipeline_rf_ord_fe, X_test_fe, y_test_fe)
accuracy_rf_ord_fe, _ = evaluer_code_final(y_pred_rf_ord_fe, X_test_fe, df_valide_fe, niveau_max=4)

# %%
df_importance_ord_fe = afficher_importance_features(pipeline_rf_ord_fe, top_n=25)

# %% [markdown]
# ## 13. Recapitulatif comparatif

# %%
print("\n" + "=" * 60)
print("RECAPITULATIF DES PERFORMANCES (niveau 4)")
print("=" * 60)
print(f"  Baseline vote majoritaire              : {accuracy_baseline:.3f}")
print(f"  Arbre simple (OneHot)                  : {accuracy_arbre:.3f}")
print(f"  Random Forest + OneHot (sans FE)       : {accuracy_rf_oh:.3f}")
print(f"  Random Forest + Ordinal (sans FE)      : {accuracy_rf_ord:.3f}")
print(f"  Random Forest + Ordinal + FE           : {accuracy_rf_ord_fe:.3f}")
print(f"  LLM-as-judge (reference)               : {accuracy_llm:.3f}")
print("=" * 60)

# %% [markdown]
# ## 14. Analyse sur les cas resolubles uniquement (sans 'aucun')
# On exclut les lignes ou aucun classifieur n'avait raison au niveau 4 :

# %%
df_valide_resoluble = df_valide[df_valide["gagnant"] != "aucun"].copy()
print(f"Lignes resolubles : {len(df_valide_resoluble)} / {len(df_valide)} "
      f"({len(df_valide_resoluble)/len(df_valide):.1%})")
print(f"\nDistribution gagnant (sans 'aucun') :\n{df_valide_resoluble['gagnant'].value_counts()}")

# %% [markdown]
# ### 14a. Accuracy du LLM sur ce sous-ensemble (reference)
# Important : on recalcule l'accuracy du LLM uniquement sur ces lignes,
# pour qu'elle soit comparable a celle du meta-modele.

# %%
from src.correspondance_hierarchique import a_raison_jusqu_a_niveau

correspondances_llm = df_valide_resoluble.apply(
    lambda row: a_raison_jusqu_a_niveau(row["llm_code"], row["code"], 4),
    axis=1,
)
accuracy_llm_resoluble = correspondances_llm.mean()
print(f"Accuracy LLM sur lignes resolubles : {accuracy_llm_resoluble:.3f}")

# %% [markdown]
# ### 14b. Baseline vote majoritaire sur ce sous-ensemble

# %%
# On a besoin de re-splitter avant
df_valide_resoluble_fe = appliquer_feature_engineering(df_valide_resoluble, niveau_max=4)
X_r_fe, y_r_fe = construire_X_y(
    df_valide_resoluble_fe,
    colonnes_categorielles_extra=cat_extra,
    colonnes_numeriques_extra=num_extra,
)
X_train_r, X_test_r, y_train_r, y_test_r = split_train_test(X_r_fe, y_r_fe)

accuracy_baseline_resoluble = evaluer_baseline_vote(X_test_r, df_valide_resoluble_fe, niveau_max=4)

# %% [markdown]
# ### 14c. Random Forest + Ordinal + FE sur lignes resolubles

# %%
pipeline_rf_resoluble = construire_pipeline_rf_ordinal(
    n_estimators=300, max_depth=None, min_samples_leaf=5,
    colonnes_categorielles=COLONNES_CATEGORIELLES + cat_extra,
)
pipeline_rf_resoluble = entrainer_rf(pipeline_rf_resoluble, X_train_r, y_train_r)

acc_train_r = accuracy_score(y_train_r, pipeline_rf_resoluble.predict(X_train_r))
acc_test_r = accuracy_score(y_test_r, pipeline_rf_resoluble.predict(X_test_r))
print(f"Train: {acc_train_r:.3f} | Test: {acc_test_r:.3f}")

y_pred_r = evaluer_prediction_gagnant(pipeline_rf_resoluble, X_test_r, y_test_r)
accuracy_rf_resoluble, _ = evaluer_code_final(y_pred_r, X_test_r, df_valide_resoluble_fe, niveau_max=4)

# %% [markdown]
# ## 15. Recapitulatif : performance sur lignes resolubles uniquement

# %%
print("\n" + "=" * 60)
print("PERFORMANCES SUR LIGNES RESOLUBLES (gagnant != 'aucun')")
print("=" * 60)
print(f"  Lignes resolubles                       : {len(df_valide_resoluble)} ({len(df_valide_resoluble)/len(df_valide):.1%})")
print(f"  Baseline vote majoritaire               : {accuracy_baseline_resoluble:.3f}")
print(f"  Random Forest + Ordinal + FE            : {accuracy_rf_resoluble:.3f}")
print(f"  LLM-as-judge                            : {accuracy_llm_resoluble:.3f}")
print("=" * 60)

# %%
print("Colonnes utilisees dans X :")
print(X_train_fe.columns.tolist())
# %%
df_test_resoluble = df_valide_resoluble_fe.loc[X_test_r.index]
correspondances_llm_test = df_test_resoluble.apply(
    lambda row: a_raison_jusqu_a_niveau(row["llm_code"], row["code"], 4),
    axis=1,
)
accuracy_llm_resoluble_test = correspondances_llm_test.mean()
print(f"Accuracy LLM sur test set des resolubles : {accuracy_llm_resoluble_test:.3f}")
