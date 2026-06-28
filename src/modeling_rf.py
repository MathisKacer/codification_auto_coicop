"""
Construction et entrainement du meta-modele Random Forest (formulation B) :
ensemble d'arbres pour predire quel classifieur suivre.
"""
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import COLONNES_CATEGORIELLES


def construire_pipeline_rf(
    n_estimators: int = 300,
    max_depth: int = None,
    min_samples_leaf: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
    colonnes_categorielles: list[str] = None,
) -> Pipeline:
    """
    Construit un pipeline complet avec Random Forest comme classifieur final.

    Parameters
    ----------
    colonnes_categorielles : liste des colonnes a one-hot encoder.
        Si None, utilise COLONNES_CATEGORIELLES par defaut (comportement d'origine).
        Sinon, utilise la liste fournie (utile avec du feature engineering).
    """
    if colonnes_categorielles is None:
        colonnes_categorielles = COLONNES_CATEGORIELLES

    preprocesseur = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), colonnes_categorielles),
    ], remainder="passthrough")

    pipeline = Pipeline([
        ("prep", preprocesseur),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=n_jobs,
        ))
    ])
    return pipeline


def entrainer_rf(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Entraine le pipeline Random Forest sur les donnees de train."""
    pipeline.fit(X_train, y_train)
    return pipeline


def afficher_importance_features(pipeline: Pipeline, top_n: int = 20) -> pd.DataFrame:
    """
    Affiche les top_n features les plus importantes selon la Random Forest.
    Utile pour interpreter ce que le modele utilise pour decider.
    """
    rf = pipeline.named_steps["clf"]
    noms_features = pipeline.named_steps["prep"].get_feature_names_out()

    df_importance = pd.DataFrame({
        "feature": noms_features,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)

    print(f"Top {top_n} features les plus importantes :")
    print(df_importance.head(top_n).to_string(index=False))
    return df_importance