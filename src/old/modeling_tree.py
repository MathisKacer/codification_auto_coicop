"""
Construction et entrainement du meta-modele (formulation B) :
predire quel classifieur suivre parmi LCS, RAG, RAGANN, TTC, plusieurs_corrects, aucun.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from src.preprocessing import COLONNES_CATEGORIELLES


def split_train_test(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Split stratifie sur la cible 'gagnant'.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"Train: {X_train.shape[0]} lignes | Test: {X_test.shape[0]} lignes")
    return X_train, X_test, y_train, y_test


def construire_pipeline_arbre(
    max_depth: int = 5,
    min_samples_leaf: int = 20,
    random_state: int = 42,
    colonnes_categorielles: list[str] = None,
) -> Pipeline:
    if colonnes_categorielles is None:
        colonnes_categorielles = COLONNES_CATEGORIELLES

    preprocesseur = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), colonnes_categorielles),
    ], remainder="passthrough")

    pipeline = Pipeline([
        ("prep", preprocesseur),
        ("clf", DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=random_state,
        ))
    ])
    return pipeline

def entrainer_arbre(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Entraine le pipeline sur les donnees de train."""
    pipeline.fit(X_train, y_train)
    return pipeline


def visualiser_arbre_supertree(pipeline, X_train, y_train, save_html_path: str = None):
    """
    Visualisation interactive de l'arbre avec supertree.
    """
    from supertree import SuperTree
    import scipy.sparse

    arbre = pipeline.named_steps["clf"]
    X_train_transforme = pipeline.named_steps["prep"].transform(X_train)
    noms_features = pipeline.named_steps["prep"].get_feature_names_out()

    # SuperTree n'accepte pas les matrices sparse -> conversion en dense
    if scipy.sparse.issparse(X_train_transforme):
        X_train_transforme = X_train_transforme.toarray()

    super_tree = SuperTree(
        arbre,
        X_train_transforme,
        y_train,
        noms_features,
        arbre.classes_,
    )

    if save_html_path:
        super_tree.save_html(save_html_path)
        print(f"Arbre sauvegarde en HTML : {save_html_path}")
    else:
        super_tree.show_tree()
