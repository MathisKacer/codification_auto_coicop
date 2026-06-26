"""
Préprocessing des données pour le projet de codification automatique COICOP.

Contient les fonctions de filtrage des lignes valides et de préparation
des features (gestion des valeurs manquantes, recodage).
"""

import pandas as pd

COLONNES_CATEGORIELLES = [
    "lcs_code", "rag_code", "ragann_code", "ttc_code_1", "ttc_code_2", "ttc_code_3",
    "shop_type_name", "codable"
]

COLONNES_NUMERIQUES = [
    "lcs_distance", "rag_confidence", "ragann_confidence", "ttc_conf_1", "ttc_conf_2", "ttc_conf_3",
    "budget"
]


def filtrer_lignes_valides(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enlève les lignes dont il manque la coicop ou la réponse du LLM.

    Parameters
    ----------
    df : DataFrame brut issu de charger_base()

    Returns
    -------
    DataFrame ne contenant que les lignes avec "code" et "llm_code" non nuls.
    """
    df_valide = df.dropna(subset=["code", "llm_code"]).copy()
    n_exclues = df.shape[0] - df_valide.shape[0]
    print(f"Lignes valides : {df_valide.shape[0]} / {df.shape[0]} ({n_exclues} lignes exclues)")
    return df_valide


def preparer_features(df_valide: pd.DataFrame) -> pd.DataFrame:
    """
    Recode les valeurs manquantes des colonnes utilisées comme features.

    Modifie/ajoute les colonnes suivantes :
    - lcs_code, lcs_distance
    - rag_code
    - ragann_code
    - shop_type_name
    - budget

    Note méthodologique : on n'ajoute pas d'indicatrice de manquant
    (ex. lcs_manquant) car l'absence est déjà encodée par la catégorie
    sentinelle elle-même (AUCUNE_SUGGESTION / NON_CODABLE), à condition
    que les catégorielles soient one-hot encodées ou gérées nativement
    par le modèle (ex. XGBoost avec enable_categorical=True). Une
    indicatrice séparée serait redondante avec cette information.

    Parameters
    ----------
    df_valide : DataFrame déjà filtré (cf. filtrer_lignes_valides)

    Returns
    -------
    Le même DataFrame, modifié en place et retourné pour chaînage.
    """
    df_valide["lcs_code"] = df_valide["lcs_code"].fillna("AUCUNE_SUGGESTION")
    df_valide["lcs_distance"] = df_valide["lcs_distance"].fillna(df_valide["lcs_distance"].max() * 1.5)

    df_valide["rag_code"] = df_valide["rag_code"].fillna("NON_CODABLE")

    df_valide["ragann_code"] = df_valide["ragann_code"].fillna("NON_CODABLE")

    df_valide["shop_type_name"] = df_valide["shop_type_name"].fillna("INCONNU")
    df_valide["budget"] = df_valide["budget"].fillna(df_valide["budget"].median())

    return df_valide


def construire_X_y(df_valide: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Construit la matrice de features X et la cible y à partir du DataFrame préparé.

    Parameters
    ----------
    df_valide : DataFrame après preparer_features()

    Returns
    -------
    (X, y)
    """
    X = df_valide[COLONNES_CATEGORIELLES + COLONNES_NUMERIQUES]
    y = df_valide["code"]
    return X, y
