"""
Preprocessing des donnees pour le projet de codification automatique COICOP.
Contient les fonctions de filtrage des lignes valides et de preparation
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
    Enleve les lignes dont il manque le code de verite terrain ou la reponse du LLM.
    """
    df_valide = df.dropna(subset=["code", "llm_code"]).copy()
    n_exclues = df.shape[0] - df_valide.shape[0]
    print(f"Lignes valides : {df_valide.shape[0]} / {df.shape[0]} ({n_exclues} lignes exclues)")
    return df_valide


def preparer_features(df_valide: pd.DataFrame) -> pd.DataFrame:
    """
    Recode les valeurs manquantes des colonnes utilisees comme features.

    Note methodologique : on n'ajoute pas d'indicatrice de manquant
    (ex. lcs_manquant) car l'absence est deja encodee par la categorie
    sentinelle elle-meme (AUCUNE_SUGGESTION / NON_CODABLE), a condition
    que les categorielles soient one-hot encodees ou gerees nativement
    par le modele (ex. XGBoost avec enable_categorical=True). Une
    indicatrice separee serait redondante avec cette information.
    """
    df_valide["lcs_code"] = df_valide["lcs_code"].fillna("AUCUNE_SUGGESTION")
    df_valide["lcs_distance"] = df_valide["lcs_distance"].fillna(df_valide["lcs_distance"].max() * 1.5)
    df_valide["rag_code"] = df_valide["rag_code"].fillna("NON_CODABLE")
    df_valide["ragann_code"] = df_valide["ragann_code"].fillna("NON_CODABLE")
    df_valide["shop_type_name"] = df_valide["shop_type_name"].fillna("INCONNU")
    df_valide["budget"] = df_valide["budget"].fillna(df_valide["budget"].median())
    return df_valide


def construire_X_y(
    df_valide: pd.DataFrame,
    colonnes_categorielles_extra: list[str] = None,
    colonnes_numeriques_extra: list[str] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Construit la matrice de features X et la cible y (le 'gagnant' a predire).

    Les arguments extra permettent d'ajouter des features derivees (feature engineering)
    aux colonnes de base. Sans les fournir, le comportement est identique a la version
    originale (utilise uniquement COLONNES_CATEGORIELLES et COLONNES_NUMERIQUES).

    Prerequis : la colonne 'gagnant' doit avoir ete creee au prealable
    via ajouter_gagnant() (depuis src.legacy.cible_gagnant).
    """
    if "gagnant" not in df_valide.columns:
        raise ValueError(
            "La colonne 'gagnant' n'existe pas. "
            "Appelle ajouter_gagnant() avant construire_X_y()."
        )

    cat = COLONNES_CATEGORIELLES + (colonnes_categorielles_extra or [])
    num = COLONNES_NUMERIQUES + (colonnes_numeriques_extra or [])

    X = df_valide[cat + num]
    y = df_valide["gagnant"]
    return X, y