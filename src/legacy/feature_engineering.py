"""
Feature engineering : construction de features derivees pour enrichir le meta-modele.
- Codes eclates par niveau hierarchique (donne la structure de la nomenclature)
- Accords pairwise entre classifieurs, par niveau (signal de consensus)
- Statistiques agregees sur les scores
"""
import pandas as pd
from itertools import combinations

from src.coicop import tronquer_niveau


# Identifiants des 4 classifieurs et de leur colonne de code principale
CLASSIFIEURS = {
    "lcs": "lcs_code",
    "rag": "rag_code",
    "ragann": "ragann_code",
    "ttc": "ttc_code_1",  # on utilise la suggestion principale de TTC pour les accords
}


def ajouter_codes_par_niveau(df: pd.DataFrame, niveau_max: int = 4) -> pd.DataFrame:
    """
    Eclate chaque code en colonnes par niveau hierarchique.
    Ex: lcs_code = "01.1.2.3" devient lcs_code_n1="01", lcs_code_n2="01.1", etc.
    """
    df = df.copy()
    for nom, col_code in CLASSIFIEURS.items():
        for niveau in range(1, niveau_max + 1):
            df[f"{nom}_code_n{niveau}"] = df[col_code].apply(
                lambda c: tronquer_niveau(c, niveau) if pd.notna(c) else "AUCUNE_SUGGESTION"
            )
    return df


def ajouter_accords_par_niveau(df: pd.DataFrame, niveau_max: int = 4) -> pd.DataFrame:
    """
    Pour chaque paire de classifieurs et chaque niveau, indicateur binaire :
    sont-ils d'accord a ce niveau hierarchique ?
    Aussi : nombre total d'accords par niveau (resume agrege).
    """
    df = df.copy()
    noms = list(CLASSIFIEURS.keys())

    for niveau in range(1, niveau_max + 1):
        accords_du_niveau = []
        for nom_a, nom_b in combinations(noms, 2):
            col_a = f"{nom_a}_code_n{niveau}"
            col_b = f"{nom_b}_code_n{niveau}"
            col_accord = f"accord_{nom_a}_{nom_b}_n{niveau}"
            df[col_accord] = (df[col_a] == df[col_b]).astype(int)
            accords_du_niveau.append(col_accord)

        # Resume agrege : nombre d'accords pairwise a ce niveau (0 a 6)
        df[f"nb_accords_n{niveau}"] = df[accords_du_niveau].sum(axis=1)

    return df


def appliquer_feature_engineering(df: pd.DataFrame, niveau_max: int = 4) -> pd.DataFrame:
    """
    Pipeline de feature engineering simplifie : uniquement codes par niveau
    et accords pairwise. Les statistiques sur les scores ne sont pas ajoutees,
    le modele a deja acces aux scores bruts.
    """
    df = ajouter_codes_par_niveau(df, niveau_max=niveau_max)
    df = ajouter_accords_par_niveau(df, niveau_max=niveau_max)
    return df


def lister_nouvelles_colonnes(niveau_max: int = 4) -> tuple[list[str], list[str]]:
    """
    Retourne (categorielles_nouvelles, numeriques_nouvelles) pour les ajouter
    aux features de base. Version simplifiee sans stats sur les scores.
    """
    noms = list(CLASSIFIEURS.keys())

    # Codes par niveau : categoriels
    categorielles = [
        f"{nom}_code_n{niveau}"
        for nom in noms
        for niveau in range(1, niveau_max + 1)
    ]

    # Accords par niveau + nb_accords agrege : tous numeriques (binaires ou entiers)
    numeriques = []
    for niveau in range(1, niveau_max + 1):
        for nom_a, nom_b in combinations(noms, 2):
            numeriques.append(f"accord_{nom_a}_{nom_b}_n{niveau}")
        numeriques.append(f"nb_accords_n{niveau}")

    return categorielles, numeriques
