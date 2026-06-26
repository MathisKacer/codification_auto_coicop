"""
Fonctions d'évaluation et de diagnostic pour le projet de codification COICOP.
"""

import pandas as pd
from sklearn.metrics import accuracy_score


def evaluer_llm_as_judge(df_valide: pd.DataFrame) -> float:
    """
    Calcule l'accuracy actuelle du LLM-as-judge sur les lignes valides.

    Parameters
    ----------
    df_valide : DataFrame contenant "code" (vérité) et "llm_code" (prédiction LLM)

    Returns
    -------
    accuracy : float
    """
    accuracy_llm = accuracy_score(df_valide["code"], df_valide["llm_code"])
    print(f"Accuracy actuelle du LLM-as-judge : {accuracy_llm:.3f}")
    return accuracy_llm


def diagnostiquer_categories_rares(y: pd.Series) -> pd.Series:
    """
    Affiche des statistiques sur la distribution des catégories (codes COICOP)
    et identifie les catégories sous-représentées.

    Parameters
    ----------
    y : Series des codes vrais

    Returns
    -------
    compte_par_categorie : Series (code -> nombre d'exemples), triée par value_counts
    """
    compte_par_categorie = y.value_counts()
    print(f"Nombre total de categories distinctes : {len(compte_par_categorie)}")
    print(f"Categories avec 1 seul exemple : {(compte_par_categorie == 1).sum()}")
    print(f"Categories avec moins de 5 exemples : {(compte_par_categorie < 5).sum()}")
    print(f"Categories avec moins de 10 exemples : {(compte_par_categorie < 10).sum()}")

    categories_rares = compte_par_categorie[compte_par_categorie < 5].index
    part_lignes_rares = y.isin(categories_rares).mean()
    print(f"\nPart des lignes concernees par des categories <5 exemples : {part_lignes_rares:.1%}")

    return compte_par_categorie


def calculer_classifieurs_corrects(df_valide: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque ligne, détermine quels classifieurs individuels (LCS, RAG,
    RAG-ANN, TTC) ont proposé le bon code, et compte le nombre total de
    classifieurs corrects.

    Ajoute les colonnes : lcs_raison, rag_raison, ragann_raison, ttc_raison,
    nb_classifieurs_corrects.

    Parameters
    ----------
    df_valide : DataFrame contenant "code" et les colonnes de prédiction
        (lcs_code, rag_code, ragann_code, ttc_code_1/2/3)

    Returns
    -------
    Le même DataFrame, modifié en place et retourné pour chaînage.
    """
    df_valide["lcs_raison"] = (df_valide["lcs_code"] == df_valide["code"]).astype(int)
    df_valide["rag_raison"] = (df_valide["rag_code"] == df_valide["code"]).astype(int)
    df_valide["ragann_raison"] = (df_valide["ragann_code"] == df_valide["code"]).astype(int)
    df_valide["ttc_raison"] = (
        (df_valide["ttc_code_1"] == df_valide["code"]) |
        (df_valide["ttc_code_2"] == df_valide["code"]) |
        (df_valide["ttc_code_3"] == df_valide["code"])
    ).astype(int)
    df_valide["nb_classifieurs_corrects"] = (
        df_valide["lcs_raison"] + df_valide["rag_raison"]
        + df_valide["ragann_raison"] + df_valide["ttc_raison"]
    )
    print(df_valide["nb_classifieurs_corrects"].value_counts().sort_index())

    return df_valide
