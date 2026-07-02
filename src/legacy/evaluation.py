"""
Fonctions d'evaluation et de diagnostic pour le projet de codification COICOP.
"""
import pandas as pd
from sklearn.metrics import accuracy_score
from src.coicop import a_raison_jusqu_a_niveau


def evaluer_llm_as_judge(df_valide: pd.DataFrame, niveau_max: int = 4) -> float:
    """
    Calcule l'accuracy actuelle du LLM-as-judge sur les lignes valides,
    en evaluant la correspondance jusqu'au niveau hierarchique cible.
    """
    correspondances = df_valide.apply(
        lambda row: a_raison_jusqu_a_niveau(row["llm_code"], row["code"], niveau_max),
        axis=1
    )
    accuracy_llm = correspondances.mean()
    print(f"Accuracy actuelle du LLM-as-judge (niveau {niveau_max}) : {accuracy_llm:.3f}")
    return accuracy_llm


def diagnostiquer_categories_rares(y: pd.Series) -> pd.Series:
    """
    Affiche des statistiques sur la distribution des categories (codes COICOP)
    et identifie les categories sous-representees.
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


def calculer_classifieurs_corrects(df_valide: pd.DataFrame, niveau_max: int = 4) -> pd.DataFrame:
    """
    Pour chaque ligne, determine quels classifieurs individuels ont propose le bon code
    jusqu'au niveau hierarchique cible. Pour TTC, prend le max sur ses 3 suggestions.

    Ajoute les colonnes : lcs_raison, rag_raison, ragann_raison, ttc_raison,
    nb_classifieurs_corrects.
    """
    df_valide["lcs_raison"] = df_valide.apply(
        lambda row: a_raison_jusqu_a_niveau(row["lcs_code"], row["code"], niveau_max), axis=1
    )
    df_valide["rag_raison"] = df_valide.apply(
        lambda row: a_raison_jusqu_a_niveau(row["rag_code"], row["code"], niveau_max), axis=1
    )
    df_valide["ragann_raison"] = df_valide.apply(
        lambda row: a_raison_jusqu_a_niveau(row["ragann_code"], row["code"], niveau_max), axis=1
    )
    df_valide["ttc_raison"] = df_valide.apply(
        lambda row: max(
            a_raison_jusqu_a_niveau(row["ttc_code_1"], row["code"], niveau_max),
            a_raison_jusqu_a_niveau(row["ttc_code_2"], row["code"], niveau_max),
            a_raison_jusqu_a_niveau(row["ttc_code_3"], row["code"], niveau_max),
        ), axis=1
    )

    df_valide["nb_classifieurs_corrects"] = (
        df_valide["lcs_raison"] + df_valide["rag_raison"]
        + df_valide["ragann_raison"] + df_valide["ttc_raison"]
    )
    print(df_valide["nb_classifieurs_corrects"].value_counts().sort_index())
    return df_valide