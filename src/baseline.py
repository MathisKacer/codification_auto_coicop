"""
Baselines de reference pour evaluer l'apport reel du meta-modele arbre.
Si l'arbre ne fait pas mieux que ces baselines triviaux, c'est qu'il
n'apporte pas de vraie capacite d'arbitrage.
"""
import pandas as pd
from collections import Counter

from src.correspondance_hierarchique import tronquer_a_niveau, a_raison_jusqu_a_niveau


def baseline_vote_majoritaire(row: pd.Series, niveau_max: int = 4) -> str:
    """
    Pour une ligne donnee, retourne le code le plus frequent parmi les
    suggestions des 4 classifieurs (tronques au niveau cible).
    """
    codes = [
        tronquer_a_niveau(row["lcs_code"], niveau_max),
        tronquer_a_niveau(row["rag_code"], niveau_max),
        tronquer_a_niveau(row["ragann_code"], niveau_max),
        tronquer_a_niveau(row["ttc_code_1"], niveau_max),
    ]
    codes_valides = [c for c in codes if c is not None and pd.notna(c)]
    if not codes_valides:
        return "PREDICTION_IMPOSSIBLE"
    return Counter(codes_valides).most_common(1)[0][0]


def evaluer_baseline_vote(X_test, df_valide, niveau_max: int = 4) -> float:
    """
    Evalue le baseline 'vote majoritaire' sur le jeu de test,
    pour le comparer a l'arbre et au LLM.
    """
    df_test = df_valide.loc[X_test.index]
    y_vrai = df_test["code"].values

    y_baseline = df_test.apply(
        lambda row: baseline_vote_majoritaire(row, niveau_max), axis=1
    ).values

    correct = [
        a_raison_jusqu_a_niveau(pred, vrai, niveau_max)
        for pred, vrai in zip(y_baseline, y_vrai)
    ]
    accuracy = sum(correct) / len(correct)
    print(f"Accuracy baseline vote majoritaire (niveau {niveau_max}) : {accuracy:.3f}")
    return accuracy