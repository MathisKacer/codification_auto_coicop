"""
Preprocessing pour le modele binaire "la baseline est-elle correcte ?".

Cible :
    y = 1 si la baseline (vote majoritaire, TTC en cas d'egalite) predit
    le bon code au niveau 4, 0 sinon.

Features :
    - codes predits par chaque classifieur, tronques au niveau 1 (division)
    - code choisi par la baseline, tronque au niveau 1
    - nb_accords_max : taille du plus gros groupe d'accord au niveau 4
    - unanimite : 1 si les 4 classifieurs sont d'accord au niveau 4
    - lcs_distance, rag_confidence, ragann_confidence, ttc_conf_1
    - budget, shop_type_name, source
"""
import pandas as pd
import numpy as np
from collections import Counter
from src.coicop import tronquer_niveau


COLS_CLASSIFIEURS = ["lcs_code", "rag_code", "ragann_code", "ttc_code_1"]

COLS_CATEGORIELLES = [
    "lcs_code_n1", "rag_code_n1", "ragann_code_n1", "ttc_code_1_n1",
    "pred_baseline_n1", "shop_type_name", "source",
]
COLS_NUMERIQUES = [
    "nb_accords_max", "unanimite",
    "lcs_distance", "rag_confidence", "ragann_confidence", "ttc_conf_1",
    "budget",
]


def construire_cible(df, y_pred_baseline, col_vrai, niveau=4):
    """
    Cible binaire : 1 si la baseline predit le bon code au niveau donne.
    """
    vrai_tronq = df[col_vrai].map(lambda x: tronquer_niveau(x, niveau))
    return (y_pred_baseline == vrai_tronq).astype(int).rename("baseline_correcte")


def calculer_nb_accords_max(df, cols_pred, niveau=4):
    """
    Pour chaque ligne : taille du plus gros groupe d'accord (au niveau donne)
    parmi les classifieurs. NaN ignores.
    """
    tronq = df[cols_pred].apply(
        lambda col: col.map(lambda x: tronquer_niveau(x, niveau))
    )

    def max_accords(row):
        codes = [v for v in row if pd.notna(v)]
        if not codes:
            return 0
        return max(Counter(codes).values())

    return tronq.apply(max_accords, axis=1)


def construire_features(df, y_pred_baseline):
    """
    Construit la matrice X des features pour le modele binaire.
    """
    X = pd.DataFrame(index=df.index)

    # 1. Codes tronques au niveau 1 (division)
    for c in COLS_CLASSIFIEURS:
        X[f"{c}_n1"] = df[c].map(lambda x: tronquer_niveau(x, niveau=1))

    # Code choisi par la baseline, tronque au niveau 1
    X["pred_baseline_n1"] = y_pred_baseline.map(lambda x: tronquer_niveau(x, niveau=1))

    # 2. Accords entre classifieurs (au niveau 4)
    X["nb_accords_max"] = calculer_nb_accords_max(df, COLS_CLASSIFIEURS, niveau=4)
    X["unanimite"] = (X["nb_accords_max"] == 4).astype(int)

    # 3. Scores de qualite des classifieurs
    X["lcs_distance"] = df["lcs_distance"]
    X["rag_confidence"] = df["rag_confidence"]
    X["ragann_confidence"] = df["ragann_confidence"]
    X["ttc_conf_1"] = df["ttc_conf_1"]

    # 4. Contexte
    X["budget"] = df["budget"]
    X["shop_type_name"] = df["shop_type_name"]
    X["source"] = df["source"]

    return X


def imputer_valeurs_manquantes(X):
    """
    Traite les NaN pour les colonnes dont la sentinelle est une constante
    connue a priori (donc sans risque de fuite train/test) :
    - lcs_distance : 1.5 (distance normalisee bornee a 1 -> hors intervalle = "tres loin")
    - confidences : -1 (sentinelle hors intervalle [0, 1])
    - categorielles : "INCONNU"

    `budget` (mediane, dependante des donnees) n'est PAS imputee ici : elle est geree
    par un SimpleImputer dans le pipeline (cf. modeling.construire_pipeline), fitte
    uniquement sur le train pour eviter toute fuite train/test.
    """
    X = X.copy()

    X["lcs_distance"] = X["lcs_distance"].fillna(1.5)

    for c in ["rag_confidence", "ragann_confidence", "ttc_conf_1"]:
        X[c] = X[c].fillna(-1)

    for c in COLS_CATEGORIELLES:
        X[c] = X[c].fillna("INCONNU").astype(str)

    return X


def preparer_donnees(df, y_pred_baseline, col_vrai, niveau=4):
    """
    Pipeline complet : cible + features + imputation.

    Returns
    -------
    X : pd.DataFrame des features pretes a etre encodees
    y : pd.Series de la cible binaire
    """
    y = construire_cible(df, y_pred_baseline, col_vrai, niveau=niveau)
    X = construire_features(df, y_pred_baseline)
    X = imputer_valeurs_manquantes(X)
    print(f"Features : {X.shape[1]} colonnes ({len(COLS_CATEGORIELLES)} categorielles, {len(COLS_NUMERIQUES)} numeriques)")
    print(f"Cible : {y.value_counts().to_dict()} (taux positif = {y.mean():.1%})")
    return X, y

