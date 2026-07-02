"""
Construction de la cible (y) pour la formulation B :
designer le 'gagnant' parmi les classifieurs sur chaque produit.

Les ex-aequo sont regroupes dans une classe 'plusieurs_corrects' pour eviter
tout biais d'ordre artificiel dans l'apprentissage du meta-modele.
"""
from collections import Counter

import pandas as pd

from src.coicop import tronquer_niveau


def determiner_gagnant(row: pd.Series) -> str:
    """
    Designe le classifieur 'gagnant' pour une ligne :
    - "aucun" si aucun classifieur n'a raison
    - le nom du classifieur s'il est le seul a avoir raison
    - "plusieurs_corrects" si plusieurs sont simultanement corrects
    """
    candidats = []
    if row["lcs_raison"] == 1:
        candidats.append("LCS")
    if row["rag_raison"] == 1:
        candidats.append("RAG")
    if row["ragann_raison"] == 1:
        candidats.append("RAGANN")
    if row["ttc_raison"] == 1:
        candidats.append("TTC")

    if len(candidats) == 0:
        return "aucun"
    if len(candidats) == 1:
        return candidats[0]
    return "plusieurs_corrects"


def ajouter_gagnant(df_valide: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la colonne 'gagnant' a partir des colonnes *_raison deja calculees."""
    df_valide["gagnant"] = df_valide.apply(determiner_gagnant, axis=1)
    print(df_valide["gagnant"].value_counts())
    return df_valide


MAPPING_GAGNANT_VERS_COLONNE = {
    "LCS": "lcs_code",
    "RAG": "rag_code",
    "RAGANN": "ragann_code",
    "TTC": "ttc_code_1",
}


def gagnant_vers_code(gagnant_predit: str, row: pd.Series, niveau_max: int = 4) -> str:
    """
    Reconvertit une prediction de gagnant en code final, SANS utiliser la verite terrain.

    - 'aucun' -> PREDICTION_IMPOSSIBLE
    - 'plusieurs_corrects' -> vote majoritaire entre les 4 classifieurs au niveau cible
      (en cas d'egalite, on prend le premier rencontre par Counter, qui depend
       de l'ordre d'apparition dans la liste)
    - sinon -> code du classifieur designe
    """
    if gagnant_predit == "aucun":
        return "PREDICTION_IMPOSSIBLE"

    if gagnant_predit == "plusieurs_corrects":
        codes = [
            tronquer_niveau(row["lcs_code"], niveau_max),
            tronquer_niveau(row["rag_code"], niveau_max),
            tronquer_niveau(row["ragann_code"], niveau_max),
            tronquer_niveau(row["ttc_code_1"], niveau_max),
        ]
        codes_valides = [c for c in codes if c is not None and pd.notna(c)]
        if not codes_valides:
            return "PREDICTION_IMPOSSIBLE"
        return Counter(codes_valides).most_common(1)[0][0]

    return row[MAPPING_GAGNANT_VERS_COLONNE[gagnant_predit]]