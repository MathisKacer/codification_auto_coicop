"""
Construction de la cible (y) pour la formulation B :
designer le 'gagnant' parmi les classifieurs sur chaque produit.

Les ex-aequo sont regroupes dans une classe 'plusieurs_corrects' pour eviter
tout biais d'ordre artificiel dans l'apprentissage du meta-modele.
"""
import pandas as pd


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


# Mapping pour reconvertir une prediction de gagnant en code final
MAPPING_GAGNANT_VERS_COLONNE = {
    "LCS": "lcs_code",
    "RAG": "rag_code",
    "RAGANN": "ragann_code",
    "TTC": "ttc_code_1",
}


def gagnant_vers_code(gagnant_predit: str, row: pd.Series) -> str:
    """
    Reconvertit une prediction de gagnant en code final.
    - 'aucun' -> PREDICTION_IMPOSSIBLE
    - 'plusieurs_corrects' -> code du premier classifieur correct trouve
      (tous sont corrects jusqu'au niveau cible, donc l'ordre n'a pas d'impact
       sur l'evaluation a ce niveau)
    - sinon -> code du classifieur designe
    """
    if gagnant_predit == "aucun":
        return "PREDICTION_IMPOSSIBLE"

    if gagnant_predit == "plusieurs_corrects":
        for col_raison, col_code in [
            ("lcs_raison", "lcs_code"),
            ("rag_raison", "rag_code"),
            ("ragann_raison", "ragann_code"),
            ("ttc_raison", "ttc_code_1"),
        ]:
            if row[col_raison] == 1:
                return row[col_code]
        return "PREDICTION_IMPOSSIBLE"

    return row[MAPPING_GAGNANT_VERS_COLONNE[gagnant_predit]]