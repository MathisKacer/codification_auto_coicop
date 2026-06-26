"""
Gestion de la correspondance hierarchique entre codes de nomenclature COICOP.
La nomenclature est structuree en niveaux separes par des points (ex: "01.1.2.3").
"""
import pandas as pd


def niveau_atteint(code) -> int:
    """Retourne la profondeur reelle d'un code (nombre de segments separes par des points)."""
    if pd.isna(code):
        return 0
    return len(str(code).strip().split("."))


def tronquer_a_niveau(code, niveau: int):
    """Tronque un code aux N premiers segments hierarchiques."""
    if pd.isna(code) or niveau <= 0:
        return None
    segments = str(code).strip().split(".")
    return ".".join(segments[:niveau])


def a_raison_jusqu_a_niveau(code_classifieur, code_verite, niveau_max: int = 4) -> int:
    """
    Verifie si le code d'un classifieur correspond a la verite terrain
    jusqu'au niveau hierarchique cible (plafonne par la profondeur reelle de la verite).

    Parameters
    ----------
    code_classifieur : code propose par un classifieur
    code_verite : code vrai (verite terrain)
    niveau_max : profondeur maximale ciblee (4 par defaut pour ton projet)

    Returns
    -------
    1 si le classifieur a raison jusqu'au niveau cible, 0 sinon.
    """
    profondeur_verite = niveau_atteint(code_verite)
    niveau_cible = min(niveau_max, profondeur_verite)
    if niveau_cible == 0 or pd.isna(code_classifieur):
        return 0
    return int(
        tronquer_a_niveau(code_classifieur, niveau_cible) ==
        tronquer_a_niveau(code_verite, niveau_cible)
    )
