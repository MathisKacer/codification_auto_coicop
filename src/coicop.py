"""
Codes COICOP : troncature au niveau hierarchique et utilitaires de correspondance.

Convention de niveau (nombre de chiffres significatifs = niveau + 1) :
    niveau 1 -> "XX"        (division)
    niveau 2 -> "XX.X"      (groupe)
    niveau 3 -> "XX.X.X"    (classe)
    niveau 4 -> "XX.X.X.X"  (sous-classe)
"""
import pandas as pd


def tronquer_niveau(code, niveau=4):
    """
    Tronque un code COICOP au niveau demandé.

    Gère les NaN et les sentinels du preprocessing
    ("AUCUNE_SUGGESTION", "NON_CODABLE") qui sont préservés tels quels.
    """
    if pd.isna(code):
        return code
    s = str(code)
    if s in ("AUCUNE_SUGGESTION", "NON_CODABLE"):
        return s

    n_chiffres_cible = niveau + 1

    chiffres = 0
    out = []
    for c in s:
        if c.isdigit():
            if chiffres == n_chiffres_cible:
                break
            chiffres += 1
            out.append(c)
        else:
            if chiffres < n_chiffres_cible:
                out.append(c)
    return "".join(out)


def niveau_atteint(code) -> int:
    """Retourne la profondeur reelle d'un code (nombre de segments separes par des points)."""
    if pd.isna(code):
        return 0
    return len(str(code).strip().split("."))


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
        tronquer_niveau(code_classifieur, niveau_cible) ==
        tronquer_niveau(code_verite, niveau_cible)
    )
