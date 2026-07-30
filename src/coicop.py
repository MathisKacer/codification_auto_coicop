"""
Codes COICOP : troncature au niveau hierarchique et utilitaires de correspondance.

Convention de niveau (nombre de chiffres significatifs = niveau + 1) :
    niveau 1 -> "XX"        (division)
    niveau 2 -> "XX.X"      (groupe)
    niveau 3 -> "XX.X.X"    (classe)
    niveau 4 -> "XX.X.X.X"  (sous-classe)
"""
import pandas as pd


DIVISIONS_COICOP = {
    "01": "Produits alimentaires et boissons non alcoolisées",
    "02": "Boissons alcoolisées, tabac et stupéfiants",
    "03": "Articles d'habillement et chaussures",
    "04": "Logement, eau, gaz, électricité et autres combustibles",
    "05": "Meubles, articles de ménage et entretien courant du foyer",
    "06": "Santé",
    "07": "Transports",
    "08": "Information et communication",
    "09": "Loisirs, sport et culture",
    "10": "Services de l'enseignement",
    "11": "Services de restauration et d'hébergement",
    "12": "Assurance et services financiers",
    "13": "Soins corporels, protection sociale et biens et services divers",
    "98": "Produit indéterminé/illisible",
    "99": "Hors COICOP (dons, impôts, opérations bancaires...)",
}


def libelle_division(code):
    """
    Renvoie "Libellé (XX)" pour un code de division COICOP à 2 chiffres.
    Renvoie `code` inchangé s'il n'est pas reconnu (ex. "TOTAL", NaN).
    """
    if pd.isna(code):
        return code
    code = str(code)
    libelle = DIVISIONS_COICOP.get(code)
    return f"{libelle} ({code})" if libelle else code


def tronquer_niveau(code, niveau=4):
    """
    Tronque un code COICOP au niveau demandé.

    Gère les NaN et les sentinels du preprocessing
    ("AUCUNE_SUGGESTION", "NON_CODABLE") qui sont préservés tels quels.

    Un "0" terminal ne correspond a aucune vraie sous-categorie (convention
    COICOP "pas de sous-classe/groupe plus precise", ex. 13.9.0 == 13.9) :
    il est retire en cascade, aussi bien sur les codes predits que sur le
    vrai code, pour que les deux soient compares sur la meme base.
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
    resultat = "".join(out)
    while resultat.endswith(".0"):
        resultat = resultat[:-2]
    return resultat


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
