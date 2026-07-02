"""
Baseline pour la codification COICOP.

Règle :
- on calcule le code majoritaire parmi les classifieurs votants (au niveau demandé)
- en cas d'égalité, on suit TTC
- si aucun classifieur n'a prédit, on prend TTC en fallback
- l'évaluation compare le code choisi au vrai code, au niveau demandé
"""
import pandas as pd
from collections import Counter
from src.stats_accord import tronquer_niveau


def baseline_majorite_ttc(df, cols_votants, col_ttc, niveau=4):
    """
    Pour chaque ligne, choisit un code COICOP :
      - code majoritaire parmi les classifieurs votants (au niveau `niveau`)
      - en cas d'égalité (plusieurs codes ex aequo), on prend TTC
      - si aucun vote possible (tous NaN), on prend TTC

    Parameters
    ----------
    df : pd.DataFrame
    cols_votants : list[str]
        Colonnes des classifieurs votants (TTC peut être inclus).
    col_ttc : str
        Colonne du classifieur TTC, utilisée en cas d'égalité.
    niveau : int
        Niveau de troncature COICOP (convention niveau N = N+1 chiffres).

    Returns
    -------
    pd.Series des codes prédits (tronqués au niveau demandé), indexée comme df.
    """
    ttc_tronq = df[col_ttc].map(lambda x: tronquer_niveau(x, niveau))
    votants_tronq = df[cols_votants].apply(
        lambda col: col.map(lambda x: tronquer_niveau(x, niveau))
    )

    def predire(row_votants, ttc_code):
        votes = [v for v in row_votants if pd.notna(v)]
        if not votes:
            return ttc_code
        compteur = Counter(votes)
        max_count = max(compteur.values())
        codes_majoritaires = [code for code, n in compteur.items() if n == max_count]

        if len(codes_majoritaires) > 1:
            return ttc_code
        return codes_majoritaires[0]

    pred = [
        predire(votants_tronq.iloc[i].tolist(), ttc_tronq.iloc[i])
        for i in range(len(df))
    ]
    return pd.Series(pred, index=df.index, name="pred_baseline")


def evaluer_baseline(df, y_pred, col_vrai, niveau=4):
    """
    Évalue une baseline qui a choisi un code COICOP.

    Calcule l'accuracy globale au niveau demandé.

    Parameters
    ----------
    df : pd.DataFrame
        Doit contenir la colonne du vrai code (`col_vrai`).
    y_pred : pd.Series
        Codes prédits (déjà tronqués au niveau demandé).
    col_vrai : str
        Colonne du vrai code COICOP.
    niveau : int
        Niveau de troncature à appliquer au vrai code pour la comparaison.

    Returns
    -------
    float : accuracy.
    """
    vrai_tronq = df[col_vrai].map(lambda x: tronquer_niveau(x, niveau))
    correct = (y_pred == vrai_tronq)
    acc = correct.mean()

    print(f"=== Accuracy baseline (niveau {niveau}) ===")
    print(f"Total observations    : {len(df)}")
    print(f"Prédictions correctes : {int(correct.sum())}")
    print(f"Accuracy              : {acc:.3f}")
    return acc
