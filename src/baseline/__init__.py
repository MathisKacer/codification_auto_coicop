"""
Pipeline baseline actif : vote majoritaire entre classifieurs, arbitrage TTC
en cas d'égalité, et modèle binaire pour prédire si la baseline a raison.
"""
from src.baseline.ttc import baseline_majorite_ttc, evaluer_baseline
from src.baseline.preprocessing import preparer_donnees
from src.baseline.modeling import (
    construire_pipeline, entrainer_evaluer, entrainer_evaluer_cv, tuner_hyperparametres,
    courbe_precision_rappel, courbe_roc,
)

__all__ = [
    "baseline_majorite_ttc",
    "evaluer_baseline",
    "preparer_donnees",
    "construire_pipeline",
    "entrainer_evaluer",
    "entrainer_evaluer_cv",
    "tuner_hyperparametres",
    "courbe_precision_rappel",
    "courbe_roc",
]
