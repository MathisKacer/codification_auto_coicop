"""
Random Forest binaire : predire si la baseline (majorite + TTC arbitre)
donne le bon code au niveau 4.
"""
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.preprocessing_baseline import COLS_CATEGORIELLES, COLS_NUMERIQUES


def construire_pipeline(random_state=42, **rf_kwargs):
    """
    Pipeline : OrdinalEncoder sur les categorielles, passthrough sur les numeriques,
    puis RandomForestClassifier.
    """
    defaults = dict(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    defaults.update(rf_kwargs)

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ), COLS_CATEGORIELLES),
            ("num", "passthrough", COLS_NUMERIQUES),
        ]
    )
    return Pipeline([
        ("preprocess", preprocess),
        ("rf", RandomForestClassifier(**defaults)),
    ])


def entrainer_evaluer(X, y, test_size=0.2, random_state=42, **rf_kwargs):
    """
    Split stratifie, entrainement RF, evaluation complete.

    Returns
    -------
    dict : {pipeline, X_train, X_test, y_train, y_test, y_pred, y_proba, importances}
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state,
    )

    pipe = construire_pipeline(random_state=random_state, **rf_kwargs)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"=== RF binaire : la baseline est-elle correcte ? ===")
    print(f"Train : {len(X_train)} lignes | Test : {len(X_test)} lignes")
    print(f"Taux positif dans le train : {y_train.mean():.1%}")
    print(f"\nAccuracy : {acc:.3f}")
    print(f"ROC AUC  : {auc:.3f}")

    print(f"\nMatrice de confusion (lignes = vrai, colonnes = predit) :")
    print(pd.DataFrame(
        cm,
        index=["vrai=baseline_fausse (0)", "vrai=baseline_correcte (1)"],
        columns=["pred=0", "pred=1"],
    ).to_string())

    print("\nRapport detaille :")
    print(classification_report(
        y_test, y_pred,
        target_names=["baseline_fausse", "baseline_correcte"],
        digits=3,
    ))

    # Importances de features
    rf = pipe.named_steps["rf"]
    noms_features = COLS_CATEGORIELLES + COLS_NUMERIQUES
    importances = pd.Series(
        rf.feature_importances_, index=noms_features
    ).sort_values(ascending=False)
    print("\nImportances des features :")
    print(importances.to_string())

    return {
        "pipeline": pipe,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "importances": importances,
        "accuracy": acc,
        "auc": auc,
    }
