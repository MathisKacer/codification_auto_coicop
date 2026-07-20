"""
Random Forest binaire : predire si la baseline (majorite + TTC arbitre)
donne le bon code au niveau 4.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_validate, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from scipy.stats import randint

from src.baseline.preprocessing import COLS_CATEGORIELLES, COLS_NUMERIQUES


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

    cols_num_sans_budget = [c for c in COLS_NUMERIQUES if c != "budget"]
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ), COLS_CATEGORIELLES),
            ("num", "passthrough", cols_num_sans_budget),
            ("budget", SimpleImputer(strategy="median"), ["budget"]),
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


def entrainer_evaluer_cv(X, y, n_splits=5, random_state=42, **rf_kwargs):
    """
    Validation croisee stratifiee : reentraine le pipeline complet (preprocessing
    inclus) sur chaque fold, pour une estimation plus robuste que le split unique
    de `entrainer_evaluer`.

    Ne renvoie pas d'importances de features : il n'y a pas un seul modele final
    mais un par fold. Pour les importances, utiliser `entrainer_evaluer`.

    Returns
    -------
    dict : {scores, y_pred, accuracy_mean, accuracy_std, auc_mean, auc_std}
    """
    pipe = construire_pipeline(random_state=random_state, **rf_kwargs)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores = cross_validate(
        pipe, X, y, cv=cv,
        scoring=["accuracy", "roc_auc"],
        return_train_score=False,
    )
    acc_mean, acc_std = scores["test_accuracy"].mean(), scores["test_accuracy"].std()
    auc_mean, auc_std = scores["test_roc_auc"].mean(), scores["test_roc_auc"].std()

    print(f"=== RF binaire : validation croisee ({n_splits} folds) ===")
    print(f"Accuracy : {acc_mean:.3f} ± {acc_std:.3f}")
    print(f"ROC AUC  : {auc_mean:.3f} ± {auc_std:.3f}")

    # Predictions out-of-fold : chaque ligne est predite par le fold ou elle etait en test
    y_pred = cross_val_predict(pipe, X, y, cv=cv)
    cm = confusion_matrix(y, y_pred)

    print(f"\nMatrice de confusion out-of-fold (lignes = vrai, colonnes = predit) :")
    print(pd.DataFrame(
        cm,
        index=["vrai=baseline_fausse (0)", "vrai=baseline_correcte (1)"],
        columns=["pred=0", "pred=1"],
    ).to_string())

    print("\nRapport detaille out-of-fold :")
    print(classification_report(
        y, y_pred,
        target_names=["baseline_fausse", "baseline_correcte"],
        digits=3,
    ))

    return {
        "scores": scores,
        "y_pred": y_pred,
        "accuracy_mean": acc_mean,
        "accuracy_std": acc_std,
        "auc_mean": auc_mean,
        "auc_std": auc_std,
    }


def _average_precision_fausse(estimator, X, y):
    """
    Scorer : aire sous la courbe precision/rappel pour la classe "baseline_fausse"
    (0), la classe qui compte operationnellement (rater une baseline fausse = un
    faux positif silencieux sur "correcte", pas detecte, jamais envoye au LLM).
    Contrairement au ROC AUC, sensible au fait que "fausse" est la classe
    minoritaire (~23%) et directement alignee avec le seuil variable utilise
    en aval (`courbe_precision_rappel`), plutot qu'un seuil fixe a 0.5.
    """
    proba_correcte = estimator.predict_proba(X)[:, 1]
    return average_precision_score(y, 1 - proba_correcte, pos_label=0)


def tuner_hyperparametres(
    X, y, test_size=0.2, random_state=42, n_iter=60, n_splits=5,
    scoring=_average_precision_fausse,
):
    """
    RandomizedSearchCV sur les hyperparametres du RF, fitte uniquement sur le train
    (le meme split que `entrainer_evaluer`, memes test_size/random_state ->
    resultats directement comparables) pour ne pas biaiser l'evaluation finale
    sur le test.

    Par defaut, optimise `_average_precision_fausse` (aire sous la courbe
    precision/rappel de la classe "baseline_fausse") plutot que le ROC AUC :
    ce dernier est une metrique de classement globale insensible au
    desequilibre des classes, et n'a donc aucune raison de privilegier les
    hyperparametres qui detectent le mieux les vraies erreurs de baseline.
    Passer `scoring="roc_auc"` pour retrouver l'ancien comportement.

    Returns
    -------
    dict : {search, best_params, best_score_cv, pipeline, X_train, X_test, y_train,
    y_test, y_pred, y_proba, importances, accuracy, auc}
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state,
    )

    pipe = construire_pipeline(random_state=random_state)
    param_distributions = {
        "rf__n_estimators": randint(100, 800),
        "rf__max_depth": [None, 5, 10, 20, 30],
        "rf__min_samples_leaf": randint(1, 20),
        "rf__max_features": ["sqrt", "log2", None],
        "rf__class_weight": ["balanced", "balanced_subsample", None],
    }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        pipe, param_distributions,
        n_iter=n_iter, scoring=scoring, cv=cv,
        random_state=random_state, n_jobs=-1,
    )
    search.fit(X_train, y_train)

    nom_scoring = scoring if isinstance(scoring, str) else getattr(scoring, "__name__", "custom")
    print(f"=== RF binaire : tuning des hyperparametres ({n_iter} tirages, {n_splits} folds) ===")
    print(f"Meilleur score CV ({nom_scoring}) : {search.best_score_:.3f}")
    print(f"Meilleurs hyperparametres : {search.best_params_}")

    best_pipe = search.best_estimator_
    y_pred = best_pipe.predict(X_test)
    y_proba = best_pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nAccuracy (test) : {acc:.3f}")
    print(f"ROC AUC (test)  : {auc:.3f}")
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

    rf = best_pipe.named_steps["rf"]
    noms_features = COLS_CATEGORIELLES + COLS_NUMERIQUES
    importances = pd.Series(
        rf.feature_importances_, index=noms_features
    ).sort_values(ascending=False)

    return {
        "search": search,
        "best_params": search.best_params_,
        "best_score_cv": search.best_score_,
        "pipeline": best_pipe,
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


def courbe_roc(courbes):
    """
    Courbe(s) ROC superposees (P(baseline_correcte) comme score de classement).

    Parameters
    ----------
    courbes : dict {nom_modele: (y_test, y_proba)}
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    for nom, (y_test, y_proba) in courbes.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{nom} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Hasard (AUC = 0.5)")
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs (rappel)")
    ax.set_title("Courbe ROC")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.show()


def courbe_precision_rappel(y_test, y_proba, seuils=(0.99, 0.95, 0.90, 0.80, 0.5)):
    """
    Compromis precision/rappel sur les deux classes selon le seuil applique a
    y_proba = P(baseline_correcte).

    Regle : on envoie au LLM (predit 0, 'fausse') quand y_proba < seuil. Le pipeline
    (`predict`) utilise implicitement seuil=0.5. Monter le seuil augmente le
    rappel sur 'fausse' (moins d'erreurs manquees) au prix de plus d'envois au LLM.

    Deux lectures possibles du tableau retourne :
    - precision/recall_fausse : parmi ce qu'on envoie au LLM, part vraiment fausse
      (precision), et part des vraies erreurs effectivement captees (recall).
    - precision/recall_correcte : parmi ce a quoi on fait confiance, part vraiment
      correcte (precision = 1 - taux d'erreur residuelle -> LA metrique a suivre si
      on veut etre sur de ce qu'on accepte comme correct), et part des vraies baselines
      correctes qu'on n'envoie pas au LLM inutilement (recall).

    Returns
    -------
    pd.DataFrame indexe par seuil, avec effectifs (n) et taux pour les deux classes.
    """
    precision_c0, recall_c0, _ = precision_recall_curve(y_test, 1 - y_proba, pos_label=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall_c0[:-1], precision_c0[:-1])
    ax.set_xlabel("Rappel (baseline_fausse detectee)")
    ax.set_ylabel("Precision (parmi les envois au LLM)")
    ax.set_title("Precision / rappel — detection des erreurs de baseline")
    ax.grid(alpha=0.3)
    plt.show()

    y_test = pd.Series(y_test).reset_index(drop=True)
    y_proba = pd.Series(y_proba).reset_index(drop=True)
    vrai0 = (y_test == 0)

    lignes = []
    for seuil in seuils:
        envoi_llm = y_proba < seuil   # predit 0 : "fausse", a verifier
        confiance = ~envoi_llm        # predit 1 : "correcte", on fait confiance

        tp = int((envoi_llm & vrai0).sum())    # envoi au LLM justifie
        fp = int((envoi_llm & ~vrai0).sum())   # envoi au LLM inutile
        fn = int((confiance & vrai0).sum())    # erreur manquee (risque)
        tn = int((confiance & ~vrai0).sum())   # confiance justifiee

        lignes.append({
            "seuil": seuil,
            "n_envoi_llm": tp + fp,
            "n_confiance": tn + fn,
            "envois_llm_justifies": tp,
            "envois_llm_inutiles": fp,
            "erreurs_manquees": fn,
            "confiances_justifiees": tn,
            "precision_fausse": tp / (tp + fp) if (tp + fp) else float("nan"),
            "recall_fausse": tp / (tp + fn) if (tp + fn) else float("nan"),
            "precision_correcte": tn / (tn + fn) if (tn + fn) else float("nan"),
            "recall_correcte": tn / (tn + fp) if (tn + fp) else float("nan"),
        })
    df_seuils = pd.DataFrame(lignes).set_index("seuil")
    print(df_seuils.round(3).to_string())
    return df_seuils
