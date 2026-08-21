#!/usr/bin/env python3
# ==============================================================================
# EVALUATION -- mesure la performance de la baseline (vote majoritaire, TTC
# arbitre) et du modele de confiance associe (Random Forest : "la baseline
# a-t-elle raison ?") sur des donnees dont le vrai code COICOP est CONNU,
# puis sauvegarde un modele final -- reentraine sur 100% des donnees
# labellisees -- pret pour la mise en production (cf. production_baseline.py,
# qui charge ce fichier .joblib).
#
# A executer : (a) pour mesurer la performance actuelle avant/apres un
# changement (nouveau run labellise...), (b) pour rafraichir le modele de
# production avec de nouvelles donnees labellisees.
#
# Usage : python evaluation_baseline.py   (depuis le dossier baseline_final/)
# ==============================================================================
from datetime import date

import joblib
import pandas as pd
import s3fs
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from fonctions_baseline import (
    COL_TTC,
    COLS_CATEGORIELLES,
    COLS_CLASSIFIEURS,
    COLS_NUMERIQUES,
    baseline_majorite_ttc,
    construire_features,
    construire_pipeline,
    imputer_valeurs_manquantes,
    tronquer_niveau,
)

# --- Parametres --------------------------------------------------------------
NIVEAU = 4
SEED_SPLIT = 42

BUCKET_S3 = "projet-budget-famille"
ENDPOINT_S3 = "https://minio.lab.sspcloud.fr"

# Runs labellises disponibles (vrai code connu) -- ajouter ici toute nouvelle
# donnee labellisee au fur et a mesure qu'elle devient disponible, puis
# relancer ce script pour rafraichir le modele de production. Memes runs que
# ceux utilises pour SIRUS (sirus_final/evaluation_sirus.R), pour des
# resultats directement comparables entre les deux modeles.
#
# Pour l'instant, entrainement sur un seul run -- les deux autres restent
# commentes ci-dessous, pour pouvoir tester `production_baseline.py` sur un
# run que le modele n'a jamais vu (generalisation), plutot que sur un
# echantillon issu du meme run que l'entrainement.
RUNS_LABELLISES = [
    f"{BUCKET_S3}/data/workflow_runs/2026-06-29/codif-vvkv9/decide-coicop/predictions.parquet",
    # f"{BUCKET_S3}/data/workflow_runs/2026-07-23/codif-8m8jn/decide-coicop/predictions.parquet",
    # f"{BUCKET_S3}/data/workflow_runs/2026-07-30/codif-x98xl/decide-coicop/predictions.parquet",
]

CHEMIN_MODELE_SORTIE = "modele_baseline_production.joblib"


def charger_run_labellise(objet_s3, fs):
    df = pd.read_parquet(objet_s3, filesystem=fs)
    if "ligne" not in df.columns:
        df["ligne"] = range(len(df))
    if "code" not in df.columns:
        raise ValueError(f"{objet_s3} : pas de colonne `code` (run non labellisé, inattendu ici).")
    # Vrai code : `code_lvl4` si present (calcule en amont depuis 2026-07-30,
    # deja tronque/elague au niveau 4), sinon troncature manuelle avec cascade
    # -- memes conventions que sirus_final (verifiees equivalentes a 100%).
    if "code_lvl4" in df.columns:
        df["vrai_n4"] = df["code_lvl4"]
    else:
        df["vrai_n4"] = df["code"].map(lambda x: tronquer_niveau(x, NIVEAU))
    return df


def main():
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": ENDPOINT_S3})

    print("Chargement des runs labellisés...")
    runs = []
    for i, objet in enumerate(RUNS_LABELLISES):
        df = charger_run_labellise(objet, fs)
        # `ligne` est un identifiant relatif a chaque run -- prefixe avant
        # d'empiler pour ne jamais confondre un produit d'un run avec un
        # produit d'un autre run.
        df["ligne"] = [f"run{i}_{ident}" for ident in df["ligne"]]
        print(f"  {objet} : {len(df)} lignes")
        runs.append(df)

    df_total = pd.concat(runs, ignore_index=True)

    # Les runs doivent porter des observations disjointes -- sinon les
    # accuracy/CV ci-dessous seraient biaisées (même produit compté deux
    # fois, ou fuite train/test si le même produit tombe des deux côtés du
    # split). `id` est un UUID stable par produit, indépendant de `ligne`
    # (qui n'est qu'un indice relatif au run) : vérifié explicitement plutôt
    # que supposé.
    if "id" in df_total.columns and df_total["id"].duplicated().any():
        n_dup = int(df_total["id"].duplicated().sum())
        raise ValueError(
            f"{n_dup} id(s) dupliqué(s) entre les runs labellisés -- "
            "au moins un produit apparaît dans plusieurs runs, à investiguer "
            "avant de poursuivre (les runs ne sont plus disjoints)."
        )
    print(f"\nTotal : {len(df_total)} lignes ({len(runs)} run(s))")

    # --- Baseline seule --------------------------------------------------------
    y_pred_baseline = baseline_majorite_ttc(df_total, COLS_CLASSIFIEURS, COL_TTC, niveau=NIVEAU)
    n_sans_code = y_pred_baseline.isna().sum()
    if n_sans_code:
        print(
            "Produits sans aucun code exploitable (aucun classifieur, TTC "
            f"compris, n'a rien proposé) : {n_sans_code}"
        )

    correct_baseline = y_pred_baseline == df_total["vrai_n4"]
    acc_baseline = correct_baseline.mean()
    print(f"\n=== Accuracy de la baseline seule (niveau {NIVEAU}) : {acc_baseline:.1%} ===")

    # --- Preparation des features pour le modele de confiance -------------------
    X = construire_features(df_total, y_pred_baseline)
    X = imputer_valeurs_manquantes(X)
    y = correct_baseline.astype(int)
    print(
        f"Features : {X.shape[1]} colonnes ({len(COLS_CATEGORIELLES)} catégorielles, "
        f"{len(COLS_NUMERIQUES)} numériques)"
    )
    print(f"Cible (baseline correcte) : taux positif = {y.mean():.1%}")

    # --- Split 80/20 (par ligne) pour ESTIMER la performance ---------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED_SPLIT,
    )
    pipe_eval = construire_pipeline(random_state=SEED_SPLIT)
    pipe_eval.fit(X_train, y_train)

    y_pred = pipe_eval.predict(X_test)
    y_proba = pipe_eval.predict_proba(X_test)[:, 1]
    acc_rf = accuracy_score(y_test, y_pred)
    auc_rf = roc_auc_score(y_test, y_proba)

    print("\n=== Modèle de confiance (RF), test 20% (tous runs labellisés confondus) ===")
    print(f"Accuracy : {acc_rf:.3f}")
    print(f"ROC AUC  : {auc_rf:.3f}")
    print("\nMatrice de confusion (lignes = vrai, colonnes = prédit) :")
    print(pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        index=["vrai=baseline_fausse (0)", "vrai=baseline_correcte (1)"],
        columns=["pred=0", "pred=1"],
    ).to_string())
    print("\nRapport détaillé :")
    print(classification_report(
        y_test, y_pred, target_names=["baseline_fausse", "baseline_correcte"], digits=3,
    ))

    rf = pipe_eval.named_steps["rf"]
    importances = pd.Series(
        rf.feature_importances_, index=COLS_CATEGORIELLES + COLS_NUMERIQUES,
    ).sort_values(ascending=False)
    print("\nImportances des features :")
    print(importances.to_string())

    # --- Calibration : sert a choisir/verifier SEUIL_DECISION dans
    # production_baseline.py -- a revérifier après chaque réentraînement.
    tranches = pd.cut(pd.Series(y_proba), bins=[i / 10 for i in range(11)], include_lowest=True)
    calib = (
        pd.DataFrame({"tranche": tranches, "correct": y_test.values})
        .groupby("tranche", observed=True)["correct"]
        .agg(n="count", pct_correct="mean")
    )
    print(
        "\n=== Calibration (test 20%) : % de baseline réellement correcte, "
        "par tranche de confiance RF ==="
    )
    calib_affichable = calib.assign(
        pct_correct=lambda d: d["pct_correct"].map(lambda x: f"{x:.1%}")
    )
    print(calib_affichable.to_string())

    # --- Modele final : reentraine sur 100% des donnees labellisees -------------
    # L'evaluation ci-dessus (split 80/20) sert a ESTIMER la performance. Une
    # fois cette estimation obtenue, le modele livre en production utilise
    # tout ce qui est disponible.
    print("\nRéentraînement du modèle final sur 100% des données labellisées...")
    pipe_final = construire_pipeline(random_state=SEED_SPLIT)
    pipe_final.fit(X, y)

    bundle = {
        "pipeline": pipe_final,
        "cols_categorielles": COLS_CATEGORIELLES,
        "cols_numeriques": COLS_NUMERIQUES,
        "cols_classifieurs": COLS_CLASSIFIEURS,
        "col_ttc": COL_TTC,
        "niveau": NIVEAU,
        "accuracy_estimee": {
            "baseline": float(acc_baseline),
            "rf_accuracy": float(acc_rf),
            "rf_auc": float(auc_rf),
        },
        # Probas et vraies étiquettes du test 20% : permet à
        # production_baseline.py d'estimer, pour N'IMPORTE QUEL seuil choisi
        # (pas seulement les tranches ci-dessus), le volume et la fiabilité
        # attendus -- sans jamais voir le vrai code du nouveau run à coder.
        "calibration_test": {
            "proba": [float(p) for p in y_proba],
            "correct": [int(c) for c in y_test.values],
        },
        "runs_entrainement": RUNS_LABELLISES,
        "date_entrainement": date.today().isoformat(),
    }
    # compress=3 : le RF (300 arbres) serialise brut depasse 35 Mo, gene le
    # partage/transfert du modele. Compression ~4x sans impact sur le
    # chargement (joblib.load gere la decompression de facon transparente).
    joblib.dump(bundle, CHEMIN_MODELE_SORTIE, compress=3)

    print(f"\nModèle de production sauvegardé : {CHEMIN_MODELE_SORTIE}")
    print(
        "(pipeline complet, features, accuracy estimée et métadonnées inclus "
        "dans le bundle -- utilisé par production_baseline.py)"
    )


if __name__ == "__main__":
    main()
