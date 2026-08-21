#!/usr/bin/env python3
# ==============================================================================
# PRODUCTION -- applique le modele baseline+RF sauvegarde (cf.
# evaluation_baseline.py) à de NOUVELLES lignes dont le vrai code COICOP est
# INCONNU, pour produire une décision par produit : code automatique (avec
# une fiabilité estimée par la calibration du modèle), ou reprise manuelle /
# LLM-judge.
#
# Ne calcule et n'utilise AUCUN vrai code, même si une colonne `code` existe
# par ailleurs dans le run -- pour mesurer une performance, voir
# evaluation_baseline.py.
#
# Usage : python production_baseline.py <chemin_s3_du_run_a_coder>
#   ex.  : python production_baseline.py \
#            data/workflow_runs/2026-08-15/codif-xxxxx/decide-coicop/predictions.parquet
# ==============================================================================
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import s3fs

from fonctions_baseline import (
    baseline_majorite_ttc,
    construire_features,
    imputer_valeurs_manquantes,
)

# --- Parametres ----------------------------------------------------------------
BUCKET_S3 = "projet-budget-famille"
ENDPOINT_S3 = "https://minio.lab.sspcloud.fr"
CHEMIN_MODELE = "modele_baseline_production.joblib"

# Seuil de décision sur `proba` (P(baseline correcte) estimée par le RF) :
# au-dessus, on accepte le choix de la baseline (code automatique) ; en
# dessous, reprise manuelle / LLM-judge plutôt que de forcer un choix peu
# fiable. 0.5 est lu sur la table de calibration produite par
# evaluation_baseline.py (test 20%, run `codif-vvkv9`) : à ce seuil, 77.6% du
# volume est codé automatiquement, avec une fiabilité observée de 96.8% sur
# ce sous-ensemble -- À RE-VÉRIFIER à chaque réentraînement du modèle (la
# calibration peut dériver d'un run à l'autre, cf. régression RAG-ANN
# documentée côté SIRUS).
SEUIL_DECISION = 0.3


def identifiant_run(chemin_s3):
    """
    Identifiant unique par run (date + codif-xxxxx), pour ne jamais écraser
    le résultat d'un run précédent (basename() seul collisionnerait, chaque
    run s'appelant "predictions.parquet").
    """
    segments = chemin_s3.split("/")
    idx = [i for i, s in enumerate(segments) if s.startswith("codif-")]
    if idx and idx[0] >= 1:
        return f"{segments[idx[0] - 1]}_{segments[idx[0]]}"
    return re.sub(r"[/.]", "_", chemin_s3.replace(".parquet", ""))


def estimer_fiabilite_au_seuil(bundle, seuil):
    """
    Estime, à partir du test 20% mis de côté par evaluation_baseline.py (pas
    du run à coder -- son vrai code est inconnu), le volume et la fiabilité
    attendus si on règle SEUIL_DECISION à `seuil`. Fonctionne pour n'importe
    quel seuil, pas seulement les tranches de la table de calibration
    affichée par evaluation_baseline.py.
    """
    if "calibration_test" not in bundle:
        return None  # bundle genere par une version anterieure du script
    proba = np.array(bundle["calibration_test"]["proba"])
    correct = np.array(bundle["calibration_test"]["correct"])
    confiant = proba >= seuil
    if not confiant.any():
        return {"volume": 0.0, "fiabilite": float("nan"), "n": 0}
    return {
        "volume": float(confiant.mean()),
        "fiabilite": float(correct[confiant].mean()),
        "n": int(confiant.sum()),
    }


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage : python production_baseline.py <chemin_s3_du_run_a_coder>")
    objet_s3 = sys.argv[1]

    if not Path(CHEMIN_MODELE).exists():
        sys.exit(
            f"Modèle introuvable ({CHEMIN_MODELE}) -- exécuter d'abord evaluation_baseline.py."
        )
    bundle = joblib.load(CHEMIN_MODELE)
    acc = bundle["accuracy_estimee"]
    print(
        f"Modèle chargé (entraîné le {bundle['date_entrainement']} sur "
        f"{len(bundle['runs_entrainement'])} run(s), accuracy baseline ESTIMÉE "
        f"{acc['baseline']:.1%}, RF {acc['rf_accuracy']:.1%} -- cf. evaluation_baseline.py)"
    )

    estimation = estimer_fiabilite_au_seuil(bundle, SEUIL_DECISION)
    if estimation is None:
        print(
            "(Estimation du volume/fiabilité au seuil actuel indisponible -- "
            "modèle généré par une version antérieure d'evaluation_baseline.py, "
            "relancer evaluation_baseline.py pour la régénérer.)"
        )
    else:
        print(
            f"À SEUIL_DECISION = {SEUIL_DECISION} : sur le test de "
            f"evaluation_baseline.py, ça correspondrait à {estimation['volume']:.1%} "
            f"du volume codé automatiquement ({estimation['n']} cas), avec une "
            f"fiabilité estimée de {estimation['fiabilite']:.1%} (donc "
            f"~{1 - estimation['fiabilite']:.1%} de faux positifs attendus parmi "
            "ce volume) -- estimation basée sur le run d'entraînement, pas sur le "
            "run ci-dessous."
        )

    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": ENDPOINT_S3})
    chemin_complet = f"{BUCKET_S3}/{objet_s3}" if not objet_s3.startswith(BUCKET_S3) else objet_s3
    df = pd.read_parquet(chemin_complet, filesystem=fs).reset_index(drop=True)
    if "ligne" not in df.columns:
        df["ligne"] = range(len(df))
    if df["ligne"].duplicated().any():
        sys.exit("Des identifiants `ligne` en double rendraient le résultat ambigu.")
    print(f"Run à coder : {len(df)} lignes")

    # --- Baseline : code candidat par produit (NaN si aucun classifieur, TTC
    # compris, n'a rien proposé) ---
    y_pred_baseline = baseline_majorite_ttc(
        df, bundle["cols_classifieurs"], bundle["col_ttc"], niveau=bundle["niveau"],
    )

    # --- Modèle de confiance : proba que ce choix soit correct ---
    X = construire_features(df, y_pred_baseline)
    X = imputer_valeurs_manquantes(X)
    proba = bundle["pipeline"].predict_proba(X)[:, 1]

    resultats = pd.DataFrame({
        "ligne": df["ligne"].values,
        "code_propose": y_pred_baseline.values,
        "proba": proba,
    })

    # Produits sans aucun candidat exploitable : ne PEUVENT PAS être codés
    # automatiquement -- la proba du RF n'a alors aucun sens (il n'y a pas de
    # code à évaluer), donc explicitement écrasée plutôt que publiée.
    sans_candidat = resultats["code_propose"].isna()
    resultats.loc[sans_candidat, "proba"] = float("nan")

    resultats["decision"] = resultats["proba"].map(
        lambda p: "Code automatique" if p >= SEUIL_DECISION else "Reprise manuelle / LLM"
    )
    resultats.loc[sans_candidat, "decision"] = (
        "Reprise manuelle / LLM (aucun candidat exploitable)"
    )

    lignes_ok = set(resultats["ligne"]) == set(df["ligne"])
    sans_doublon = not resultats["ligne"].duplicated().any()
    assert lignes_ok and sans_doublon, (
        "chaque produit d'entrée doit apparaître exactement une fois en sortie"
    )

    print("\n=== Bilan de la décision ===")
    bilan = resultats.groupby("decision").size().rename("n").reset_index()
    bilan["pct_volume"] = (bilan["n"] / len(resultats) * 100).map(lambda x: f"{x:.1f}%")
    print(bilan.to_string(index=False))

    if sans_candidat.any():
        print(
            f"\n({int(sans_candidat.sum())} produit(s) sans aucun candidat "
            "proposé par les 4 classifieurs.)"
        )

    chemin_sortie = f"{identifiant_run(objet_s3)}_codes_baseline.csv"
    resultats.to_csv(chemin_sortie, index=False)
    print(
        f"\nRésultats écrits dans {chemin_sortie} "
        f"({len(resultats)} lignes, une par produit d'entrée)"
    )


if __name__ == "__main__":
    main()
