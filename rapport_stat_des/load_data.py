import pandas as pd
import s3fs

ENDPOINT_URL = "https://minio.lab.sspcloud.fr"

# Run utilise par defaut. Rejouable sur un autre run sans toucher a ce
# fichier via la variable d'environnement CHEMIN_S3_STATS_DESCRIPTIVES
# (cf. l'entete de rapport_stats_descriptives.py).
CHEMIN_S3_STATS_DESCRIPTIVES = (
    "s3://projet-budget-famille/data/workflow_runs/2026-06-29/codif-vvkv9"
    "/decide-coicop/predictions.parquet"
)


def charger_donnees(chemin_s3, format=None, **kwargs):
    """
    Charge un fichier depuis MinIO S3 (SSPCloud).

    Parameters
    ----------
    chemin_s3 : str
        Chemin S3 du fichier, au format "bucket/dossier/fichier.ext"
        ou "s3://bucket/dossier/fichier.ext" (le préfixe est retiré).
    format : str, optional
        "csv" ou "parquet". Si None, déduit de l'extension.
    **kwargs :
        Arguments passés à pd.read_csv ou pd.read_parquet.

    Returns
    -------
    pd.DataFrame
    """
    # Retire le préfixe s3:// éventuel
    if chemin_s3.startswith("s3://"):
        chemin_s3 = chemin_s3[len("s3://"):]

    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": ENDPOINT_URL})

    if format is None:
        ext = chemin_s3.rsplit(".", 1)[-1].lower()
        if ext in ("csv", "tsv", "txt"):
            format = "csv"
        elif ext in ("parquet", "pq"):
            format = "parquet"
        else:
            raise ValueError(
                f"Format non reconnu pour l'extension '.{ext}'. "
                "Précise format='csv' ou format='parquet'."
            )

    if format == "csv":
        with fs.open(chemin_s3, "rb") as f:
            df = pd.read_csv(f, **kwargs)
    elif format == "parquet":
        df = pd.read_parquet(chemin_s3, filesystem=fs, **kwargs)
    else:
        raise ValueError(f"Format '{format}' non supporté (csv ou parquet).")

    print(f"[charger_donnees] {chemin_s3} → shape = {df.shape}")
    return df
