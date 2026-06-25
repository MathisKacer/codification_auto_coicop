import pandas as pd
import s3fs

fs = s3fs.S3FileSystem(
    client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"}
)

chemin_s3 = "s3://projet-budget-famille/data/workflow_runs/2026-06-18/codif-lvqfj/decide-coicop/predictions.parquet"

df = pd.read_parquet(chemin_s3, filesystem=fs)
