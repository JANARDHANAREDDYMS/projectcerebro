from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml_core.data.schema import (
    drop_bad_feature_rows,
    resolve_filter_version,
    validate_epoch_dataframe,
)


def create_spark_session(app_name: str = "ProjectCerebro-ML"):
    from delta import configure_spark_with_delta_pip
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "6g")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.shuffle.partitions", "8")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def read_delta(
    path: str | Path,
    filter_version: str | None = None,
    dataset: str | None = None,
    include_synthetic_rest: bool = True,
    limit: int | None = None,
    validate: bool = True,
    spark=None,
) -> pd.DataFrame:
    delta_path = Path(path)
    if not delta_path.exists():
        raise FileNotFoundError(
            f"Delta path not found: {delta_path}. Download shared delta_lake/ first."
        )

    owned_spark = spark is None
    spark = spark or create_spark_session()
    try:
        sdf = spark.read.format("delta").load(str(delta_path))
        actual_filter = resolve_filter_version(filter_version)
        if actual_filter:
            sdf = sdf.where(sdf.filter_version == actual_filter)
        if dataset:
            sdf = sdf.where(sdf.dataset == dataset)
        if not include_synthetic_rest:
            sdf = sdf.where(~sdf.is_rest_synthetic)
        if limit is not None:
            sdf = sdf.limit(int(limit))
        df = sdf.toPandas()
    finally:
        if owned_spark:
            spark.stop()

    df = drop_bad_feature_rows(df)
    if validate:
        validate_epoch_dataframe(df)
    return df
