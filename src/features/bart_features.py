
from src.util.bart_torvik import import_bart
import polars as pl


def bart_features()->list[str]:
    return [
        "adj_eff", "adj_off", "adj_def", "adj_tempo",
        "off_efg", "off_tor", "orb_pct", "off_ftr",
        "seed", "sos"
    ]


def get_bart_features(seasons: int | list[int])->pl.DataFrame:

    # non feature cols to be returned
    indicators: list[str] = ["team", "season"]

    df: pl.DataFrame = import_bart(seasons)

    # Assign proper column names
    df = df.with_columns(

        pl.col("barthag").alias("adj_eff"),
        pl.col("adj_o").alias("adj_off"),
        pl.col("adj_d").alias("adj_def"),
        pl.col("adj_t").alias("adj_tempo"),
        pl.col("off_to").alias("off_tor"),
        pl.col("off_or").alias("orb_pct"),
        pl.col("ov_cur_sos").alias("sos")
    )

    cols: list[str] = indicators + bart_features()
    return df.select(*cols)