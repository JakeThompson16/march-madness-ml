
from sportsdataverse.mbb import load_mbb_team_boxscore
import polars as pl

def sdv_features()->list[str]:
    return [
        "away_win_pct", "3pt_rate"
    ]

def generate_away_win_pct(df: pl.DataFrame)->pl.DataFrame:

    # Add away game win column (1 if away game AND won, 0 otherwise)
    df = df.with_columns(
        pl.when((pl.col("home_away") == "away") & (pl.col("team_winner") == True))
        .then(1)
        .otherwise(0)
        .alias("away_game_won"),

        pl.when(pl.col("home_away") == "away").then(1).otherwise(0).alias("away_game")
    )

    # Calculate away game win pct
    df = df.group_by(["team_id", "season"]).agg(
        (pl.col("away_game_won").sum() / pl.col("away_game").sum())
        .alias("away_win_pct")
    )

    cols = ["team_id", "season", "away_win_pct"]
    return df.select(cols)


def generate_three_point_rate(df: pl.DataFrame)->pl.DataFrame:

    df = df.group_by(["team_id", "season"]).agg(
        (pl.col("three_point_field_goals_attempted").sum() / pl.col("field_goals_attempted").sum())
        .alias("3pt_rate")
    )

    cols = ["team_id", "season", "3pt_rate"]
    return df.select(cols)


def get_sdv_features(seasons: int | list[int])->pl.DataFrame:
    """
    :return: Features extracted from sportsdataverse from seasons
    """
    if isinstance(seasons, int):
        seasons = [seasons]
    sdv = load_mbb_team_boxscore(seasons=seasons)
    sdv = sdv.filter(pl.col("game_type") == 2)

    indicators = ["team_id", "season"]

    df1 = generate_away_win_pct(sdv)
    df2 = generate_three_point_rate(sdv)
    df = df1.join(
        other=df2,
        left_on=["team_id", "season"],
        right_on=["team_id", "season"],
        how="inner"
    )

    cols = indicators + sdv_features()
    return df.select(cols)