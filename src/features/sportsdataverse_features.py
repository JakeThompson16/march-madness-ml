
from sportsdataverse.mbb import load_mbb_team_boxscore
import polars as pl


def get_raw_boxscores(seasons: int | list[int])->pl.DataFrame:
    """
    :param seasons: Seasons to get data from
    :return: Df of game level data
    """
    if isinstance(seasons, int):
        seasons = [seasons]

    frames: list[pl.DataFrame] = []
    cols: list[list[str]] = []
    for season in seasons:
        frame = load_mbb_team_boxscore(seasons=[season])
        cols.append(frame.columns)
        frames.append(frame)

    cols = list(set(cols[0]).intersection(*cols[1:]))
    frames = [frame.select(cols) for frame in frames]

    df: pl.DataFrame = pl.concat(frames)
    return df.filter(pl.col("season_type") == 2)

def sdv_features()->list[str]:
    return [
        "away_win_pct", "last_10_win_pct"
    ]

def generate_away_win_pct(df: pl.DataFrame)->pl.DataFrame:

    # Add away game win column (1 if away game AND won, 0 otherwise)
    df = df.with_columns(
        pl.when((pl.col("team_home_away") == "away") & (pl.col("team_winner") == True))
        .then(1)
        .otherwise(0)
        .alias("away_game_won"),

        pl.when(pl.col("team_home_away") == "away").then(1).otherwise(0).alias("away_game")
    )

    # Calculate away game win pct
    df = df.group_by(["team_location", "season"]).agg(
        (pl.col("away_game_won").sum() / pl.col("away_game").sum())
        .alias("away_win_pct")
    )

    cols = ["team_location", "season", "away_win_pct"]
    return df.select(cols)

def generate_last_10_win_pct(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("game_date_time").cast(pl.Datetime).alias("game_date_time"),
        pl.col("team_winner").cast(pl.Int32).alias("won")
    )

    df = df.sort(["team_location", "season", "game_date_time"])

    df = df.with_columns(
        pl.col("won")
          .rolling_mean(window_size=10)
          .over(["team_location", "season"])
          .alias("last_10_win_pct")
    )

    return df.group_by(["team_location", "season"]).agg(
        pl.col("last_10_win_pct").last()
    ).select(["team_location", "season", "last_10_win_pct"])

def get_sdv_features(raw_boxscores: pl.DataFrame)->pl.DataFrame:
    """
    :return: Features extracted from sportsdataverse from seasons
    """

    indicators = ["team_location", "season"]

    away = generate_away_win_pct(raw_boxscores)
    last_10 = generate_last_10_win_pct(raw_boxscores)

    df = away.join(last_10, on=["team_location", "season"], how="inner")
    cols = indicators + sdv_features()
    return df.select(cols)