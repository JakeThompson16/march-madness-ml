
from sportsdataverse.mbb import load_mbb_team_boxscore
import polars as pl

def sdv_features()->list[str]:
    return [
        "away_win_pct"
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


def get_sdv_features(seasons: int | list[int])->pl.DataFrame:
    """
    :return: Features extracted from sportsdataverse from seasons
    """
    if isinstance(seasons, int):
        seasons = [seasons]

    # Cols from load_mbb_boxscore vary year to year, only keep intersecting cols
    frames: list[pl.DataFrame] = []
    col_sets:list[list[str]] = []
    for season in seasons:
        frame = load_mbb_team_boxscore(seasons=[season])
        frame = frame.filter(pl.col("season_type") == 2)
        col_sets.append(frame.columns)
        frames.append(frame)

    common_cols = list(set(col_sets[0]).intersection(*col_sets[1:]))
    frames = [frame.select(common_cols) for frame in frames]

    sdv: pl.DataFrame = pl.concat(frames)
    indicators = ["team_location", "season"]

    df = generate_away_win_pct(sdv)

    cols = indicators + sdv_features()
    return df.select(cols)