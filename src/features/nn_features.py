
import polars as pl

from src.features.sportsdataverse_features import get_raw_boxscores

NN_TEAM_FEATURES: list[str] = [
    "rolling_o_rate", "rolling_d_rate", "rolling_tempo", "rolling_win_rate",
    "rolling_to_rate", "rolling_orb_rate", "rolling_efg_pct", "rolling_sos"
]

GAME_LEVEL_FEATURES: list[str] = [
    "o_rate", "d_rate", "possessions", "efg_pct", "win_rate",
    "to_rate", "orb_rate"
]

_NN_FEATURES_RAW: list[str] = [
    "assists", "turnovers", "three_point_field_goals_attempted", "field_goal_pct",
    "turnover_points", "free_throws_made", "free_throw_pct", "field_goals_attempted",
    "three_point_field_goal_pct", "offensive_rebounds", "free_throws_attempted",
    "team_score", "opponent_team_score", "team_winner", "field_goals_made",
    "three_point_field_goals_made"
]

IDENTIFIERS: list[str] = [
    "team_location", "opponent_team_location", "season", "game_date"
]

def generate_nn_feature_df(seasons: int | list[int])->pl.DataFrame:
    """
    :param seasons: Seasons to get feature df for
    :return: Team level feature data frame for seasons
    """
    df = get_raw_boxscores(seasons)
    df = df.sort((['team_location', 'season', 'game_date']))
    df = df.select(_NN_FEATURES_RAW + IDENTIFIERS)

    # prep
    df = df.with_columns(
        # possessions
        (pl.col("field_goals_attempted") - pl.col("offensive_rebounds")
         + pl.col("turnovers") + pl.col("free_throws_attempted") * 0.44)
        .alias("possessions"),

        # fg missed
        (pl.col("field_goals_attempted") - pl.col("field_goals_made"))
        .alias("fg_missed")
    )

    # prep
    df = df.with_columns(
        # Offensive rating
        (pl.col("team_score") / pl.col("possessions"))
         .alias("o_rate"),

         # Defensive rating
        (pl.col("opponent_team_score") / pl.col("possessions"))
        .alias("d_rate"),

        # off rebound rate
        (pl.col("offensive_rebounds") / pl.col("fg_missed"))
        .alias("orb_rate"),

        # effective field goal pct
        ((pl.col("field_goals_made") + pl.col("three_point_field_goals_made") * 0.5) / pl.col("field_goals_attempted"))
        .alias("efg_pct"),

        # turnover rate
        (pl.col("turnovers") / pl.col("possessions"))
        .alias("to_rate"),

        # team won
        pl.col("team_winner").cast(pl.Float64)
        .alias("win_rate")
    )

    # Finish game level data prep
    cols = IDENTIFIERS + GAME_LEVEL_FEATURES
    df = df.select(cols)

    # generate team level training data
    df = df.with_columns([
        (pl.col(f).shift(1).cum_sum().over(["team_location", "season"]) /
         pl.col(f).shift(1).is_not_null().cast(pl.Float64).cum_sum().over(["team_location", "season"]))
        .alias(f"rolling_{f}")
        for f in GAME_LEVEL_FEATURES
    ])
    df = df.filter(pl.col("rolling_o_rate").is_not_null())


    opp_strength = df.with_columns(
        pl.col("team_location")
        .alias("opponent_team_location"),

        (pl.col("rolling_o_rate") - pl.col("rolling_d_rate"))
        .alias("rolling_opp_net_rate")
    )

    opp_strength = opp_strength.select([
        "opponent_team_location", "rolling_opp_net_rate", "game_date", "season"
    ])

    df = df.join(
        opp_strength,
        on=["opponent_team_location", "season", "game_date"],
        how="inner"
    )

    df = df.rename({
        "rolling_possessions" : "rolling_tempo",
        "rolling_opp_net_rate" : "rolling_sos"
    })

    df = df.select(NN_TEAM_FEATURES + IDENTIFIERS)

    return df

def build_nn_matchup_df(seasons: int | list[int]) -> pl.DataFrame:
    if isinstance(seasons, int):
        seasons = [seasons]

    df = generate_nn_feature_df(seasons)

    # Get game outcomes from raw boxscores
    outcomes = (
        get_raw_boxscores(seasons)
        .select(["team_location", "season", "game_date", "team_winner"])
    )
    df = df.join(outcomes, on=["team_location", "season", "game_date"], how="left")

    team_a = df.rename({f: f"a_{f}" for f in NN_TEAM_FEATURES + ["team_winner"]})
    team_b = df.rename({f: f"b_{f}" for f in NN_TEAM_FEATURES})

    matchups = team_a.join(
        team_b,
        left_on=["team_location", "season", "game_date"],
        right_on=["opponent_team_location", "season", "game_date"],
        how="inner"
    )

    # remove dupes
    matchups = matchups.filter(
        pl.col("team_location") < pl.col("opponent_team_location")
    )

    return matchups.rename({"a_team_winner": "team_a_won"}).select(
        ["season", "game_date", "team_location", "opponent_team_location", "team_a_won"]
        + [f"a_{f}" for f in NN_TEAM_FEATURES]
        + [f"b_{f}" for f in NN_TEAM_FEATURES]
    )
