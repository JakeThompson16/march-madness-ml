import polars as pl
from src.features.cbbd_features import extract_cbbd_data
from src.features.sportsdataverse_features import get_sdv_features
from data.team_name_map import KAGGLE_TO_CBBD

TOURNEY_RESULTS_PATH = "data/MNCAATourneyDetailedResults.csv"
TOURNEY_SEEDS_PATH = "data/MNCAATourneySeeds.csv"
TEAMS_PATH = "data/MTeams.csv"


def parse_seed(seed_str: str) -> int:
    return int("".join(filter(str.isdigit, seed_str)))


def map_kaggle_name(name: str) -> str | None:
    return KAGGLE_TO_CBBD.get(name, name)


def generate_team_features(seasons: int | list[int]) -> pl.DataFrame:
    if isinstance(seasons, int):
        seasons = [seasons]

    cbbd = extract_cbbd_data(seasons)
    sdv = get_sdv_features(seasons)
    seeds = (
        pl.read_csv("data/MNCAATourneySeeds.csv")
        .filter(pl.col("Season").is_in(seasons))
        .with_columns(
            pl.col("Seed").map_elements(parse_seed, return_dtype=pl.Int32).alias("seed")
        )
        .join(pl.read_csv("data/MTeams.csv").with_columns(
            pl.col("TeamName").map_elements(map_kaggle_name, return_dtype=pl.String).alias("TeamName")
        ).filter(pl.col("TeamName").is_not_null()), on="TeamID", how="left")
        .select(["TeamName", "Season", "seed"])
        .rename({"TeamName": "team", "Season": "season"})
    )

    df = cbbd.join(sdv, left_on=["team", "season"], right_on=["team_location", "season"], how="inner")
    return df.join(seeds, on=["team", "season"], how="left")


def build_matchup_df(seasons: int | list[int]) -> pl.DataFrame:
    if isinstance(seasons, int):
        seasons = [seasons]

    results = pl.read_csv(TOURNEY_RESULTS_PATH).filter(pl.col("Season").is_in(seasons))
    seeds = (
        pl.read_csv(TOURNEY_SEEDS_PATH)
        .filter(pl.col("Season").is_in(seasons))
        .with_columns(pl.col("Seed").map_elements(parse_seed, return_dtype=pl.Int32).alias("seed_num"))
    )

    # Load teams and apply kaggle -> cbbd name mapping, drop defunct teams
    teams = (
        pl.read_csv(TEAMS_PATH)
        .with_columns(
            pl.col("TeamName")
            .map_elements(map_kaggle_name, return_dtype=pl.String)
            .alias("TeamName")
        )
        .filter(pl.col("TeamName").is_not_null())
    )

    # Add seeds for winner and loser
    results = (
        results
        .join(seeds.select(["Season", "TeamID", "seed_num"]),
              left_on=["Season", "WTeamID"], right_on=["Season", "TeamID"], how="left")
        .rename({"seed_num": "w_seed"})
        .join(seeds.select(["Season", "TeamID", "seed_num"]),
              left_on=["Season", "LTeamID"], right_on=["Season", "TeamID"], how="left")
        .rename({"seed_num": "l_seed"})
    )

    # Assign team A = higher seed (lower seed number)
    results = results.with_columns(
        pl.when(pl.col("w_seed") <= pl.col("l_seed"))
        .then(pl.col("WTeamID")).otherwise(pl.col("LTeamID")).alias("team_a_kaggle_id"),
        pl.when(pl.col("w_seed") <= pl.col("l_seed"))
        .then(pl.col("LTeamID")).otherwise(pl.col("WTeamID")).alias("team_b_kaggle_id"),
        pl.when(pl.col("w_seed") <= pl.col("l_seed"))
        .then(1).otherwise(0).alias("team_a_won")
    )

    # Map kaggle team IDs to cbbd team names via MTeams.csv
    results = (
        results
        .join(teams.select(["TeamID", "TeamName"]),
              left_on="team_a_kaggle_id", right_on="TeamID", how="inner")
        .rename({"TeamName": "team_a_name"})
        .join(teams.select(["TeamID", "TeamName"]),
              left_on="team_b_kaggle_id", right_on="TeamID", how="inner")
        .rename({"TeamName": "team_b_name"})
    )

    features = generate_team_features(seasons)

    # Join team A and B features by team name + season
    df = (
        results
        .join(features, left_on=["team_a_name", "Season"],
              right_on=["team", "season"], how="inner")
        .rename({c: f"a_{c}" for c in features.columns if c not in ["team", "season"]})
        .join(features, left_on=["team_b_name", "Season"],
              right_on=["team", "season"], how="inner")
        .rename({c: f"b_{c}" for c in features.columns if c not in ["team", "season"]})
    )

    return df.select(
        ["Season", "team_a_name", "team_b_name", "team_a_won"]
        + [c for c in df.columns if c.startswith("a_") or c.startswith("b_")]
    )