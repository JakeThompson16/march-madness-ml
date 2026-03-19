

from functools import lru_cache
import polars as pl
from src.features.generate_features import generate_team_features
from src.features.nn_features import generate_nn_feature_df


@lru_cache(maxsize=None)
def retrieve_features_df(season: int)->pl.DataFrame:
    df = generate_team_features(season)
    return df

@lru_cache(maxsize=None)
def retrieve_nn_df(seasons: int)->pl.DataFrame:
    df = generate_nn_feature_df(seasons)
    return df


class Team:
    """Class for a march madness team"""
    def __init__(self, team_name: str, seed: int, season: int):
        """
        :param team_name: Name of team
        :param seed: Teams seed
        :param season: Season of tournament
        """
        self.team_name: str = team_name
        self.seed: int = seed
        self.season: int = season

        df = retrieve_features_df(season)
        self.features_df = (
            df.filter(pl.col("team") == self.team_name)
            .drop("seed")
            .with_columns(pl.lit(seed).alias("seed"))
        )

        nn_df = retrieve_nn_df(season)
        self.nn_df = (
            nn_df.filter(pl.col("team_location") == self.team_name)
            .sort("game_date")
            .tail(1)
        )

    def to_dict(self) -> dict:
        return {"team_name": self.team_name, "seed": self.seed}