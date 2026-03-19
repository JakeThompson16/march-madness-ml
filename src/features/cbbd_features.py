import os
import cbbd
import polars as pl
from hidden.secrets import CBBD_API_KEY

config = cbbd.Configuration(access_token=CBBD_API_KEY)

CBBD_CACHE_PATH = "data/cbbd_cache.joblib"


def _fetch_from_api(seasons: list[int]) -> pl.DataFrame:
    frames = []
    with cbbd.ApiClient(config) as client:
        ratings_api = cbbd.RatingsApi(client)
        stats_api = cbbd.StatsApi(client)

        for season in seasons:
            ratings = ratings_api.get_adjusted_efficiency(season=season)
            stats = stats_api.get_team_season_stats(season=season)

            ratings_df = pl.DataFrame([
                {
                    "team": r.team,
                    "season": r.season,
                    "adj_em": r.net_rating,
                    "adj_o": r.offensive_rating,
                    "adj_d": r.defensive_rating,
                }
                for r in ratings
            ])

            stats_df = pl.DataFrame([
                {
                    "team": s.team,
                    "season": s.season,
                    "tempo": s.pace,
                    "efg_pct": s.team_stats.four_factors.effective_field_goal_pct,
                    "to_rate": s.team_stats.four_factors.turnover_ratio,
                    "orb_pct": s.team_stats.four_factors.offensive_rebound_pct,
                    "ft_rate": s.team_stats.four_factors.free_throw_rate,
                    "three_pt_rate": (
                        s.team_stats.three_point_field_goals.attempted / s.team_stats.field_goals.attempted
                        if s.team_stats.field_goals.attempted > 0
                        else 0
                    ),
                }
                for s in stats
            ])

            frames.append(ratings_df.join(stats_df, on=["team", "season"], how="inner"))

    return pl.concat(frames) if len(frames) > 1 else frames[0]


def cache_cbbd_data(seasons: int | list[int]) -> None:
    """
    Pulls data from CBBD API and saves to disk cache
    """
    import joblib
    if isinstance(seasons, int):
        seasons = [seasons]
    df = _fetch_from_api(seasons)
    joblib.dump(df, CBBD_CACHE_PATH)
    print(f"Cached {len(seasons)} seasons to {CBBD_CACHE_PATH}")


def extract_cbbd_data(seasons: int | list[int]) -> pl.DataFrame:
    """
    Returns polars df of necessary cbbd data from seasons.
    Loads from disk cache if available, otherwise hits API.
    """
    import joblib
    if isinstance(seasons, int):
        seasons = [seasons]

    if os.path.exists(CBBD_CACHE_PATH):
        df = joblib.load(CBBD_CACHE_PATH)
        return df.filter(pl.col("season").is_in(seasons))

    return _fetch_from_api(seasons)