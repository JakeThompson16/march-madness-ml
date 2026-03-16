

import time
import requests
import polars as pl
from functools import lru_cache


TRANK_COLUMNS = [
    "team", "conf", "barthag", "barthag_rk",
    "adj_o", "adj_o_rk", "adj_d", "adj_d_rk",
    "adj_t", "adj_t_rk", "wab", "wab_rk",
    "off_efg", "off_to", "off_or", "off_ftr",
    "def_efg", "def_to", "def_or", "def_ftr",
    "wins", "losses", "games",
    "nc_elite_sos", "nc_fut_sos", "nc_cur_sos",
    "ov_elite_sos", "ov_fut_sos", "ov_cur_sos",
    "seed",
]


@lru_cache(maxsize=None)
def _fetch_season(season: int) -> pl.DataFrame:
    url = f"https://barttorvik.com/trank.php?year={season}&json=1"
    rows = requests.get(url, timeout=10).json()
    # Trim each row to the number of known columns in case BartTorvik adds extras
    trimmed = [row[:len(TRANK_COLUMNS)] for row in rows]
    return (
        pl.DataFrame(trimmed, schema=TRANK_COLUMNS, orient="row")
        .with_columns(pl.lit(season).alias("season"))
    )


def import_bart(season: int | list[int]) -> pl.DataFrame:
    """
    :return: Bart Torvig data frame form seasons
    """
    seasons = [season] if isinstance(season, int) else season
    frames = []
    for i, yr in enumerate(seasons):
        frames.append(_fetch_season(yr))
        if i < len(seasons) - 1:
            time.sleep(0.5)
    return pl.concat(frames) if len(frames) > 1 else frames[0]