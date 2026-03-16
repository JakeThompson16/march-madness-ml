
from sportsdataverse.mbb import espn_mbb_teams
from rapidfuzz import process
import polars as pl
from bart_torvik import import_bart


def get_bart_espn_crosswalk() -> pl.DataFrame:
    espn = espn_mbb_teams().select(["id", "display_name"])
    espn_names = espn["display_name"].to_list()
    espn_ids = espn["id"].to_list()
    name_to_id = dict(zip(espn_names, espn_ids))

    bart = import_bart(2025).select("team").unique()
    bart_names = bart["team"].to_list()

    matches = [
        (name, *process.extractOne(name, espn_names))
        for name in bart_names
    ]

    return pl.DataFrame({
        "bart_name": [m[0] for m in matches],
        "espn_name": [m[1] for m in matches],
        "espn_team_id": [name_to_id[m[1]] for m in matches]
    })