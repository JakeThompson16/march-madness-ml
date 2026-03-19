from typing import Dict, Any

from src.bracket.bracket import Bracket
from src.bracket.team import Team
from src.model.simulate import simulate_game
from itertools import combinations
import time


def _precompute_matchups(bracket: Bracket) -> dict:
    """
    Precomputes all possible matchup probabilities for teams in the bracket.
    :return: Dict mapping (team_a_name, team_b_name) -> (a_prob, b_prob)
    """
    all_teams = {}
    for region, teams in bracket.region_config.items():
        for seed, name in teams.items():
            all_teams[name] = Team(team_name=name, seed=seed, season=bracket.season)

    prob_cache = {}
    names = list(all_teams.keys())
    for name_a, name_b in combinations(names, 2):
        team_a = all_teams[name_a]
        team_b = all_teams[name_b]
        a_prob, b_prob = simulate_game(team_a.features_df, team_b.features_df)
        prob_cache[(name_a, name_b)] = (a_prob, b_prob)
        prob_cache[(name_b, name_a)] = (b_prob, a_prob)

    return prob_cache


def run_simulation(sim_amt=10000, verbose=False) -> dict[str, dict[Any, float | Any]]:
    """
    :param sim_amt: Amount of simulations to run
    :param verbose: True for updates of how many sims have run
    :return: Dictionary mapping team to win probability
    """
    bracket = Bracket(season=2026)

    if verbose:
        print("Precomputing matchup probabilities...")
    t_start = time.time()
    prob_cache = _precompute_matchups(bracket)
    if verbose:
        print(f"Matchup cache built ({len(prob_cache) // 2} unique matchups) in {time.time() - t_start:.1f}s — starting simulations...")

    # Monkey-patch Game.generate_probabilities to use cache
    from src.bracket import game as game_module
    original_generate = game_module.Game.generate_probabilities

    def cached_generate_probabilities(self):
        key = (self.team_a.team_name, self.team_b.team_name)
        self.a_prob, self.b_prob = prob_cache[key]

    game_module.Game.generate_probabilities = cached_generate_probabilities

    winners = {}
    final_four_counts = {}
    start = time.time()

    for i in range(sim_amt):
        bracket.regions = {}
        bracket.final_four = []
        bracket.championship = None
        bracket.winner = None

        result = bracket.simulate(mode="probabilistic")
        winner = result["winner"]["team_name"]
        winners[winner] = winners.get(winner, 0) + 1

        for game in result["final_four"]:
            for key in ["team_a", "team_b"]:
                team = game[key]["team_name"]
                final_four_counts[team] = final_four_counts.get(team, 0) + 1

        if verbose and (i + 1) % 100 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            remaining = (sim_amt - i - 1) / rate
            print(f"  {i + 1}/{sim_amt} | {rate:.0f} sims/sec | ~{remaining:.0f}s remaining")

    # Restore original method
    game_module.Game.generate_probabilities = original_generate

    sorted_winners = sorted(winners.items(), key=lambda x: x[1], reverse=True)
    sorted_ff = sorted(final_four_counts.items(), key=lambda x: x[1], reverse=True)

    return {
        "champions": {team: count / sim_amt for team, count in sorted_winners},
        "final_four": {team: count / sim_amt for team, count in sorted_ff}
    }