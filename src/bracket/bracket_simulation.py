

from src.bracket.bracket import Bracket

def run_simulation(sim_amt=10000, verbose=False)->dict[str,float]:
    """
    :param sim_amt: Amount of simulations to run
    :param verbose: True for updates of how many sims have run
    :return: Dictionary mapping team to win probability
    """

    bracket = Bracket(season=2026)
    winners = {}

    for i in range(sim_amt):
        bracket.regions = {}
        bracket.final_four = []
        bracket.championship = None
        bracket.winner = None

        result = bracket.simulate(mode="probabilistic")
        winner = result["winner"]["team_name"]
        winners[winner] = winners.get(winner, 0) + 1

        if verbose and i % 250 == 0:
            print(f"{(i/sim_amt) * 100}% done")

    sorted_winners = sorted(winners.items(), key=lambda x: x[1], reverse=True)

    result = {}
    for team, count in sorted_winners:
        result[team] = count / sim_amt

    return result