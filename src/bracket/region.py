
import random
from src.bracket.team import Team
from src.bracket.game import Game

class Region:
    def __init__(self, name: str, teams: list[Team]):
        """
        :param name: Region name (East, West, South, Midwest)
        :param teams: 16 teams seeded 1-16
        """
        self.name: str = name
        self.teams: list[Team] = sorted(teams, key=lambda t: t.seed)
        self.games: list[Game] = []
        self.winner: Team = None

    def _pick_winner(self, game: Game, mode: str) -> Team:
        game.generate_probabilities()
        if mode == "deterministic":
            return game.team_a if game.a_prob >= game.b_prob else game.team_b
        else:
            return game.team_a if random.random() <= game.a_prob else game.team_b

    def simulate(self, mode: str = "probabilistic") -> Team:
        """
        Simulate all 4 rounds of a region
        :param mode: 'deterministic' or 'probabilistic'
        :return: Region winner
        """
        # First round matchups: 1v16, 2v15, 3v14, 4v13, 5v12, 6v11, 7v10, 8v9
        matchups = [(0, 15), (1, 14), (2, 13), (3, 12), (4, 11), (5, 10), (6, 9), (7, 8)]
        current_round = [self.teams[a] for a in range(16)]

        for round_num in range(1, 5):
            next_round = []
            pairs = [(current_round[i], current_round[i+1]) for i in range(0, len(current_round), 2)] \
                if round_num > 1 else [( self.teams[a], self.teams[b]) for a, b in matchups]

            for team_a, team_b in pairs:
                game = Game(team_a, team_b, round_num)
                game.winner = self._pick_winner(game, mode)
                self.games.append(game)
                next_round.append(game.winner)

            current_round = next_round

        self.winner = current_round[0]
        return self.winner

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "games": [g.to_dict() for g in self.games],
            "winner": self.winner.to_dict() if self.winner else None
        }