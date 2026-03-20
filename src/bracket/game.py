
from src.bracket.team import Team
from src.model.simulate import simulate_game

class Game:
    def __init__(self, team_a: Team, team_b: Team, tournament_round: int, winner=None):
        """
        :param team_a: Higher seeded team
        :param team_b: Lower seeded team
        :param tournament_round: Round of game
        :param winner: Winner of game if applicable
        """
        # ensure team a is higher seed
        if team_a.seed > team_b.seed:
            temp = team_a
            team_a = team_b
            team_b = temp
        self.team_a: Team = team_a
        self.team_b: Team = team_b
        self.tournament_round: int = tournament_round
        self.winner: Team = winner
        self.a_prob = None
        self.b_prob = None

    def generate_probabilities(self):
        """ Assigns a_prob and b_prob based on model output """
        a_features = self.team_a.features_df
        b_features = self.team_b.features_df
        self.a_prob, self.b_prob = simulate_game(a_features, b_features)

    def to_dict(self) -> dict:
        return {
            "team_a": self.team_a.to_dict(),
            "team_b": self.team_b.to_dict(),
            "round": self.tournament_round,
            "a_prob": self.a_prob,
            "b_prob": self.b_prob,
            "winner": self.winner.to_dict() if self.winner else None
        }