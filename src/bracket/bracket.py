import random
import importlib
from src.bracket.team import Team
from src.bracket.game import Game


class Bracket:
    def __init__(self, season: int):
        self.season = season
        self.regions: dict = {}
        self.final_four: list[Game] = []
        self.championship: Game | None = None
        self.winner: Team | None = None

        config = importlib.import_module(f"data.bracket_{season}")
        self.region_config: dict[str, dict[int, str]] = config.REGIONS_2026

    def _build_teams(self, region_name: str) -> list[Team]:
        seed_map = self.region_config[region_name]
        return [
            Team(team_name=name, seed=seed, season=self.season)
            for seed, name in sorted(seed_map.items())
        ]

    def _pick_winner(self, game: Game, mode: str) -> Team:
        game.generate_probabilities()
        if mode == "deterministic":
            return game.team_a if game.a_prob >= game.b_prob else game.team_b
        else:
            return game.team_a if random.random() <= game.a_prob else game.team_b

    def _simulate_region(self, region_name: str, mode: str) -> Team:
        teams = self._build_teams(region_name)
        games = []

        # Standard NCAA first round matchups: 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15
        matchup_order = [(0, 15), (7, 8), (4, 11), (3, 12), (5, 10), (2, 13), (6, 9), (1, 14)]
        current_round = [(teams[a], teams[b]) for a, b in matchup_order]

        for round_num in range(1, 5):
            next_round = []
            for team_a, team_b in current_round:
                game = Game(team_a, team_b, round_num)
                game.winner = self._pick_winner(game, mode)
                games.append(game)
                next_round.append(game.winner)
            if len(next_round) > 1:
                current_round = [(next_round[i], next_round[i + 1]) for i in range(0, len(next_round), 2)]

        self.regions[region_name] = games
        return games[-1].winner

    def simulate(self, mode: str = "probabilistic") -> dict:
        """ Simulates bracket and returns dict form of it """
        region_winners = [
            self._simulate_region(name, mode)
            for name in self.region_config.keys()
        ]

        ff_game_1 = Game(region_winners[0], region_winners[1], 5)
        ff_game_1.winner = self._pick_winner(ff_game_1, mode)
        ff_game_2 = Game(region_winners[2], region_winners[3], 5)
        ff_game_2.winner = self._pick_winner(ff_game_2, mode)
        self.final_four = [ff_game_1, ff_game_2]

        self.championship = Game(ff_game_1.winner, ff_game_2.winner, 6)
        self.championship.winner = self._pick_winner(self.championship, mode)
        self.winner = self.championship.winner

        return self.to_dict()

    def to_dict(self) -> dict:
        return {
            "season": self.season,
            "regions": {
                name: [g.to_dict() for g in games]
                for name, games in self.regions.items()
            },
            "final_four": [g.to_dict() for g in self.final_four],
            "championship": self.championship.to_dict() if self.championship else None,
            "winner": self.winner.to_dict() if self.winner else None,
        }