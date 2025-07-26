import pandas as pd
import logging

logger = logging.getLogger(__name__)


class FeaturesEngine:
    def __init__(self):
        self.league = None
        self.leaderboard = None

    def generate_features(self, df):
        logger.info("Generating features from the dataset.")
        self.league = pd.DataFrame(
            {'Points': 0, 'Goals_scored': 0, 'Goals_conceded': 0},
            index=df['HomeTeam'].unique()
        )
        self.leaderboard = pd.DataFrame(
            {'Goals_scored': 0, 'Goals_conceded': 0},
            index=df['HomeTeam'].unique()
        )

        df['H_points'] = df['FTR'].map({'H': 3, 'D': 1, 'A': 0})
        df['A_points'] = df['FTR'].map({'A': 3, 'D': 1, 'H': 0})

        df['h_pts'], df['a_pts'], df['h_rank'], df['a_rank'] = zip(
            *df.apply(self._points_rank, axis=1))
        df['h_gls'], df['h_gls_conc'], df['a_gls'], df['a_gls_con'] = zip(
            *df.apply(self._goals, axis=1))

        return df.drop(columns=['H_points', 'A_points'])

    def _points_rank(self, row):
        home_points = self.league.loc[row.HomeTeam, 'Points']
        away_points = self.league.loc[row.AwayTeam, 'Points']

        rank = self.league.rank(method='first', ascending=False).astype(int)
        self.league.loc[row.HomeTeam, 'Points'] += row.H_points
        self.league.loc[row.AwayTeam, 'Points'] += row.A_points

        return [home_points, away_points, rank.loc[row.HomeTeam, 'Points'], rank.loc[row.AwayTeam, 'Points']]

    def _goals(self, row):
        home_goals = self.leaderboard.loc[row.HomeTeam, [
            'Goals_scored', 'Goals_conceded']]
        away_goals = self.leaderboard.loc[row.AwayTeam, [
            'Goals_scored', 'Goals_conceded']]

        self.leaderboard.loc[row.HomeTeam, [
            'Goals_scored', 'Goals_conceded']] += row.FTHG, row.FTAG
        self.leaderboard.loc[row.AwayTeam, [
            'Goals_scored', 'Goals_conceded']] += row.FTAG, row.FTHG

        return [*home_goals, *away_goals]
