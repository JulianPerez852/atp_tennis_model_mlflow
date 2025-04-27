import pandas as pd

class PlayerStatictis():
    """
        Esta clase se encargara de los metodos para explorar estadisticamente
        cada jugador.
    """
    def __init__(self, player_name: str, df: pd.DataFrame):
        self.player_name = player_name
        self.df = df

    def _filter_df(self):
        df_filtered = self.df[
            (self.df['Player_1'] == self.player_name)\
            | (self.df['Player_2']==self.player_name)
            ]
        return df_filtered
    
    def _filter_no_points_games(self, df):
        df_no_pts = df[(df['Pts_1']==-1) | (df['Pts_2']==-1)]
        if len(df_no_pts) == 0:
            return (None, False)
        return (df_no_pts, True)
    
    def extract_stats(self):
        df_filtered = self._filter_df()
        df_no_pts, is_procesable = self._filter_no_points_games(df_filtered)

        if is_procesable:
            df_winrates = self._extract_winrates(df_no_pts)
        else:
            df_winrates = self._extract_winrates_pts_matchs(df_filtered)

        df_stats = self.calculate_winrates(df_winrates, df_filtered)

        return df_stats

    def _extract_winrates(self, df_no_pts):
        """
        Calcula el win rate total, por tipo de cancha, Series y superficie
        para los partidos donde los jugadores no tienen puntos.
        """
        if df_no_pts is None:
            return None

        # Filtrar partidos ganados por el jugador
        df_no_pts['Winner'] = df_no_pts.apply(
            lambda row: self.player_name if (
                (row['Player_1'] == self.player_name and row['Result'] == 1) or
                (row['Player_2'] == self.player_name and row['Result'] == 2)
            ) else None, axis=1
        )
        df_no_pts = df_no_pts[df_no_pts['Winner'] == self.player_name]

        # Calcular win rates
        winrate_total = len(df_no_pts) / len(df_no_pts)
        winrate_by_surface = df_no_pts.groupby('Surface').size() / df_no_pts['Surface'].value_counts()
        winrate_by_series = df_no_pts.groupby('Series').size() / df_no_pts['Series'].value_counts()
        winrate_by_court = df_no_pts.groupby('Court').size() / df_no_pts['Court'].value_counts()

        return {
            'Total': winrate_total,
            'By_Surface': winrate_by_surface.to_dict(),
            'By_Series': winrate_by_series.to_dict(),
            'By_Court': winrate_by_court.to_dict()
        }

    def _extract_winrates_pts_matchs(self, df_filtered):
        """
        Calcula el win rate total, por tipo de cancha, Series y superficie
        para los primeros 15% de partidos del jugador.
        """
        # Ordenar por fecha y seleccionar el 15% de los partidos
        df_filtered = df_filtered.sort_values(by='Date')
        top_15_percent = int(len(df_filtered) * 0.15)
        df_top_matches = df_filtered.iloc[:top_15_percent]

        # Filtrar partidos ganados por el jugador
        df_top_matches['Winner'] = df_top_matches.apply(
            lambda row: self.player_name if (
                (row['Player_1'] == self.player_name and row['Result'] == 1) or
                (row['Player_2'] == self.player_name and row['Result'] == 2)
            ) else None, axis=1
        )
        df_top_matches = df_top_matches[df_top_matches['Winner'] == self.player_name]

        # Calcular win rates
        winrate_total = len(df_top_matches) / top_15_percent
        winrate_by_surface = df_top_matches.groupby('Surface').size() / df_top_matches['Surface'].value_counts()
        winrate_by_series = df_top_matches.groupby('Series').size() / df_top_matches['Series'].value_counts()
        winrate_by_court = df_top_matches.groupby('Court').size() / df_top_matches['Court'].value_counts()

        return {
            'Total': winrate_total,
            'By_Surface': winrate_by_surface.to_dict(),
            'By_Series': winrate_by_series.to_dict(),
            'By_Court': winrate_by_court.to_dict()
        }

    def calculate_winrates(self, df_winrates, df_filtered):
        """
        Actualiza los win rates partido a partido en orden cronológico.
        """
        df_filtered = df_filtered.sort_values(by='Date')
        winrate_1, winrate_2 = 0, 0
        winrate_by_surface_1, winrate_by_surface_2 = {}, {}
        winrate_by_series_1, winrate_by_series_2 = {}, {}
        winrate_by_court_1, winrate_by_court_2 = {}, {}

        for index, row in df_filtered.iterrows():
            if row['Player_1'] == self.player_name:
                winrate_1 += 1 if row['Result'] == 1 else 0
                winrate_by_surface_1[row['Surface']] = winrate_by_surface_1.get(row['Surface'], 0) + (1 if row['Result'] == 1 else 0)
                winrate_by_series_1[row['Series']] = winrate_by_series_1.get(row['Series'], 0) + (1 if row['Result'] == 1 else 0)
                winrate_by_court_1[row['Court']] = winrate_by_court_1.get(row['Court'], 0) + (1 if row['Result'] == 1 else 0)
                row['Winrate_1'] = winrate_1 / (index + 1)
            elif row['Player_2'] == self.player_name:
                winrate_2 += 1 if row['Result'] == 2 else 0
                winrate_by_surface_2[row['Surface']] = winrate_by_surface_2.get(row['Surface'], 0) + (1 if row['Result'] == 2 else 0)
                winrate_by_series_2[row['Series']] = winrate_by_series_2.get(row['Series'], 0) + (1 if row['Result'] == 2 else 0)
                winrate_by_court_2[row['Court']] = winrate_by_court_2.get(row['Court'], 0) + (1 if row['Result'] == 2 else 0)
                row['Winrate_2'] = winrate_2 / (index + 1)

        return df_filtered