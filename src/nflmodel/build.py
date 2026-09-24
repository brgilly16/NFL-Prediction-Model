import numpy as np
import pandas as pd
from src.nflmodel.download import DATA, RAW, downloadAll, loadGames, loadSeasons, currentSeason
# builds one row per team per game where every feature only uses information from BEFORE that game
# upcoming games get rows too (with no result), and each team's and QB's current state is saved for predicting any matchup
# relocated franchises keep their history
FRANCHISE_MAP = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
TEAM_NAMES = {"ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills",
              "CAR": "Carolina Panthers", "CHI": "Chicago Bears", "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns",
              "DAL": "Dallas Cowboys", "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
              "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs",
              "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers", "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins",
              "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
              "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers", "SEA": "Seattle Seahawks",
              "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans", "WAS": "Washington Commanders"}
# team ratings are kept at several decay speeds (half-life in games) so training can pick the best one
HALFLIVES = [4, 8, 16]
TEAM_PRIOR_PLAYS = 300     # a team's EPA rating is shrunk toward average by this many plays (about 5 games)
TEAM_SEASON_KEEP = 0.5     # share of a team's accumulated evidence kept over the offseason
POINTS_PRIOR_GAMES = 4
# QB rating: recency weighted EPA per dropback, shrunk toward a below-average (backup level) prior
QB_HALFLIFE = 12           # games
QB_PRIOR_DROPBACKS = 150
QB_PRIOR_EPA = -0.08
QB_SEASON_KEEP = 0.75
QB_TYPICAL_HALFLIFE = 4    # the team's usual starter quality, in games
DROPBACKS_PER_GAME = 38    # to show QB ratings as points per game above average
# Elo, in the style of FiveThirtyEight's NFL model
ELO_K = 20
ELO_HOME = 48
ELO_REGRESS = 1 / 3
ELO_MEAN = 1505
def franchise(code):
    return FRANCHISE_MAP.get(code, code)
def teamRows(games):
    # one row per team per game from the schedule (home and away)
    games = games.copy()
    games["neutral"] = (games["location"] == "Neutral").astype(int)
    games["played"] = games["home_score"].notna()
    rows = []
    for side, other in (("home", "away"), ("away", "home")):
        rows.append(pd.DataFrame({
            "gameId": games["game_id"], "season": games["season"], "week": games["week"], "gameType": games["game_type"],
            "date": pd.to_datetime(games["gameday"]), "code": games[side + "_team"], "oppCode": games[other + "_team"],
            "isHome": int(side == "home"), "home": ((side == "home") & (games["neutral"] == 0)).astype(int),
            "neutral": games["neutral"], "played": games["played"],
            "pointsFor": games[side + "_score"], "pointsAgainst": games[other + "_score"],
            "rest": games[side + "_rest"], "divGame": games["div_game"],
            "qbId": games[side + "_qb_id"], "qbName": games[side + "_qb_name"],
            # betting lines from this team's side: the expected margin (positive = favored) and the expected total
            "spreadLine": games["spread_line"] * (1 if side == "home" else -1), "totalLine": games["total_line"]
        }))
    df = pd.concat(rows, ignore_index=True)
    df["team"] = df["code"].map(franchise)
    df["opponent"] = df["oppCode"].map(franchise)
    return df
def attachStats(df):
    # play-by-play uses today's team codes for every season (LA for the St. Louis Rams), so match on the franchise
    stats = loadSeasons("team_games").drop(columns=["season"]).rename(columns={"game_id": "gameId"})
    stats["team"] = stats["team"].map(franchise)
    df = df.merge(stats, on=["gameId", "team"], how="left")
    # the defense side of each row is the opponent's offense in the same game
    allowed = stats.rename(columns={"team": "opponent"}).rename(columns={c: "opp_" + c for c in stats.columns if c not in ("gameId", "team")})
    df = df.merge(allowed, on=["gameId", "opponent"], how="left")
    return df
def matchQbIds(df):
    # upcoming games in the schedule often list the projected starter by name only, match the name to a player id
    known = df.dropna(subset=["qbId", "qbName"]).sort_values("date").drop_duplicates("qbName", keep="last").set_index("qbName")["qbId"]
    try:
        roster = pd.read_csv(RAW + "rosters.csv")
        roster = roster[roster["position"] == "QB"].dropna(subset=["gsis_id"])
        known = pd.concat([roster.set_index("full_name")["gsis_id"], known])
        known = known[~known.index.duplicated(keep="first")]
    except FileNotFoundError:
        pass
    missing = df["qbId"].isna() & df["qbName"].notna()
    df.loc[missing, "qbId"] = df.loc[missing, "qbName"].map(known)
    return df
class Rating:
    # recency weighted rate (sum of values / sum of weights), shrunk toward `center` by `prior` weight,
    # with part of the evidence dropped between seasons. pre() gives each row's value going into the game.
    def __init__(self, halflife, prior, center, seasonKeep):
        self.decay, self.prior, self.center, self.keep = 0.5 ** (1 / halflife), prior, center, seasonKeep
        self.S, self.W, self.last = {}, {}, {}
    def value(self, key, season):
        if key not in self.S:
            return self.center, 0.0
        keep = self.keep if self.last[key] != season else 1.0
        S, W = self.S[key] * keep, self.W[key] * keep
        return (S + self.prior * self.center) / (W + self.prior), W
    def update(self, key, season, value, weight):
        if key not in self.S:
            self.S[key], self.W[key], self.last[key] = 0.0, 0.0, season
        if self.last[key] != season:
            self.S[key] *= self.keep
            self.W[key] *= self.keep
            self.last[key] = season
        if weight > 0 and not np.isnan(value):
            self.S[key] = self.decay * self.S[key] + value
            self.W[key] = self.decay * self.W[key] + weight
def runRating(df, key, value, weight, rating):
    pre = np.empty(len(df))
    for i, (k, season, v, w) in enumerate(zip(df[key].values, df["season"].values, df[value].values, df[weight].values)):
        pre[i] = rating.value(k, season)[0]
        rating.update(k, season, v, 0.0 if np.isnan(w) else w)
    return pre
# (feature, numerator, denominator): offense is the team's own plays, defense is what it allowed
TEAM_STATS = [("off", "epa", "plays"), ("def", "opp_epa", "opp_plays"),
              ("offPass", "passEPA", "passPlays"), ("offRush", "rushEPA", "rushPlays"),
              ("defPass", "opp_passEPA", "opp_passPlays"), ("defRush", "opp_rushEPA", "opp_rushPlays"),
              ("offN", "neutralEPA", "neutralPlays"), ("defN", "opp_neutralEPA", "opp_neutralPlays")]
def addTeamRatings(df, season):
    df = df.sort_values(["date", "gameId", "isHome"]).reset_index(drop=True)
    df["one"] = df["played"].astype(float)
    leaguePoints = float(df.loc[df["played"], "pointsFor"].mean())
    state = {}
    for h in HALFLIVES:
        for name, num, den in TEAM_STATS:
            rating = Rating(h, TEAM_PRIOR_PLAYS, 0.0, TEAM_SEASON_KEEP)
            df[f"{name}{h}"] = runRating(df, "team", num, den, rating)
            state[f"{name}{h}"] = rating
        for name, col in (("pf", "pointsFor"), ("pa", "pointsAgainst")):
            rating = Rating(h, POINTS_PRIOR_GAMES, leaguePoints, TEAM_SEASON_KEEP)
            df[f"{name}{h}"] = runRating(df, "team", col, "one", rating)
            state[f"{name}{h}"] = rating
    current = pd.DataFrame({name: {team: r.value(team, season)[0] for team in r.S} for name, r in state.items()})
    return df, current
def addLeague(df):
    # league scoring environment going into each date (points per team game)
    played = df[df["played"]]
    daily = played.groupby("date")["pointsFor"].agg(["sum", "count"]).sort_index()
    rate = daily["sum"].ewm(halflife=256).mean() / daily["count"].ewm(halflife=256).mean()
    df["leaguePoints"] = df["date"].map(rate.shift(1)).ffill().fillna(float(rate.iloc[0]))
    df.loc[~df["played"], "leaguePoints"] = float(rate.iloc[-1])
    # league home field edge going into each date: recency weighted home margin in non-neutral games (it has shrunk over the years)
    home = played[(played["isHome"] == 1) & (played["neutral"] == 0)]
    margin = (home["pointsFor"] - home["pointsAgainst"]).groupby(home["date"]).agg(["sum", "count"]).sort_index()
    edge = margin["sum"].ewm(halflife=512).mean() / margin["count"].ewm(halflife=512).mean()
    df["homeEdge"] = df["date"].map(edge.shift(1)).ffill().fillna(3.0)
    df.loc[~df["played"], "homeEdge"] = float(edge.iloc[-1])
    return df, {"leaguePoints": float(rate.iloc[-1]), "homeEdge": float(edge.iloc[-1])}
def addQbRatings(df, season):
    # each starter's rating going into the game, from his own dropbacks on any team
    qb = loadSeasons("qb_games").rename(columns={"game_id": "gameId"})
    dates = df.drop_duplicates("gameId").set_index("gameId")["date"]
    qb["date"] = qb["gameId"].map(dates)
    qb = qb.dropna(subset=["date"])
    # events: QB game logs (updates) and team rows (lookups), processed in date order with lookups first on each date
    rating = Rating(QB_HALFLIFE, QB_PRIOR_DROPBACKS, QB_PRIOR_EPA, QB_SEASON_KEEP)
    cpoe = Rating(QB_HALFLIFE, 300, 0.0, QB_SEASON_KEEP)
    seasons = df.drop_duplicates("gameId").set_index("gameId")["season"]
    qb["season"] = qb["gameId"].map(seasons)
    qb = qb.sort_values(["date", "gameId"])
    updates = {d: g for d, g in qb.groupby("date")}
    pre, preCpoe, exp = np.full(len(df), np.nan), np.full(len(df), np.nan), np.zeros(len(df))
    order = df.sort_values(["date", "gameId"]).index
    byDate = {d: idx for d, idx in pd.Series(order, index=df.loc[order, "date"].values).groupby(level=0)}
    for d in sorted(set(byDate) | set(updates)):
        for i in byDate.get(d, []):
            q, s = df.at[i, "qbId"], df.at[i, "season"]
            if isinstance(q, str):
                pre[i], exp[i] = rating.value(q, s)
                preCpoe[i] = cpoe.value(q, s)[0]
        if d in updates:
            for row in updates[d].itertuples():
                rating.update(row.qbId, row.season, row.epa, row.dropbacks)
                cpoe.update(row.qbId, row.season, row.cpoe, row.cpoePlays)
    # games with no listed starter (rare) get the backup level prior
    df["qbRating"] = np.where(np.isnan(pre), QB_PRIOR_EPA, pre)
    df["qbCpoe"] = np.where(np.isnan(preCpoe), 0.0, preCpoe)
    df["qbExperience"] = exp
    # the team's usual starter quality, recency weighted over its recent games (before this one)
    df = df.sort_values(["date", "gameId", "isHome"]).reset_index(drop=True)
    typical = Rating(QB_TYPICAL_HALFLIFE, 0.5, QB_PRIOR_EPA, 1.0)
    df["qbTypical"] = runRating(df, "team", "qbRating", "one", typical)
    df["qbDelta"] = df["qbRating"] - df["qbTypical"]
    # current state: every QB's rating for a game this season, and each team's usual starter quality
    names = qb.sort_values("date").groupby("qbId")["name"].last()
    fullNames = df.dropna(subset=["qbId", "qbName"]).sort_values("date").groupby("qbId")["qbName"].last()
    qbState = pd.DataFrame([{"qbId": q, "name": fullNames.get(q, names.get(q)), "rating": rating.value(q, season)[0],
                             "cpoe": cpoe.value(q, season)[0], "experience": rating.value(q, season)[1]} for q in rating.S])
    teamTypical = pd.Series({t: typical.value(t, season)[0] for t in typical.S}, name="qbTypical")
    return df, qbState, teamTypical, qb
def addElo(df):
    home = df[(df["isHome"] == 1) & df["played"]].sort_values(["date", "gameId"])
    elo, last = {}, {}
    pre = {}
    for row in home.itertuples():
        for team in (row.team, row.opponent):
            if team not in elo:
                elo[team] = ELO_MEAN
            elif last[team] != row.season:
                elo[team] = ELO_MEAN + (elo[team] - ELO_MEAN) * (1 - ELO_REGRESS)
            last[team] = row.season
        h, a = row.team, row.opponent
        pre[(row.gameId, h)], pre[(row.gameId, a)] = elo[h], elo[a]
        edge = elo[h] - elo[a] + (0 if row.neutral else ELO_HOME)
        expected = 1 / (1 + 10 ** (-edge / 400))
        margin = row.pointsFor - row.pointsAgainst
        result = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
        winnerEdge = edge if margin >= 0 else -edge
        multiplier = np.log(abs(margin) + 1) * 2.2 / (winnerEdge * 0.001 + 2.2)
        change = ELO_K * multiplier * (result - expected)
        elo[h] += change
        elo[a] -= change
    df["elo"] = [pre.get((g, t), np.nan) for g, t in zip(df["gameId"], df["team"])]
    # upcoming games: the latest Elo, regressed if it is a new season
    upcoming = df["elo"].isna()
    df.loc[upcoming, "elo"] = [ELO_MEAN + (elo.get(t, ELO_MEAN) - ELO_MEAN) * ((1 - ELO_REGRESS) if last.get(t) != s else 1)
                               for t, s in zip(df.loc[upcoming, "team"], df.loc[upcoming, "season"])]
    return df, elo, last
def ratingColumns():
    return [f"{name}{h}" for h in HALFLIVES for name in [s[0] for s in TEAM_STATS] + ["pf", "pa"]]
def addOpponent(df):
    columns = ["elo", "rest", "qbRating", "qbCpoe", "qbTypical", "qbDelta", "qbExperience"] + ratingColumns()
    opp = df[["gameId", "team"] + columns].rename(columns={"team": "opponent"})
    opp = opp.rename(columns={c: "opp_" + c for c in columns})
    return df.merge(opp, on=["gameId", "opponent"], how="left")
def buildFeatures():
    downloadAll()
    season = currentSeason()
    df = teamRows(loadGames())
    df = attachStats(df)
    df = matchQbIds(df)
    df, teamState = addTeamRatings(df, season)
    df, leagueNow = addLeague(df)
    df, qbState, teamTypical, qbGames = addQbRatings(df, season)
    df, elo, eloSeason = addElo(df)
    df = addOpponent(df)
    df["rest"] = df["rest"].fillna(7)
    df["opp_rest"] = df["opp_rest"].fillna(7)
    df = df.sort_values(["date", "gameId", "isHome"]).reset_index(drop=True)
    keep = [c for c in df.columns if not c.startswith("opp_") or c[4:] not in
            ["plays", "epa", "success", "passPlays", "passEPA", "rushPlays", "rushEPA", "neutralPlays", "neutralEPA", "turnovers"]]
    df[keep].round(5).to_csv(DATA + "features.csv", index=False)
    saveState(df, teamState, teamTypical, qbState, qbGames, elo, eloSeason, leagueNow, season)
    print("Finished building features. Rows:", len(df), "played:", int(df["played"].sum()))
    return df
def saveState(df, teamState, teamTypical, qbState, qbGames, elo, eloSeason, leagueNow, season):
    teams = teamState.loc[list(TEAM_NAMES)].copy()
    teams.index.name = "team"
    teams["name"] = pd.Series(TEAM_NAMES)
    teams["elo"] = [ELO_MEAN + (elo[t] - ELO_MEAN) * ((1 - ELO_REGRESS) if eloSeason[t] != season else 1) for t in teams.index]
    teams["qbTypical"] = teamTypical.reindex(teams.index)
    teams["leaguePoints"] = leagueNow["leaguePoints"]
    teams["homeEdge"] = leagueNow["homeEdge"]
    played = df[df["played"]]
    teams["lastGame"] = played.groupby("team")["date"].max().reindex(teams.index).dt.strftime("%Y-%m-%d")
    teams["season"] = season
    teams.reset_index().to_csv(DATA + "team_state.csv", index=False)
    # QBs per team: the current roster's QBs plus anyone who started for the team this season or last
    try:
        roster = pd.read_csv(RAW + "rosters.csv")
        roster = roster[(roster["position"] == "QB") & roster["team"].notna()].copy()
        roster["team"] = roster["team"].map(franchise)
        roster = roster[roster["status"].isin(["ACT", "RES", "INA", "DEV"]) | roster["status"].isna()]
        rosterQbs = roster[["team", "gsis_id", "full_name", "status"]].rename(columns={"gsis_id": "qbId", "full_name": "rosterName"})
    except FileNotFoundError:
        rosterQbs = pd.DataFrame(columns=["team", "qbId", "rosterName", "status"])
    recent = df[df["played"] & (df["season"] >= season - 1) & df["qbId"].notna()]
    starters = recent.groupby(["team", "qbId"]).agg(starts=("gameId", "size"), lastStart=("date", "max")).reset_index()
    seasonStarts = df[df["played"] & (df["season"] == season)].groupby(["team", "qbId"]).size().rename("seasonStarts")
    # a starter who has since left the team (on another roster now) is dropped from his old team
    onRoster = set(rosterQbs["qbId"])
    rosterTeam = rosterQbs.set_index("qbId")["team"].to_dict()
    starters = starters[[q not in onRoster or rosterTeam[q] == t for t, q in zip(starters["team"], starters["qbId"])]]
    qbs = pd.concat([rosterQbs[["team", "qbId", "rosterName"]], starters[["team", "qbId"]]]).drop_duplicates(["team", "qbId"])
    qbs = qbs.merge(starters, on=["team", "qbId"], how="left").join(seasonStarts, on=["team", "qbId"])
    qbs = qbs.merge(qbState, on="qbId", how="left")
    qbs["name"] = qbs["name"].fillna(qbs["rosterName"])
    qbs["rating"] = qbs["rating"].fillna(QB_PRIOR_EPA)
    qbs["cpoe"] = qbs["cpoe"].fillna(0.0)
    qbs[["starts", "seasonStarts", "experience"]] = qbs[["starts", "seasonStarts", "experience"]].fillna(0)
    # projected starter: the QB listed for the team's next scheduled game, otherwise its most recent starter
    upcoming = df[~df["played"] & df["qbId"].notna()].sort_values("date").drop_duplicates("team")
    projected = upcoming.set_index("team")["qbId"].to_dict()
    lastStarter = df[df["played"] & df["qbId"].notna()].sort_values("date").drop_duplicates("team", keep="last").set_index("team")["qbId"].to_dict()
    qbs["projected"] = [q == projected.get(t, lastStarter.get(t)) for t, q in zip(qbs["team"], qbs["qbId"])]
    qbs = qbs.sort_values(["team", "projected", "seasonStarts", "starts", "experience"], ascending=[True, False, False, False, False])
    qbs = qbs[qbs["team"].isin(TEAM_NAMES)]
    qbs.drop(columns=["rosterName"]).to_csv(DATA + "qb_state.csv", index=False)
if __name__ == "__main__":
    buildFeatures()
