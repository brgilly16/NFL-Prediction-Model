import re
import json
import time as clock
import pickle
import numpy as np
import pandas as pd
from src.nflmodel.download import DATA
from src.nflmodel.train import GameModel, homeGames
from src.nflmodel.build import TEAM_NAMES, DROPBACKS_PER_GAME
# writes webapp/data.js: the trained model's weights plus every team's and QB's current state, the schedule and the backtest
# the webpage runs the model itself in JavaScript, so it works as a static page with no server
OUTPUT = "webapp/data.js"
FIXED = ("home", "homeEdge", "leaguePoints")
def linear(scaler, columns, coef, intercept):
    return {"features": list(columns), "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "coef": np.ravel(coef).tolist(), "intercept": float(np.ravel(intercept)[0])}
def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return None if np.isnan(value) else round(float(value), 5)
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value
def oppName(column):
    return "opp" + column[0].upper() + column[1:]
def neutralPower(model, df):
    # PowerScore: the model's expected margin against a league-average team on a neutral field, both teams rested normally
    X = model.features(df)
    columns = list(X.columns)
    mean = pd.Series(model.pointsScaler.mean_, index=columns)
    own = [c for c in columns if not c.startswith("opp") and c not in FIXED]
    versus, against = X.copy(), X.copy()
    for c in own:
        versus[oppName(c)] = mean[oppName(c)]
        against[oppName(c)] = X[c]
        against[c] = mean[c]
    for frame in (versus, against):
        frame["home"] = 0
        if "homeEdge" in frame:
            frame["homeEdge"] = 0
        frame["rest"] = frame["oppRest"] = 7
    predict = lambda frame: model.points.predict(model.pointsScaler.transform(frame[columns]))
    return predict(versus) - predict(against)
def exportSite():
    with open(DATA + "game_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(DATA + "model_report.json") as f:
        report = json.load(f)
    features = pd.read_csv(DATA + "features.csv", parse_dates=["date"])
    teams = pd.read_csv(DATA + "team_state.csv")
    qbs = pd.read_csv(DATA + "qb_state.csv")
    log = pd.read_csv(DATA + "backtest_recent.csv")
    season = int(teams["season"].max())
    played = features[features["played"]]
    dataSeason = int(played["season"].max())
    h = model.p["halflife"]
    # records: this season once games are played, last season's in the preseason
    record = played[played["season"] == dataSeason]
    teamRows = []
    for _, t in teams.iterrows():
        games = record[(record["team"] == t["team"]) & (record["gameType"] == "REG")]
        wins, losses = int((games["pointsFor"] > games["pointsAgainst"]).sum()), int((games["pointsFor"] < games["pointsAgainst"]).sum())
        row = {"team": t["team"], "name": t["name"], "elo": t["elo"], "qbTypical": t["qbTypical"], "leaguePoints": t["leaguePoints"],
               "homeEdge": t["homeEdge"], "lastGame": t["lastGame"], "record": [wins, losses, len(games) - wins - losses]}
        for name in ["off", "def", "offPass", "offRush", "defPass", "defRush", "offN", "defN", "pf", "pa"]:
            row[name] = t[f"{name}{h}"]
        teamRows.append(row)
    qbRows = {team: [{"id": q["qbId"], "name": q["name"], "rating": q["rating"], "cpoe": q["cpoe"], "starts": int(q["starts"]),
                      "seasonStarts": int(q["seasonStarts"]), "dropbacks": q["experience"], "projected": bool(q["projected"])}
                     for _, q in group.iterrows()] for team, group in qbs.groupby("team")}
    # PowerScore trend over the last two seasons, going into each game
    recentRows = played[played["season"] >= dataSeason - 1].copy()
    recentRows["power"] = neutralPower(model, recentRows)
    trends, recent = {}, {}
    for team, group in recentRows.sort_values("date").groupby("team"):
        trends[team] = [[d.strftime("%Y-%m-%d"), int(s), int(w), p, e] for d, s, w, p, e in
                        zip(group["date"], group["season"], group["week"], group["power"], group["elo"])]
        last = group.tail(8).iloc[::-1]
        recent[team] = [[d.strftime("%Y-%m-%d"), int(w), o, int(hm), int(n), int(pf), int(pa), q, s] for d, w, o, hm, n, pf, pa, q, s in
                        zip(last["date"], last["week"], last["opponent"], last["isHome"], last["neutral"], last["pointsFor"],
                            last["pointsAgainst"], last["qbName"], last["spreadLine"])]
    # upcoming games of the current season (the webpage shows the next week or two)
    upcoming = homeGames(features[~features["played"] & (features["season"] == season)])
    away = features[(features["isHome"] == 0) & ~features["played"]].set_index("gameId")
    games = pd.read_csv("data/raw/games.csv").set_index("game_id")
    schedule = []
    for _, g in upcoming.sort_values(["date", "gameId"]).iterrows():
        a = away.loc[g["gameId"]]
        info = games.loc[g["gameId"]]
        schedule.append({"id": g["gameId"], "week": int(g["week"]), "type": g["gameType"], "date": g["date"].strftime("%Y-%m-%d"),
                         "time": info["gametime"] if isinstance(info["gametime"], str) else None, "weekday": info["weekday"],
                         "home": g["team"], "away": a["team"], "neutral": int(g["neutral"]), "div": int(g["divGame"]),
                         "homeRest": g["rest"], "awayRest": a["rest"], "homeQb": g["qbId"] if isinstance(g["qbId"], str) else None,
                         "awayQb": a["qbId"] if isinstance(a["qbId"], str) else None, "spread": g["spreadLine"], "total": g["totalLine"],
                         "stadium": info["stadium"] if isinstance(info["stadium"], str) else None})
    columns = model.features(features.head(2)).columns
    data = {
        "season": season, "dataSeason": dataSeason, "completeSeason": int(report["completeSeason"]),
        "asOf": str(played["date"].max().date()), "names": TEAM_NAMES, "dropbacksPerGame": DROPBACKS_PER_GAME,
        "model": {"params": model.p, "sigma": model.sigma,
                  "points": linear(model.pointsScaler, columns, model.points.coef_, model.points.intercept_),
                  "win": linear(model.winScaler, columns, model.win.coef_, model.win.intercept_)},
        "teams": teamRows, "qbs": qbRows, "trends": trends, "recent": recent, "schedule": schedule, "report": report,
        "log": log[["date", "season", "week", "home", "away", "homeScore", "awayScore", "homePoints", "awayPoints", "pWin",
                    "spreadLine", "neutral", "homeQb", "awayQb"]].values.tolist()
    }
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write("window.NFL_DATA = " + json.dumps(clean(data), separators=(",", ":")) + ";\n")
    # stamp a version on the script links so browsers load the new files right after an update instead of a cached copy
    page = open("webapp/index.html", encoding="utf-8").read()
    page = re.sub(r'src="(data|app)\.js(\?v=\d+)?"', lambda m: f'src="{m.group(1)}.js?v={int(clock.time())}"', page)
    open("webapp/index.html", "w", encoding="utf-8").write(page)
    print("Wrote", OUTPUT)
if __name__ == "__main__":
    exportSite()
