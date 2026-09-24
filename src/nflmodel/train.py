import json
import pickle
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from src.nflmodel.download import DATA
# the game prediction model
# points: each team's points are a linear regression on both teams' pregame features; the margin is treated as normal
# win probability: a logistic model on the same features, blended with the win chance implied by the points model
# everything is validated walk-forward: each test season is predicted by a model trained only on earlier seasons
FIRST_SEASON = 2001
FIRST_TEST_SEASON = 2010
TIERS = [("Toss-up", 0.50), ("Lean", 0.55), ("Solid", 0.65), ("Strong", 0.75)]
DEFAULTS = {"halflife": 8, "epa": "all", "qb": "delta", "elo": "raw", "points": True, "home": "trend", "alpha": 1.0, "blend": 0.5}
def teamFeatures(df, p):
    # features for a row where "team" is scoring against "opponent"
    h = p["halflife"]
    X = pd.DataFrame(index=df.index)
    for side, pre in (("", ""), ("opp", "opp_")):
        cap = (lambda n: side + n[0].upper() + n[1:]) if side else (lambda n: n)
        if p["epa"] == "split":
            for name in ["offPass", "offRush", "defPass", "defRush"]:
                X[cap(name)] = df[f"{pre}{name}{h}"]
        else:
            suffix = "N" if p["epa"] == "neutral" else ""
            X[cap("off")] = df[f"{pre}off{suffix}{h}"]
            X[cap("def")] = df[f"{pre}def{suffix}{h}"]
        if p["points"]:
            X[cap("pointsFor")] = df[f"{pre}pf{h}"]
            X[cap("pointsAgainst")] = df[f"{pre}pa{h}"]
        if p["elo"] == "raw":
            X[cap("elo")] = df[pre + "elo"]
        if p["qb"] in ("delta", "both"):
            X[cap("qbDelta")] = df[pre + "qbDelta"]
        if p["qb"] in ("raw", "both"):
            X[cap("qb")] = df[pre + "qbRating"]
        # rest: short weeks (Thursday games) and byes, capped at 4 and 14 days
        X[cap("rest")] = df[pre + "rest"].clip(4, 14)
    X["home"] = df["home"]
    if p["home"] == "trend":
        # home field scaled by the league's recent home edge in points
        X["homeEdge"] = df["home"] * df["homeEdge"]
    X["leaguePoints"] = df["leaguePoints"]
    return X
def loadFeatures():
    df = pd.read_csv(DATA + "features.csv", parse_dates=["date"])
    return df[df["played"] & (df["season"] >= FIRST_SEASON)].reset_index(drop=True)
def testSeasons(df):
    return list(range(FIRST_TEST_SEASON, int(df["season"].max()) + 1))
def homeGames(df):
    # one row per game from the listed home team's side, with the result (ties count as half a win)
    home = df[df["isHome"] == 1].copy()
    home["homeWin"] = np.where(home["pointsFor"] > home["pointsAgainst"], 1.0, np.where(home["pointsFor"] < home["pointsAgainst"], 0.0, 0.5))
    return home
def awayRows(df, home):
    away = df[df["isHome"] == 0].set_index("gameId")
    return away.loc[home["gameId"]]
def logLoss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
def winMetrics(y, p):
    decided = y != 0.5
    return {"logLoss": logLoss(y, p), "accuracy": float(np.mean((p[decided] > 0.5) == (y[decided] == 1))), "brier": float(np.mean((p - y) ** 2))}
def expandTies(X, y):
    # ties count as half a win and half a loss when training the classifier
    tie = y == 0.5
    X2 = pd.concat([X, X[tie]])
    y2 = np.concatenate([np.where(tie, 1, y), np.zeros(tie.sum())])
    weights = np.concatenate([np.where(tie, 0.5, 1.0), np.full(tie.sum(), 0.5)])
    return X2, y2.astype(int), weights
class GameModel:
    def __init__(self, **params):
        self.p = DEFAULTS | params
    def features(self, df):
        return teamFeatures(df, self.p)
    def fit(self, df):
        X = self.features(df)
        self.pointsScaler = StandardScaler().fit(X)
        self.points = Ridge(alpha=self.p["alpha"]).fit(self.pointsScaler.transform(X), df["pointsFor"])
        home = homeGames(df)
        W = self.features(home)
        self.winScaler = StandardScaler().fit(W)
        Wx, y, weights = expandTies(pd.DataFrame(self.winScaler.transform(W), index=W.index), home["homeWin"].values)
        self.win = LogisticRegression(C=1.0, max_iter=2000).fit(Wx, y, sample_weight=weights)
        # spread of the actual margin around the predicted margin, for the points model's win chance
        margin = self.predictPoints(home) - self.predictPoints(awayRows(df, home))
        self.sigma = float(np.std((home["pointsFor"] - home["pointsAgainst"]).values - margin))
        return self
    def predictPoints(self, df):
        return self.points.predict(self.pointsScaler.transform(self.features(df)))
    def predictGames(self, df):
        home = df[df["isHome"] == 1]
        away = awayRows(df, home)
        homePoints, awayPoints = self.predictPoints(home), self.predictPoints(away)
        pLogistic = self.win.predict_proba(self.winScaler.transform(self.features(home)))[:, 1]
        pPoints = norm.cdf((homePoints - awayPoints) / self.sigma)
        pWin = self.p["blend"] * pLogistic + (1 - self.p["blend"]) * pPoints
        return pd.DataFrame({"gameId": home["gameId"].values, "homePoints": homePoints, "awayPoints": awayPoints,
                             "pLogistic": pLogistic, "pPoints": pPoints, "pWin": pWin}, index=home.index)
def walkForward(df, **params):
    results = []
    for season in testSeasons(df):
        model = GameModel(**params).fit(df[df["season"] < season])
        predictions = model.predictGames(df[df["season"] == season])
        predictions["season"] = season
        results.append(predictions)
    return pd.concat(results)
def againstSpread(home, margin):
    # picks against the betting line: the home team if the model's margin is above the line, pushes left out
    line = home["spreadLine"].values
    actual = (home["pointsFor"] - home["pointsAgainst"]).values
    valid = ~np.isnan(line) & (actual != line)
    return float(np.mean((margin[valid] > line[valid]) == (actual[valid] > line[valid])))
def evaluate(df, predictions, name):
    home = homeGames(df).loc[predictions.index]
    away = awayRows(df, home)
    y = home["homeWin"].values
    margin = (predictions["homePoints"] - predictions["awayPoints"]).values
    actualMargin = (home["pointsFor"] - home["pointsAgainst"]).values
    points = np.concatenate([home["pointsFor"].values, away["pointsFor"].values])
    predicted = np.concatenate([predictions["homePoints"].values, predictions["awayPoints"].values])
    result = {"model": name, **winMetrics(y, predictions["pWin"].values),
              "marginMAE": float(np.mean(np.abs(actualMargin - margin))), "pointsMAE": float(np.mean(np.abs(points - predicted))),
              "totalMAE": float(np.mean(np.abs((home["pointsFor"] + home["pointsAgainst"]).values - predictions[["homePoints", "awayPoints"]].sum(axis=1).values))),
              "ats": againstSpread(home, margin), "games": int(len(home))}
    print(f"{name:60s} logLoss {result['logLoss']:.4f}  acc {result['accuracy']:.4f}  brier {result['brier']:.4f}  "
          f"marginMAE {result['marginMAE']:.2f}  ATS {result['ats']:.3f}")
    return result
def baselines(df):
    home = homeGames(df)
    test = home[home["season"].isin(testSeasons(df))]
    y = test["homeWin"].values
    results = []
    homeRate = np.array([home[home["season"] < s]["homeWin"].mean() for s in test["season"]])
    results.append({"model": "Always home field rate", **winMetrics(y, homeRate)})
    edge = test["elo"] - test["opp_elo"] + np.where(test["neutral"] == 1, 0, 48)
    results.append({"model": "Elo only", **winMetrics(y, (1 / (1 + 10 ** (-edge / 400))).values)})
    # the betting market's point spread, turned into a win chance
    lined = test["spreadLine"].notna().values
    vegas = norm.cdf(test["spreadLine"].fillna(0).values / 13.2)
    market = {"model": "Betting line (Vegas spread)", **winMetrics(y[lined], vegas[lined]),
              "marginMAE": float(np.mean(np.abs((test["pointsFor"] - test["pointsAgainst"] - test["spreadLine"]).values[lined]))),
              "totalMAE": float(np.nanmean(np.abs((test["pointsFor"] + test["pointsAgainst"] - test["totalLine"]).values)))}
    results.append(market)
    for r in results:
        print(f"{r['model']:60s} logLoss {r['logLoss']:.4f}  acc {r['accuracy']:.4f}  brier {r['brier']:.4f}")
    return results
def tierOf(p):
    confidence = max(p, 1 - p)
    return [name for name, cut in TIERS if confidence >= cut][-1]
def tierReport(df, predictions, latest):
    home = homeGames(df).loc[predictions.index]
    games = pd.DataFrame({"season": predictions["season"].values, "p": predictions["pWin"].values, "y": home["homeWin"].values})
    games = games[games["y"] != 0.5]
    games["tier"] = games["p"].map(tierOf)
    games["correct"] = (games["p"] >= 0.5) == (games["y"] == 1)
    report = []
    for name, cut in TIERS:
        row = {"tier": name, "minConfidence": cut}
        for label, subset in (("all", games), ("latest", games[games["season"] == latest])):
            tier = subset[subset["tier"] == name]
            row[label] = {"games": int(len(tier)), "share": float(len(tier) / max(len(subset), 1)),
                          "accuracy": float(tier["correct"].mean()) if len(tier) else None}
        report.append(row)
        print(f"{name:8s} all seasons {row['all']['accuracy']:.3f} ({row['all']['share']:.0%} of games)")
    return report
def calibration(df, predictions):
    # games grouped by predicted home win chance (10% bins), with how often the home team really won
    y = homeGames(df).loc[predictions.index, "homeWin"].values
    games = pd.DataFrame({"p": predictions["pWin"].values, "y": y})
    games["bin"] = np.floor(games["p"] * 10).clip(0, 9)
    bins = games.groupby("bin").agg(x=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    return [{"x": float(r.x), "y": float(r.y), "n": int(r.n)} for r in bins.itertuples() if r.n >= 30]
def trainModel():
    df = loadFeatures()
    seasons = testSeasons(df)
    # the latest season that is finished (the current one may only be a few weeks in)
    complete = int(df.groupby("season")["week"].max().loc[lambda w: w >= 18].index.max())
    print("Rows:", len(df), "Seasons:", df["season"].min(), "-", df["season"].max(), "test:", seasons[0], "-", seasons[-1])
    report = {"baselines": baselines(df), "candidates": [], "defaults": DEFAULTS}
    changes = [{"halflife": 4}, {"halflife": 16}, {"epa": "neutral"}, {"epa": "split"}, {"qb": "none"}, {"qb": "raw"}, {"qb": "both"},
               {"elo": "none"}, {"points": False}, {"home": "fixed"}, {"alpha": 100.0}, {"blend": 0.0}, {"blend": 1.0}]
    results, predictionsBy = {}, {}
    def score(params):
        key = json.dumps(params, sort_keys=True)
        if key in results:
            return
        predictions = walkForward(df, **params)
        result = evaluate(df, predictions, key)
        result["params"] = params
        report["candidates"].append(result)
        results[key], predictionsBy[key] = result["logLoss"], predictions
    score({})
    for change in changes:
        score(change)
    # combine every single change that beat the base model (the best option for each setting)
    base = results[json.dumps({})]
    combined = {}
    for setting in DEFAULTS:
        options = [(results[json.dumps(c, sort_keys=True)], c) for c in changes if list(c) == [setting]]
        good = [o for o in options if o[0] < base]
        if good:
            combined |= min(good, key=lambda o: o[0])[1]
    score(combined)
    bestKey = min(results, key=results.get)
    best = json.loads(bestKey)
    bestPredictions = predictionsBy[bestKey]
    print("Best:", best)
    report["best"] = best
    report["bestBySeason"] = [evaluate(df[df["season"] == s], bestPredictions[bestPredictions["season"] == s], f"season {s}") | {"season": s}
                              for s in seasons]
    report["completeSeason"] = complete
    report["tiers"] = tierReport(df, bestPredictions, complete)
    report["calibration"] = calibration(df, bestPredictions)
    writeLog(df, best, complete)
    final = GameModel(**best).fit(df)
    report["pointsCoefficients"] = dict(zip(final.features(df.head()).columns, final.points.coef_.round(4).tolist()))
    report["winCoefficients"] = dict(zip(final.features(df.head()).columns, final.win.coef_[0].round(4).tolist()))
    report["sigma"] = final.sigma
    with open(DATA + "game_model.pkl", "wb") as f:
        pickle.dump(final, f)
    with open(DATA + "model_report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)
    print("Saved model")
    return final
def writeLog(df, best, complete):
    # out of sample predictions for the latest finished season and the current one, for the game log on the webpage
    recent = pd.concat([GameModel(**best).fit(df[df["season"] < s]).predictGames(df[df["season"] == s]).assign(season=s)
                        for s in sorted(df.loc[df["season"] >= complete, "season"].unique())])
    home = homeGames(df).loc[recent.index]
    away = awayRows(df, home)
    log = recent.assign(date=home["date"].dt.strftime("%Y-%m-%d").values, week=home["week"].values, gameType=home["gameType"].values,
                        home=home["team"].values, away=away["team"].values, homeScore=home["pointsFor"].values,
                        awayScore=away["pointsFor"].values, spreadLine=home["spreadLine"].values, neutral=home["neutral"].values,
                        homeQb=home["qbName"].values, awayQb=away["qbName"].values)
    log.to_csv(DATA + "backtest_recent.csv", index=False)
    return recent
def refitModel():
    # daily update: refit the chosen model on every game without re-running the backtest, and add the newest games to the game log
    with open(DATA + "model_report.json") as f:
        report = json.load(f)
    df = loadFeatures()
    final = GameModel(**report["best"]).fit(df)
    with open(DATA + "game_model.pkl", "wb") as f:
        pickle.dump(final, f)
    # new games this season join the game log and the season table, still predicted by a model that never saw them
    recent = writeLog(df, report["best"], report["completeSeason"])
    current = int(df["season"].max())
    if current > report["completeSeason"]:
        row = evaluate(df[df["season"] == current], recent[recent["season"] == current], f"season {current}") | {"season": current}
        report["bestBySeason"] = [r for r in report["bestBySeason"] if r["season"] != current] + [row]
        with open(DATA + "model_report.json", "w") as f:
            json.dump(report, f, indent=2, default=float)
    print("Refit model on", len(df), "rows through", df["date"].max().date())
    return final
if __name__ == "__main__":
    from src.nflmodel.train import trainModel as run
    run()
