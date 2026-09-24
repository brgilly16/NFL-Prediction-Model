import io
import os
import datetime
import requests
import numpy as np
import pandas as pd
# downloads nflverse data: the schedule (scores, betting lines, starting QBs) and play-by-play reduced to per game team and QB stats
# play-by-play is large, so each season is boiled down to two small files that are kept in the repo:
#   data/seasons/team_games_{season}.csv   one row per team per game (offense EPA and plays, the defense side is the opponent's row)
#   data/seasons/qb_games_{season}.csv     one row per passer per game (dropbacks, EPA, CPOE)
# finished seasons are only downloaded once, the current season is refreshed every run
DATA = "data/"
RAW = DATA + "raw/"
SEASONS = DATA + "seasons/"
FIRST_PBP_SEASON = 1999
GAMES_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
RELEASE = "https://github.com/nflverse/nflverse-data/releases/download/"
PBP_COLUMNS = ["game_id", "season", "week", "game_date", "posteam", "defteam", "pass", "rush", "epa", "success", "wp",
               "qb_dropback", "passer_id", "passer", "qb_epa", "cpoe", "interception", "fumble_lost", "special", "qb_kneel", "qb_spike"]
def fetch(url):
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            return response.content
        except requests.RequestException as error:
            print("  retrying", url, error)
    raise RuntimeError("Could not download " + url)
def currentSeason():
    # the NFL season starts in September, so January to August still belong to the previous season
    today = datetime.date.today()
    return today.year if today.month >= 9 else today.year - 1
def downloadGames():
    os.makedirs(RAW, exist_ok=True)
    open(RAW + "games.csv", "wb").write(fetch(GAMES_URL))
    print("Downloaded schedule")
def loadGames():
    return pd.read_csv(RAW + "games.csv")
def reduceSeason(pbp):
    # offense per game: EPA and success on every run and pass (scrambles and sacks count as passes), kneels and spikes left out
    plays = pbp[((pbp["pass"] == 1) | (pbp["rush"] == 1)) & pbp["epa"].notna() & (pbp["qb_kneel"] != 1) & (pbp["qb_spike"] != 1)].copy()
    # "neutral" plays: win probability between 10% and 90%, so garbage time does not count
    plays["neutral"] = plays["wp"].between(0.1, 0.9).astype(int)
    plays["turnover"] = plays["interception"].fillna(0) + plays["fumble_lost"].fillna(0)
    for kind in ["pass", "rush"]:
        plays[kind + "EPA"] = plays["epa"] * plays[kind]
    plays["neutralEPA"] = plays["epa"] * plays["neutral"]
    team = plays.groupby(["game_id", "posteam"]).agg(
        plays=("epa", "size"), epa=("epa", "sum"), success=("success", "sum"),
        passPlays=("pass", "sum"), passEPA=("passEPA", "sum"), rushPlays=("rush", "sum"), rushEPA=("rushEPA", "sum"),
        neutralPlays=("neutral", "sum"), neutralEPA=("neutralEPA", "sum"), turnovers=("turnover", "sum")
    ).reset_index().rename(columns={"posteam": "team"})
    # QBs: every dropback (passes, sacks, scrambles) credited to the passer
    drop = plays[(plays["qb_dropback"] == 1) & plays["passer_id"].notna()]
    qb = drop.groupby(["game_id", "posteam", "passer_id"]).agg(
        name=("passer", "last"), dropbacks=("qb_epa", "size"), epa=("qb_epa", "sum"),
        cpoe=("cpoe", "sum"), cpoePlays=("cpoe", "count")
    ).reset_index().rename(columns={"posteam": "team", "passer_id": "qbId"})
    return team, qb
def downloadSeason(season):
    os.makedirs(SEASONS, exist_ok=True)
    pbp = pd.read_parquet(io.BytesIO(fetch(f"{RELEASE}pbp/play_by_play_{season}.parquet")), columns=PBP_COLUMNS)
    team, qb = reduceSeason(pbp)
    team.to_csv(f"{SEASONS}team_games_{season}.csv", index=False)
    qb.round(4).to_csv(f"{SEASONS}qb_games_{season}.csv", index=False)
    print(f"Play-by-play {season}: {len(team) // 2} games")
def downloadRosters(season):
    # current rosters, for each team's QB list and to match QB names in the schedule to player ids
    columns = ["season", "team", "position", "full_name", "gsis_id", "status", "depth_chart_position", "years_exp", "rookie_year"]
    for year in (season, season - 1):
        try:
            roster = pd.read_parquet(io.BytesIO(fetch(f"{RELEASE}rosters/roster_{year}.parquet")))
            roster[[c for c in columns if c in roster.columns]].to_csv(RAW + "rosters.csv", index=False)
            print("Downloaded rosters", year)
            return
        except RuntimeError:
            continue
def downloadAll(refreshAll=False):
    downloadGames()
    season = currentSeason()
    games = loadGames()
    played = games[games["home_score"].notna()]
    latest = int(played["season"].max()) if len(played) else season - 1
    for year in range(FIRST_PBP_SEASON, latest + 1):
        # past seasons never change, the latest one with games is refreshed every run
        if refreshAll or year >= latest or not os.path.exists(f"{SEASONS}team_games_{year}.csv"):
            downloadSeason(year)
    downloadRosters(season)
def loadSeasons(kind):
    files = sorted(f for f in os.listdir(SEASONS) if f.startswith(kind + "_"))
    return pd.concat([pd.read_csv(SEASONS + f).assign(season=int(f[-8:-4])) for f in files], ignore_index=True)
if __name__ == "__main__":
    downloadAll()
