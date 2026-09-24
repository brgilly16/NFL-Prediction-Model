# NFL PowerScore

NFL game predictions, power rankings and quarterback ratings, built the same way as the NHL PowerScore project.

The model is trained on every game since 2001 using [nflverse](https://github.com/nflverse) play-by-play, schedules and betting lines.
Each team's points are predicted from pregame features only:

- offense and defense EPA per play, recency weighted and carried across seasons
- the starting QB's rating (recency-weighted EPA per dropback) and how he compares with the team's usual starter
- Elo, recent points for and against, days of rest, home field and the league scoring level

A logistic win model on the same features is blended with the win chance implied by the points model.
Every version is scored with a walk-forward backtest (each season from 2010 is predicted by a model trained only on earlier seasons),
and the version with the lowest log loss is used. The Vegas spread is reported as the benchmark to beat.

## Running it

```
pip install -r requirements.txt
python -m src.nflmodel.run            # download data, build features, backtest every version, train, export the site
python -m src.nflmodel.run --daily    # sync new games, refit the chosen version, export (what the GitHub Action runs)
python -m http.server 5001 --directory webapp
```

The website in `webapp/` is static: `data.js` holds the model weights and every team's and QB's current state,
and `app.js` runs the model in the browser.
