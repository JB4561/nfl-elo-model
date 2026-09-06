"""
Live prediction for upcoming, unplayed games.

The backtest and this script answer two different questions, and the difference
drives everything below.

    BACKTEST          predicts games that have already been played, so every
                      game is predict-then-update: the score exists, so the
                      ratings can learn from it immediately.

    LIVE PREDICTION   targets games that have NOT been played. There is no
                      score to learn from, so it is predict-only. Ratings are
                      built from completed games, frozen, and then applied to
                      the upcoming slate without any update step.

Concretely: completed games move the ratings, upcoming games only read them.

Output is a JSON file for a separate website project to consume, so the record
shape is part of the contract rather than an implementation detail.
"""

import os

import pandas as pd

# Reused rather than reimplemented: the download/cache, the column subset, and
# the relocated-franchise mapping are the same for live data as for historical
# data, and duplicating them here would let the two copies drift apart.
from data import COLUMNS, FIRST_SEASON, FRANCHISE_MAP, download_games

# The rating machinery is imported wholesale. Nothing about Elo changes when
# the target game happens to lie in the future -- only whether an update
# follows the prediction -- so re-deriving any of this would risk the live
# numbers quietly diverging from the backtested ones.
from backtest import apply_season_carryover, run_backtest
from elo import home_adjusted_diff, predicted_margin
from params import BASE_RATING, HOME_FIELD_ADVANTAGE_POINTS, SEASON_CARRYOVER

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions.json")


def load_all_games(first_season: int = FIRST_SEASON, force_refresh: bool = False) -> pd.DataFrame:
    """Load every game from `first_season` onward, played AND scheduled.

    This deliberately differs from data.load_games in two ways, and both are
    required for the live path:

      1. NO UPPER SEASON BOUND. The backtest stops at a fixed final season so
         its evaluation window is reproducible. Live ratings must instead
         include every completed season up to today, or they would be stale by
         however long ago that window ended.

      2. UNPLAYED GAMES ARE KEPT. data.load_games drops rows with null scores,
         because feeding a null score to a rating update is meaningless. Here
         those same rows are the prediction targets, so they are retained and
         separated out by split_completed_upcoming() below.

    Set force_refresh=True to re-download. Worth doing before a real slate:
    the schedule is fixed well in advance, but market spreads populate closer
    to kickoff, so a stale cache can mean missing lines on later weeks.
    """
    df = download_games(force_refresh=force_refresh)

    df = df[COLUMNS].copy()
    df = df[df["season"] >= first_season]

    # Same franchise consolidation the backtest uses, so a team's rating
    # history is continuous across a relocation.
    df["home_team"] = df["home_team"].replace(FRANCHISE_MAP)
    df["away_team"] = df["away_team"].replace(FRANCHISE_MAP)

    df["gameday"] = pd.to_datetime(df["gameday"])

    # Chronological order still matters: the completed slice is walked in this
    # order to build ratings, exactly as in the backtest.
    df = df.sort_values(["gameday", "gametime", "game_id"], kind="mergesort").reset_index(drop=True)
    return df


def split_completed_upcoming(games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate games that have been played from games that have not.

    A null score is the ONLY reliable marker of an unplayed game in this data.
    Two fields that look like they should help do not:

      * `game_type` marks the round (REG / WC / DIV / CON / SB), not the
        status. An unplayed regular-season game is labelled REG exactly like a
        completed one.
      * `week` is likewise just the week number, present on both.

    `gameday` and `gametime` ARE populated for unplayed games -- the schedule
    is known in advance -- which is what allows the upcoming slate to be
    ordered and filtered by date.

    Deriving the split from scores rather than from dates also means the
    script cannot be confused by a run on a day when games are in progress:
    what matters is whether a result exists, not whether kickoff has passed.
    """
    played = games["home_score"].notna() & games["away_score"].notna()
    completed = games[played].reset_index(drop=True)
    upcoming = games[~played].reset_index(drop=True)
    return completed, upcoming


def next_slate(upcoming: pd.DataFrame) -> tuple[int, int]:
    """The (season, week) of the earliest unplayed games.

    Taken as the minimum season among unplayed games, then the minimum week
    within that season, rather than simply reading the first row by date. The
    two agree in normal circumstances, but a postponed or rescheduled game can
    put a later week's kickoff earlier on the calendar; keying off the week
    numbers is robust to that.

    Detecting this rather than hardcoding it means the script keeps working as
    the season progresses -- each run naturally targets whatever slate is next.
    """
    if upcoming.empty:
        raise ValueError(
            "No unplayed games found. Either the data needs refreshing "
            "(force_refresh=True) or the season is complete."
        )
    season = int(upcoming["season"].min())
    week = int(upcoming.loc[upcoming["season"] == season, "week"].min())
    return season, week


def select_slate(upcoming: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """The unplayed games for one specific season and week.

    The assertion is cheap insurance: if a played game ever leaked into the
    upcoming set, it would be silently "predicted" and reported as a forecast
    for a game whose result is already known.
    """
    slate = upcoming[(upcoming["season"] == season) & (upcoming["week"] == week)].copy()
    assert slate["home_score"].isna().all(), "A completed game leaked into the prediction slate"
    return slate.sort_values(["gameday", "gametime", "game_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Ratings
# ---------------------------------------------------------------------------

def current_ratings(completed: pd.DataFrame) -> tuple[dict, int]:
    """Walk every completed game to produce today's ratings.

    This is the ordinary backtest loop, run purely for its side effect on the
    ratings; the per-game prediction frame it also returns is discarded here.
    Reusing it rather than writing a shorter "just update" loop guarantees the
    live ratings are built by exactly the same code path, with the same
    ordering and the same season-boundary handling, as the evaluated ones.

    Returns the ratings and the last season that contributed to them.
    """
    _, ratings = run_backtest(completed)
    return ratings, int(completed["season"].max())


def ratings_for_season(
    ratings: dict,
    last_completed_season: int,
    target_season: int,
    carryover: float = SEASON_CARRYOVER,
) -> tuple[dict, int]:
    """Regress ratings forward across any season boundaries not yet crossed.

    THIS IS THE ONE PIECE OF LOGIC THE BACKTEST DOES NOT ALREADY PERFORM, and
    getting it wrong in either direction is easy.

    run_backtest applies carryover *at* each season boundary it crosses -- but
    only when it encounters the first game of the next season. After the final
    completed game there is no next game, so the boundary between the last
    completed season and an upcoming one is never triggered. Predicting the
    2026 opener from ratings that still end at the 2025 Super Bowl would treat
    every team as identical to its end-of-last-season self, ignoring the draft,
    free agency and retirements the carryover exists to represent.

    Hence the count is derived, not assumed:

        boundaries = target_season - last_completed_season

    which handles all three cases correctly:

      * 1  -- the normal preseason case. Last completed season is 2025, the
              target is 2026, so exactly one regression is applied.
      * 0  -- a MID-SEASON run. Once week 1 of 2026 has been played, the
              completed set contains 2026 games, run_backtest has already
              applied the 2025->2026 boundary internally, and applying another
              here would regress every team toward 1500 twice. Zero boundaries
              means this correctly does nothing.
      * 2+ -- an entire season missing from the data. Unlikely, but the
              regression should compound rather than be applied once.

    Returns the forwarded ratings and how many regressions were applied, so
    the caller can report it rather than have it happen invisibly.
    """
    boundaries = target_season - last_completed_season
    if boundaries < 0:
        raise ValueError(
            f"Target season {target_season} precedes the last completed season "
            f"{last_completed_season}; ratings cannot be regressed backwards."
        )

    for _ in range(boundaries):
        ratings = apply_season_carryover(ratings, carryover)
    return ratings, boundaries


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_games(
    ratings: dict,
    slate: pd.DataFrame,
    hfa_points: float = HOME_FIELD_ADVANTAGE_POINTS,
    base_rating: float = BASE_RATING,
) -> pd.DataFrame:
    """Predict a slate of unplayed games. Predict only -- nothing is updated.

    The rating lookup, the home-field adjustment and the margin conversion are
    identical to the backtest's prediction step. The difference is what is
    ABSENT: there is no call to update_ratings, because these games have no
    result to learn from. The ratings passed in are read by every game in the
    slate and modified by none of them.

    A consequence worth being explicit about: every game in a slate is
    predicted from the SAME ratings. Thursday's result does not inform
    Sunday's prediction, because at the time of writing it has not happened.
    Re-running the script after Thursday's game moves it into the completed
    set and the remaining predictions shift accordingly.

    `pred_margin` is the expected home margin, positive when the home team is
    favoured -- the same units and sign convention as `spread_line`, so the
    two are directly comparable.
    """
    rows = []
    for game in slate.itertuples(index=False):
        # Falling back to base_rating would mean an unrecognised team code,
        # which in live data signals a franchise rename the mapping has not
        # caught up with -- worth failing loudly rather than silently rating
        # an unknown team as exactly average.
        for team in (game.home_team, game.away_team):
            if team not in ratings:
                raise KeyError(
                    f"No rating for team {team!r} in game {game.game_id}. This "
                    f"usually means a new or renamed team code that FRANCHISE_MAP "
                    f"in data.py does not yet handle."
                )

        home_rating = ratings[game.home_team]
        away_rating = ratings[game.away_team]
        neutral_site = game.location == "Neutral"
        rating_diff = home_adjusted_diff(home_rating, away_rating, neutral_site, hfa_points)

        rows.append(
            {
                "game_id": game.game_id,
                "season": int(game.season),
                "week": int(game.week),
                "gameday": game.gameday,
                "away_team": game.away_team,
                "home_team": game.home_team,
                "neutral_site": neutral_site,
                "home_rating": home_rating,
                "away_rating": away_rating,
                "pred_margin": predicted_margin(rating_diff),
                "spread_line": game.spread_line,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_predictions(preds: pd.DataFrame, path: str = OUTPUT_PATH) -> str:
    """Write predictions as a JSON array of records.

    The output is consumed by a separate website project, so the shape is a
    contract rather than an internal detail:

      game_id       stable key for a game
      season, week  which slate this is
      gameday       ISO date string. Written explicitly rather than left as a
                    timestamp, which pandas would otherwise serialise as epoch
                    milliseconds -- technically fine, needlessly awkward to read.
      away_team     visiting team code
      home_team     home team code
      pred_margin   model's expected home margin, positive = home favoured
      spread_line   market's expected home margin, same convention, or null

    `spread_line` is ALWAYS present, carrying null when the market has not
    posted a line yet, rather than the key being omitted. Lines populate closer
    to kickoff, so for later weeks it is frequently absent; a stable schema
    means the site tests one field for null instead of handling two record
    shapes.

    Ratings are not written out. They are an internal quantity on an arbitrary
    scale, and publishing them invites the site to display a number that means
    nothing without the whole model behind it.
    """
    out = preds[[
        "game_id", "season", "week", "gameday",
        "away_team", "home_team", "pred_margin", "spread_line",
    ]].copy()

    out["gameday"] = out["gameday"].dt.strftime("%Y-%m-%d")
    out["pred_margin"] = out["pred_margin"].round(2)

    out.to_json(path, orient="records", indent=2)
    return path


if __name__ == "__main__":
    games = load_all_games()
    completed, upcoming = split_completed_upcoming(games)
    season, week = next_slate(upcoming)
    slate = select_slate(upcoming, season, week)

    ratings, last_season = current_ratings(completed)
    ratings, regressions = ratings_for_season(ratings, last_season, season)
    preds = predict_games(ratings, slate)
    path = write_predictions(preds)

    print(f"Ratings built from {len(completed)} completed games through season {last_season}.")
    print(f"Season carryover applied {regressions}x for target season {season}.")
    print(f"\nSeason {season}, week {week} -- {len(preds)} games\n")

    show = preds.copy()
    show["matchup"] = show["away_team"] + " @ " + show["home_team"]
    show["edge"] = show["pred_margin"] - show["spread_line"]
    show["site"] = show["neutral_site"].map({True: "neutral", False: ""})
    print(show[["gameday", "matchup", "site", "pred_margin", "spread_line", "edge"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.2f}",
                     formatters={"gameday": lambda d: d.strftime("%a %d %b")}))

    print(f"\nWrote {os.path.relpath(path)}")
