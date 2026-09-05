"""
The chronological, no-look-ahead backtest.

This is the piece that makes or breaks the whole project. The single most
common way a sports/quant backtest is silently invalidated is by letting
information from a game influence the prediction OF that game (or of an
earlier one). Everything below is arranged so that cannot happen.

The guarantee rests on three things, in order:

  1. The games arrive already sorted by kickoff (data.load_games).
  2. For each game we RECORD THE PREDICTION FIRST, then apply the result.
     The prediction functions are pure and never receive a score, so they
     physically cannot see the outcome.
  3. Ratings live in one dict that is only ever written by step 2's update.
     At the moment game N is predicted, that dict contains contributions from
     games 1..N-1 and nothing else.

Season carryover is applied at the season boundary, before the first game of
the new season is predicted, so it too uses only prior information.
"""

import pandas as pd

from elo import home_adjusted_diff, predicted_margin, update_ratings
from params import (
    BASE_RATING,
    HOME_FIELD_ADVANTAGE_POINTS,
    MOV_MULTIPLIER_MODE,
    SEASON_CARRYOVER,
)


def apply_season_carryover(
    ratings: dict,
    carryover: float = SEASON_CARRYOVER,
    base_rating: float = BASE_RATING,
) -> dict:
    """Regress every rating partway toward the league mean between seasons.

        new = base + carryover * (old - base)

    carryover = 1.0 keeps the ratings untouched; 0.0 resets everyone to base.
    Rosters, coaches and schemes turn over every offseason, so a team's Elo
    at the end of one season overstates what we know about it at the start of
    the next -- but it is far from worthless, so we shrink rather than reset.

    This is applied to a COPY, so a caller cannot accidentally mutate the
    ratings it passed in.
    """
    return {
        team: base_rating + carryover * (rating - base_rating)
        for team, rating in ratings.items()
    }


def run_backtest(
    games: pd.DataFrame,
    mov_mode: str = MOV_MULTIPLIER_MODE,
    carryover: float = SEASON_CARRYOVER,
    base_rating: float = BASE_RATING,
    hfa_points: float = HOME_FIELD_ADVANTAGE_POINTS,
) -> tuple[pd.DataFrame, dict]:
    """Walk every game in order, predicting before updating.

    Returns:
      predictions -- one row per game: the pregame ratings, what the model
                     predicted, what actually happened, and the market line.
      ratings     -- the final rating dict, after the last game.

    The returned frame is the raw material for every metric and plot; no
    evaluation logic lives here, so this function has exactly one job.
    """
    ratings: dict[str, float] = {}
    rows = []

    current_season = None

    for game in games.itertuples(index=False):
        # ---- Season boundary --------------------------------------------
        # Games are ordered by date, and a season's playoffs (played in
        # Jan/Feb) still carry the PREVIOUS season's label -- so the season
        # column is non-decreasing through the sorted frame. We assert that
        # rather than trust it, because if the ordering were ever wrong this
        # is where it would corrupt the ratings silently.
        if current_season is not None and game.season < current_season:
            raise ValueError(
                f"Games are not in chronological season order: saw season "
                f"{game.season} after {current_season}. The no-look-ahead "
                f"guarantee depends on this ordering."
            )

        if current_season is not None and game.season != current_season:
            # Regress BEFORE the first game of the new season is predicted,
            # so the new season's very first prediction already reflects it.
            ratings = apply_season_carryover(ratings, carryover, base_rating)
        current_season = game.season

        # ---- Predict (no score is visible to any of this) ----------------
        # A team we have never seen starts at the base rating. In practice
        # this only fires in week 1 of the first season.
        home_rating = ratings.get(game.home_team, base_rating)
        away_rating = ratings.get(game.away_team, base_rating)

        neutral_site = game.location == "Neutral"
        # The SAME hfa_points feeds prediction here and the rating update
        # below. Letting them differ would score the model against one
        # expectation while training it against another.
        rating_diff = home_adjusted_diff(home_rating, away_rating, neutral_site, hfa_points)

        rows.append(
            {
                "game_id": game.game_id,
                "season": game.season,
                "game_type": game.game_type,
                "week": game.week,
                "gameday": game.gameday,
                "home_team": game.home_team,
                "away_team": game.away_team,
                "neutral_site": neutral_site,
                # Pregame state -- kept so any prediction can be audited and
                # recomputed by hand from the ratings that produced it.
                "home_rating_pre": home_rating,
                "away_rating_pre": away_rating,
                "rating_diff": rating_diff,
                # The model's forecast, in the same units and sign convention
                # as both `actual_margin` and `spread_line`: home perspective.
                "pred_margin": predicted_margin(rating_diff),
                # Truth and market price. Recorded for evaluation only -- note
                # they are written AFTER pred_margin, and nothing above reads
                # them.
                "actual_margin": game.result,
                "spread_line": game.spread_line,
            }
        )

        # ---- Update (only now does the score enter the system) -----------
        new_home, new_away = update_ratings(
            home_rating,
            away_rating,
            game.home_score,
            game.away_score,
            neutral_site=neutral_site,
            mov_mode=mov_mode,
            hfa_points=hfa_points,
        )
        ratings[game.home_team] = new_home
        ratings[game.away_team] = new_away

    return pd.DataFrame(rows), ratings


if __name__ == "__main__":
    from data import load_games

    games = load_games()
    preds, final_ratings = run_backtest(games)

    print(f"Backtested {len(preds)} games.\n")
    print("Final ratings, top 10:")
    for team, rating in sorted(final_ratings.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {team:4s} {rating:7.1f}")
