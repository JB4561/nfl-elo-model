"""
The Elo model.

Two halves, deliberately separated:
  1. PREDICTION -- ratings in, forecast out (home_field_elo,
     home_adjusted_diff, expected_score, predicted_margin).
  2. UPDATING   -- game result in, new ratings out (update_ratings).

Keeping the prediction half pure and stateless is not just tidiness. It makes
the no-look-ahead property easy to argue: none of those functions accepts a
score, so none of them can leak one. The backtest calls them BEFORE calling
update_ratings, and that ordering is the whole guarantee.

Every function here is pure -- no module state, no mutation, same inputs
always giving the same outputs.
"""

import math

from params import (
    ELO_PER_POINT,
    ELO_SCALE,
    HOME_FIELD_ADVANTAGE_POINTS,
    K_FACTOR,
    MOV_CORRECTION_B,
    MOV_CORRECTION_SLOPE,
    MOV_MULTIPLIER_MODE,
)


def home_field_elo(
    neutral_site: bool = False,
    hfa_points: float = HOME_FIELD_ADVANTAGE_POINTS,
) -> float:
    """Home-field advantage, converted from points into Elo units.

    Returns 0 at a neutral site. The data flags this with location ==
    "Neutral", which covers the Super Bowl and the ~42 international games in
    the backtest window, so that flag is used rather than special-casing the
    Super Bowl.

    `hfa_points` is an argument rather than a bare module read so that tuning
    can sweep it without mutating global state.
    """
    if neutral_site:
        return 0.0
    return hfa_points * ELO_PER_POINT


def home_adjusted_diff(
    home_rating: float,
    away_rating: float,
    neutral_site: bool = False,
    hfa_points: float = HOME_FIELD_ADVANTAGE_POINTS,
) -> float:
    """The rating gap from the HOME team's perspective, including home field.

    Positive => home team favored. This single number is the model's entire
    view of the matchup; both forecasts below are just two different ways of
    reading it.

    Note that home-field advantage is applied as a bonus to the home team's
    EFFECTIVE rating for this game only -- it never touches the stored rating.
    Home advantage is a property of the venue, not of the team.
    """
    return (home_rating + home_field_elo(neutral_site, hfa_points)) - away_rating


def expected_score(rating_diff: float) -> float:
    """Expected outcome (roughly, win probability) for the team `rating_diff` favors.

    The standard Elo logistic:

        E = 1 / (1 + 10 ** (-diff / ELO_SCALE))

    Two properties to note:
      * diff = 0   -> E = 0.5. Even teams, coin flip.
      * E(diff) + E(-diff) = 1. The two teams' expectations always sum to one,
        which is what makes the update rule below zero-sum.

    Calling this a "win probability" is a slight abuse of terms: the value we
    compare it against is a game OUTCOME (1 win / 0.5 tie / 0 loss), so E is
    really an expected outcome. For the NFL, where ties are ~0.3% of games,
    the two are near enough to identical.
    """
    return 1.0 / (1.0 + 10.0 ** (-rating_diff / ELO_SCALE))


def predicted_margin(rating_diff: float) -> float:
    """Convert a rating difference into a predicted point margin.

    This is the number we actually evaluate: it is directly comparable to the
    market's spread_line, which is also "expected home margin".

        margin = rating_diff / ELO_PER_POINT

    WHY LINEAR, when expected_score() above is a nonlinear logistic? Because
    the two answer different questions and are used in different places:

      * expected_score is used INSIDE the update rule, where the thing being
        predicted is a bounded outcome (0 to 1), so it needs a bounded S-curve.
      * predicted_margin is used for EVALUATION, where the thing being
        predicted is an unbounded point margin, and the empirical relationship
        between rating gap and average margin is close to a straight line.

    The linear form's known weakness is at the extremes: it will happily
    predict a 30-point margin for an enormous rating gap, where reality
    compresses (good teams rest starters, coast with a lead). Those matchups
    are rare enough in the NFL that v1 accepts this.
    """
    return rating_diff / ELO_PER_POINT


# ---------------------------------------------------------------------------
# The update rule
# ---------------------------------------------------------------------------

def update_ratings(
    home_rating: float,
    away_rating: float,
    home_score: float,
    away_score: float,
    neutral_site: bool = False,
    mov_mode: str = MOV_MULTIPLIER_MODE,
    hfa_points: float = HOME_FIELD_ADVANTAGE_POINTS,
) -> tuple[float, float]:
    """Apply one game's result to two ratings, returning the new pair.

    The heart of the model. Everything else -- the backtest, the metrics, the
    ATS comparison -- is scaffolding around this one function.

    Args:
        home_rating:  Pregame rating of the home team. The raw stored rating,
                      with NO home-field advantage in it; HFA is applied here,
                      per-game, via home_adjusted_diff().
        away_rating:  Pregame rating of the away team.
        home_score:   Final points scored by the home team.
        away_score:   Final points scored by the away team.
        neutral_site: True for Super Bowls and international games, which
                      zeroes the home-field bonus.
        mov_mode:     "538" or "none" -- see MOV_MULTIPLIER_MODE in params.py.
                      Taken as an argument rather than read from the module so
                      the backtest can run both variants in one process and
                      measure which is better.
        hfa_points:   Home-field advantage in points. Likewise an argument so
                      tuning can sweep it. Must match the value used for
                      prediction, or the ratings would be updated against a
                      different expectation than the one being scored.

    Returns:
        (new_home_rating, new_away_rating)

    HOW IT WORKS

    The rating change is driven entirely by SURPRISE -- the gap between what
    happened and what we expected:

        shift = K * mov_multiplier * (actual_outcome - expected_outcome)

    A result the ratings already predicted moves nothing; an upset moves a
    lot. Ratings only change when the model was wrong, which is what stops
    them drifting on information they already contain.

    `expected_outcome` includes home-field advantage, and that inclusion is
    load-bearing. Beating a peer at home is ordinary and should barely move
    the needle; beating that same peer on the road is genuine evidence. If
    HFA were left out of the expectation, home teams would accumulate rating
    across the league simply for playing at home.

    WHY IT IS ZERO-SUM

    The shift is computed once, from the home team's perspective, then added
    to the home rating and subtracted from the away rating. So rating is
    transferred between the two teams and never created:

        (new_home - home_rating) == -(new_away - away_rating)

    The league mean therefore never drifts, which is what keeps 1500
    permanently meaningful as "average". This falls out of expected_score()'s
    symmetry, E(d) + E(-d) = 1: the away team's surprise is exactly the
    negative of the home team's, so one calculation serves both.

    THE MARGIN-OF-VICTORY MULTIPLIER

    "none" -> 1.0. Every win counts the same regardless of scoreline. Discards
        real information, but it is the honest baseline that makes "does MOV
        weighting help?" a question we can answer with a number.

    "538" -> log(|margin| + 1) * (B / (SLOPE * winner_elo_diff + B))

        The log gives diminishing returns: winning by 28 rather than 14 is
        informative, but not twice as informative. Empirically each DOUBLING
        of the margin adds a roughly constant amount of rating movement.

        The second factor is an autocorrelation correction, and it is the
        subtle part. `winner_elo_diff` is the winner's home-adjusted rating
        minus the loser's, and its SIGN carries the meaning: positive when the
        favorite won (denominator grows, multiplier shrinks -- we already knew
        they were better, so a blowout is weak evidence), negative when the
        underdog won (denominator shrinks, multiplier grows -- genuinely
        surprising). Without it, strong teams blowing out weak teams would
        compound their own ratings upward in a feedback loop.

        Concretely: a 1700 team beating a 1300 team by 14 at home moves
        ratings 2.9 points; the same 14-point margin the other way moves them
        64.3, over twenty times as much.

    KNOWN EDGE CASES

    * TIES. Scored as half-wins (outcome 0.5). Under "538" a tie has margin 0,
      so log(0 + 1) = 0 and NOTHING MOVES -- even a tie between badly
      mismatched teams, which is genuinely informative. Under "none" that same
      tie does move ratings. The tie rule therefore means something different
      in each mode. Only 9 games in 2015-2024, so it cannot shift the metrics,
      but it is a deliberate acceptance rather than an oversight.

    * A POLE IN THE FORMULA. If winner_elo_diff approaches -B/SLOPE (-2200 at
      the default constants), the denominator approaches zero and the
      multiplier explodes. Real NFL rating gaps come nowhere near this, so it
      is left unguarded -- but it is a property of the formula worth knowing
      about rather than discovering.

    Reference: FiveThirtyEight, "How Our NFL Predictions Work".
    """
    actual_outcome = 0.5 if home_score == away_score else (1.0 if home_score > away_score else 0.0)
    expected_outcome = expected_score(home_adjusted_diff(home_rating, away_rating, neutral_site, hfa_points))

    margin = home_score - away_score
    diff = home_adjusted_diff(home_rating, away_rating, neutral_site, hfa_points)
    winner_elo_diff = diff if home_score > away_score else -diff

    multiplier = 1.0 if mov_mode == "none" else math.log(abs(margin) + 1) * (MOV_CORRECTION_B / (MOV_CORRECTION_SLOPE * winner_elo_diff + MOV_CORRECTION_B))

    shift = K_FACTOR * multiplier * (actual_outcome - expected_outcome)
    new_home = home_rating + shift
    new_away = away_rating - shift
    return (new_home, new_away)

