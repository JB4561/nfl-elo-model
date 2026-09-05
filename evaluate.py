"""
Evaluation metrics, and the MOV-multiplier comparison.

METRIC CHOICE: MARGIN ERROR, NOT ATS
------------------------------------
Whether MOV weighting improves the ratings is measured with MAE/RMSE against
the actual margin. That is the model's direct target, every game contributes a
real-valued error, and 2743 games gives a stable estimate.

ATS win rate is a far noisier statistic. It discards magnitude, keeping only
which side of the line a game landed on, so three test seasons amount to
roughly 800 near-coin-flips -- selecting between two model variants on a
number that noisy would mostly select the luckier one. ATS is the project's
headline result, but it is the wrong instrument for model selection, and it is
evaluated separately.

TRAIN/TEST DISCIPLINE
---------------------
Every metric is reported on both train and test, for insight into how the
model generalises. Model-selection decisions are made on TRAIN only. Selecting
the MOV mode by its TEST performance would make test a tuning input rather
than a clean holdout, leaving the final ATS figure quietly optimistic -- the
standard way a backtest is invalidated. Test exists to confirm that a choice
made on train generalises, and is reported once.
"""

import math

import numpy as np
import pandas as pd

from backtest import run_backtest
from params import (
    ATS_ODDS_AMERICAN,
    BURN_IN_SEASONS,
    TEST_SEASONS,
    TRAIN_SEASONS,
)


def select_seasons(preds: pd.DataFrame, span: tuple[int, int], burn_in: int = 0) -> pd.DataFrame:
    """Filter predictions to a season range, optionally skipping a warm-up.

    `burn_in` drops that many seasons from the START of the span. Those games
    still ran in the backtest (they are what warms the ratings up) -- they are
    only excluded from the reported numbers, so the model is not judged on
    predictions it made while every team was still sitting at 1500.
    """
    first, last = span
    return preds[(preds["season"] >= first + burn_in) & (preds["season"] <= last)]


def margin_metrics(preds: pd.DataFrame) -> dict:
    """Accuracy of the predicted margin, plus straight-up win/loss accuracy."""
    error = preds["pred_margin"] - preds["actual_margin"]

    # Straight-up accuracy: did the model favour the team that actually won?
    # Games that ended in a tie are excluded -- there is no correct winner to
    # have picked, so scoring them either way would be arbitrary. They are
    # reported separately so the exclusion is visible rather than hidden.
    decided = preds[preds["actual_margin"] != 0]
    correct = np.sign(decided["pred_margin"]) == np.sign(decided["actual_margin"])

    return {
        "n": len(preds),
        "mae": error.abs().mean(),
        "rmse": np.sqrt((error**2).mean()),
        "bias": error.mean(),  # systematic lean toward home (+) or away (-)
        "su_accuracy": correct.mean(),
        "n_ties_excluded": len(preds) - len(decided),
    }


def market_baseline(preds: pd.DataFrame) -> dict:
    """The same metrics for the MARKET's prediction instead of the model's.

    This is the reference point that makes the model's MAE mean anything. An
    MAE of 10 points sounds bad in isolation; what matters is how it compares
    to the number the betting market achieves on the identical games. The
    spread is a genuinely hard benchmark, and beating it is not expected.
    """
    market = preds.rename(columns={"spread_line": "pred_margin", "pred_margin": "_model"})
    return margin_metrics(market)


def compare_mov_modes(games: pd.DataFrame, modes=("none", "538")) -> pd.DataFrame:
    """Run the full backtest once per MOV mode and tabulate the metrics.

    Each mode gets its own independent run over all seasons -- the ratings
    differ from the very first game, so they cannot share state.
    """
    rows = []
    for mode in modes:
        preds, _ = run_backtest(games, mov_mode=mode)
        for label, span, burn_in in (
            ("train", TRAIN_SEASONS, BURN_IN_SEASONS),
            ("test", TEST_SEASONS, 0),
        ):
            subset = select_seasons(preds, span, burn_in)
            rows.append({"mov_mode": mode, "split": label, **margin_metrics(subset)})

    # Market baseline on the identical rows, for reference. The market's
    # numbers do not depend on mov_mode, so one run is enough.
    preds, _ = run_backtest(games, mov_mode=modes[0])
    for label, span, burn_in in (
        ("train", TRAIN_SEASONS, BURN_IN_SEASONS),
        ("test", TEST_SEASONS, 0),
    ):
        subset = select_seasons(preds, span, burn_in)
        rows.append({"mov_mode": "MARKET", "split": label, **market_baseline(subset)})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Against-the-spread evaluation
# ---------------------------------------------------------------------------

def breakeven_win_rate(odds: int = ATS_ODDS_AMERICAN) -> float:
    """Win rate required to break even at the given American odds.

    At -110 a bettor risks 110 to win 100, so 110 / 210 = 52.38% of bets must
    win merely to finish level. This is the bar an ATS result has to clear;
    beating 50% means nothing, because 50% loses money.
    """
    risk = abs(odds)
    return risk / (risk + 100.0)


def ats_record(preds: pd.DataFrame) -> dict:
    """The model's against-the-spread record.

    THE PICK. Both `pred_margin` and `spread_line` are expected home margins in
    the same units and sign convention, so the model's disagreement with the
    market is simply their difference:

        edge = pred_margin - spread_line

    A positive edge means the model thinks the home team beats the line, so it
    takes the home side; negative takes the away side. |edge| is how strongly
    the model disagrees, and doubles as its confidence.

    SETTLEMENT. A home pick wins if actual_margin > spread_line, an away pick
    if actual_margin < spread_line. Games landing exactly on the line are
    PUSHES: the stake is returned and nobody wins, so they are excluded from
    the win rate and reported separately. This matches how sportsbooks settle
    and is the basis on which the 52.38% breakeven figure is derived, keeping
    the comparison valid.

    SIGNIFICANCE. With a few hundred bets, an ATS record is a noisy statistic,
    so a standard error and a p-value against the breakeven rate are reported
    alongside. Without them a 53% record reads as an edge when it is routinely
    what a coin flip produces at this sample size.
    """
    edge = preds["pred_margin"] - preds["spread_line"]

    # A game where the model exactly matches the line offers no side to take.
    # (Empirically this never happens, but leaving it implicit would silently
    # score a non-existent pick as a loss.)
    has_pick = edge != 0
    bets = preds[has_pick]
    edge = edge[has_pick]

    picked_home = edge > 0
    push = bets["actual_margin"] == bets["spread_line"]
    covered = np.where(
        picked_home,
        bets["actual_margin"] > bets["spread_line"],
        bets["actual_margin"] < bets["spread_line"],
    )

    wins = int((covered & ~push).sum())
    losses = int((~covered & ~push).sum())
    pushes = int(push.sum())
    decided = wins + losses

    win_rate = wins / decided if decided else float("nan")
    breakeven = breakeven_win_rate()

    # Standard error under the null that each bet is a coin flip, and a
    # two-sided p-value for the observed rate against the breakeven rate.
    se = math.sqrt(0.25 / decided) if decided else float("nan")
    z = (win_rate - breakeven) / se if decided else float("nan")
    p_value = math.erfc(abs(z) / math.sqrt(2.0)) if decided else float("nan")

    return {
        "n_games": len(preds),
        "n_bets": decided,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": win_rate,
        "breakeven": breakeven,
        "edge_vs_breakeven": win_rate - breakeven,
        "beats_breakeven": bool(win_rate > breakeven),
        "std_error": se,
        "z_vs_breakeven": z,
        "p_value": p_value,
        "mean_abs_edge": float(edge.abs().mean()),
    }


def ats_by_confidence(preds: pd.DataFrame, cuts=(0.0, 1.0, 2.0, 3.0, 6.0, np.inf)) -> pd.DataFrame:
    """ATS record bucketed by how strongly the model disagrees with the market.

    The question this answers: does the model do BETTER when it disagrees more?
    If the model held real information the market lacked, its large
    disagreements should be its most profitable bets. A flat or inverted
    pattern says the disagreements are mostly noise -- which is the expected
    result against an efficient price.
    """
    edge = (preds["pred_margin"] - preds["spread_line"]).abs()
    rows = []
    for low, high in zip(cuts[:-1], cuts[1:]):
        subset = preds[(edge >= low) & (edge < high)]
        if subset.empty:
            continue
        label = f"{low:.0f}-{high:.0f}" if np.isfinite(high) else f"{low:.0f}+"
        rec = ats_record(subset)
        rows.append({
            "edge_bucket": label,
            "n_bets": rec["n_bets"],
            "win_rate": rec["win_rate"],
            "vs_breakeven": rec["edge_vs_breakeven"],
            # Reported so a small bucket cannot be misread as a signal: a
            # 60% win rate on 50 bets carries a +/-7pp standard error and is
            # entirely consistent with a coin flip.
            "std_error": rec["std_error"],
        })
    return pd.DataFrame(rows)


def ats_by_season(preds: pd.DataFrame) -> pd.DataFrame:
    """ATS record season by season, to show how much it bounces around."""
    rows = []
    for season, subset in preds.groupby("season"):
        rec = ats_record(subset)
        rows.append({
            "season": season,
            "n_bets": rec["n_bets"],
            "wins": rec["wins"],
            "losses": rec["losses"],
            "pushes": rec["pushes"],
            "win_rate": rec["win_rate"],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from data import load_games

    games = load_games()
    pd.set_option("display.width", 200)
    fmt = lambda v: f"{v:.4f}"

    print("=" * 78)
    print("MOV MULTIPLIER COMPARISON")
    print("=" * 78)
    print(compare_mov_modes(games).to_string(index=False, float_format=fmt))
    print()
    print("Model selection is made on TRAIN; TEST confirms generalisation.")

    preds, _ = run_backtest(games)
    splits = (
        ("TRAIN 2016-2021", select_seasons(preds, TRAIN_SEASONS, BURN_IN_SEASONS)),
        ("TEST 2022-2024", select_seasons(preds, TEST_SEASONS, 0)),
    )

    print()
    print("=" * 78)
    print("AGAINST THE SPREAD")
    print("=" * 78)
    for label, subset in splits:
        r = ats_record(subset)
        verdict = "BEATS breakeven" if r["beats_breakeven"] else "does NOT beat breakeven"
        print(f"\n{label}")
        print(f"  record        {r['wins']}-{r['losses']}-{r['pushes']} (W-L-P), {r['n_bets']} decided bets")
        print(f"  win rate      {r['win_rate']:.2%}")
        print(f"  breakeven     {r['breakeven']:.2%}  (at {ATS_ODDS_AMERICAN} odds)")
        print(f"  result        {r['edge_vs_breakeven']:+.2%} vs breakeven -- {verdict}")
        print(f"  significance  std error {r['std_error']:.2%}, z {r['z_vs_breakeven']:+.2f}, p {r['p_value']:.3f}")

    print("\n\nTEST: ATS record by how far the model disagrees with the market")
    print("-" * 78)
    print(ats_by_confidence(splits[1][1]).to_string(index=False, float_format=fmt))
    print("\nIf the model held information the market lacked, its LARGEST")
    print("disagreements should be its best bets. Check whether they are, and")
    print("weigh any bucket against its standard error before believing it.")

    print("\n\nATS record by season (all seasons)")
    print("-" * 78)
    print(ats_by_season(preds).to_string(index=False, float_format=fmt))
