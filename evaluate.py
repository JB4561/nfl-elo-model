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

import numpy as np
import pandas as pd

from backtest import run_backtest
from params import BURN_IN_SEASONS, TEST_SEASONS, TRAIN_SEASONS


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


if __name__ == "__main__":
    from data import load_games

    games = load_games()
    table = compare_mov_modes(games)

    pd.set_option("display.width", 200)
    print("MOV multiplier comparison")
    print("=" * 78)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print("Model selection is made on TRAIN; TEST confirms generalisation (see module docstring).")
