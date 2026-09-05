"""
Parameter tuning, on the training seasons only.

Tuning touches TRAIN exclusively. A value chosen because it performs well on
TEST would make TEST a tuning input rather than a holdout, and every headline
number reported afterwards would be optimistic by an unknown amount.

TWO OBJECTIVES, WHICH ARE NOT THE SAME THING
--------------------------------------------
Home-field advantage can be chosen to minimise either:

  * MAE  -- average size of the prediction error. This is accuracy: how close
            the predictions land, game by game.
  * |bias| -- average SIGNED error. This is calibration: whether the model
            leans systematically toward home or away teams.

They need not agree, and the distinction matters. A model can be perfectly
calibrated on average (bias = 0) while being wildly wrong on individual games,
and a small systematic lean can occasionally buy better average accuracy. MAE
is the primary objective here because game-level accuracy is what the model is
for; bias is reported alongside as a diagnostic, since a large residual bias
points at a mis-specified parameter rather than at irreducible noise.

WHY THE SWEEP IS NOT SIMPLY "SUBTRACT THE BIAS"
-----------------------------------------------
Lowering HFA does not just shift predictions down by a fixed amount. It also
changes every expected outcome inside the rating update, which changes the
ratings themselves, which changes future predictions. The effect is therefore
close to linear but not exactly, and neutral-site games receive no HFA at all.
Running the full backtest per candidate value captures that feedback; algebra
on the bias alone would not.
"""

import numpy as np
import pandas as pd

from backtest import run_backtest
from evaluate import margin_metrics, select_seasons
from params import BURN_IN_SEASONS, MOV_MULTIPLIER_MODE, TEST_SEASONS, TRAIN_SEASONS


def sweep_hfa(
    games: pd.DataFrame,
    values,
    mov_mode: str = MOV_MULTIPLIER_MODE,
) -> pd.DataFrame:
    """Run a full backtest per candidate HFA and score it on TRAIN.

    Each candidate gets an independent run over all seasons: HFA affects the
    ratings from the first game onward, so runs cannot share state. Only the
    TRAIN rows are scored.
    """
    rows = []
    for hfa in values:
        preds, _ = run_backtest(games, mov_mode=mov_mode, hfa_points=hfa)
        train = select_seasons(preds, TRAIN_SEASONS, BURN_IN_SEASONS)
        rows.append({"hfa_points": hfa, **margin_metrics(train)})
    return pd.DataFrame(rows)


def confirm_on_test(
    games: pd.DataFrame,
    hfa_points: float,
    mov_mode: str = MOV_MULTIPLIER_MODE,
) -> dict:
    """Score a single already-chosen HFA on the held-out seasons.

    Called once, after the choice is made on train. This confirms that the
    choice generalises; it is not part of choosing.
    """
    preds, _ = run_backtest(games, mov_mode=mov_mode, hfa_points=hfa_points)
    return margin_metrics(select_seasons(preds, TEST_SEASONS, 0))


if __name__ == "__main__":
    from data import load_games
    from params import HOME_FIELD_ADVANTAGE_POINTS

    games = load_games()
    grid = np.round(np.arange(0.0, 4.01, 0.1), 2)
    table = sweep_hfa(games, grid)

    best_mae = table.loc[table["mae"].idxmin()]
    best_bias = table.loc[table["bias"].abs().idxmin()]

    pd.set_option("display.width", 200)
    print("HFA sweep, scored on TRAIN seasons only")
    print("=" * 72)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print(f"baseline (538's value):  hfa={HOME_FIELD_ADVANTAGE_POINTS}")
    print(f"minimises MAE:           hfa={best_mae.hfa_points}  "
          f"mae={best_mae.mae:.4f}  bias={best_mae.bias:+.4f}  su={best_mae.su_accuracy:.4f}")
    print(f"minimises |bias|:        hfa={best_bias.hfa_points}  "
          f"mae={best_bias.mae:.4f}  bias={best_bias.bias:+.4f}  su={best_bias.su_accuracy:.4f}")
