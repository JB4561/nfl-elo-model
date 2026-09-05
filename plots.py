"""
Plots for the v1 results.

Three figures, each answering one question:

  1. elo_over_time      -- does the rating system track reality?
  2. error_distribution -- how are the model's misses shaped, vs the market's?
  3. ats_by_confidence  -- does disagreeing with the market more pay off?

Every figure is rendered twice, light and dark, so the README can serve the
one matching the reader's theme. The dark variant is a separate set of colour
steps chosen for the dark surface, not an automatic inversion of the light one.

Run: python plots.py   (writes PNGs into plots/)
"""

import math
import os

import matplotlib
matplotlib.use("Agg")  # no display needed; write files directly

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from data import load_games
from backtest import run_backtest
from evaluate import ats_by_confidence, ats_record, margin_metrics, select_seasons
from params import BASE_RATING, BURN_IN_SEASONS, TEST_SEASONS

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
# Categorical slots are taken in fixed order and never cycled. This three-slot
# set was checked with the palette validator and clears every gate on the
# all-pairs list in both modes (worst CVD dE 9.2 light / 9.4 dark; worst
# normal-vision dE 24.0 light / 20.9 dark).
#
# One caveat the validator flags: light-mode aqua sits at 2.74:1 against the
# light surface, below the 3:1 bar. The mitigation is that every line carrying
# it is directly labelled, so identity never rests on colour alone.

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70"],
    },
}

LINE_WIDTH = 2.0  # thin marks; the data should outweigh the chrome


def style_axes(ax, t, ygrid=True):
    """Recessive chrome: hairline horizontal grid, no box, muted ticks."""
    ax.set_facecolor(t["surface"])
    ax.figure.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=t["muted"], labelsize=9, length=0)
    if ygrid:
        ax.grid(axis="y", color=t["grid"], linewidth=1.0, zorder=0)
        ax.set_axisbelow(True)


def titled(ax, t, title, subtitle):
    """Title plus a wrapped subtitle, with enough pad that they never collide.

    Pad is computed from the subtitle's line count rather than guessed once,
    so adding a line to any subtitle cannot silently overlap the title.
    """
    lines = subtitle.count("\n") + 1
    ax.set_title(title, fontsize=13, color=t["ink"], pad=16 + 13 * lines, loc="left")
    ax.annotate(subtitle, xy=(0, 1.012), xycoords="axes fraction", fontsize=9,
                color=t["ink2"], va="bottom", linespacing=1.5)


def rounded_bar(ax, x, height, width, color, radius_frac=0.06, zorder=3):
    """A bar with rounded data-end corners, square where it meets the baseline.

    Rounding the top only keeps the bar visually anchored to zero; rounding all
    four corners would lift it off the baseline and misrepresent the origin.
    """
    r = min(width * radius_frac, abs(height) / 2)
    left, right, top = x - width / 2, x + width / 2, height
    verts = [
        (left, 0), (left, top - r),
        (left, top), (left + r, top),
        (right - r, top),
        (right, top), (right, top - r),
        (right, 0), (left, 0),
    ]
    codes = [
        Path.MOVETO, Path.LINETO,
        Path.CURVE3, Path.CURVE3,
        Path.LINETO,
        Path.CURVE3, Path.CURVE3,
        Path.LINETO, Path.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", zorder=zorder))


def save(fig, name, theme):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}_{theme}.png")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 1. Elo over time
# ---------------------------------------------------------------------------

def team_rating_history(preds: pd.DataFrame, team: str) -> pd.DataFrame:
    """Every rating this team carried INTO a game, in date order.

    Uses the pregame rating recorded by the backtest, so the series shows what
    the model believed before each game -- the same number that produced the
    prediction, not a hindsight-adjusted one.
    """
    home = preds[preds["home_team"] == team][["gameday", "home_rating_pre"]]
    away = preds[preds["away_team"] == team][["gameday", "away_rating_pre"]]
    home = home.rename(columns={"home_rating_pre": "rating"})
    away = away.rename(columns={"away_rating_pre": "rating"})
    return pd.concat([home, away]).sort_values("gameday").reset_index(drop=True)


def plot_elo_over_time(preds, teams, theme="light"):
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(10, 5))
    style_axes(ax, t)

    # The league mean never drifts (the update rule is zero-sum), so 1500 is a
    # fixed, meaningful reference rather than a moving average.
    ax.axhline(BASE_RATING, color=t["axis"], linewidth=1.0, linestyle=(0, (4, 4)), zorder=1)
    # Boxed in the surface colour so a team line crossing 1500 cannot bury it.
    ax.annotate("1500 = league average", xy=(0.004, BASE_RATING), xycoords=("axes fraction", "data"),
                va="center", ha="left", fontsize=8.5, color=t["muted"], zorder=5,
                bbox=dict(facecolor=t["surface"], edgecolor="none", pad=1.5))

    for i, team in enumerate(teams):
        h = team_rating_history(preds, team)
        color = t["series"][i]
        ax.plot(h["gameday"], h["rating"], color=color, linewidth=LINE_WIDTH, zorder=3, label=team)
        # Direct label at the line end. Text stays in ink; the line beside it
        # carries the identity, so colour is never the only channel.
        last = h.iloc[-1]
        ax.annotate(f" {team}", xy=(last["gameday"], last["rating"]), xytext=(6, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=10, fontweight="bold", color=t["ink"], zorder=4)

    titled(ax, t, "Elo rating through time",
           "Rating carried into each game, 2015-2024.\n"
           "Sawtooth pattern is the between-season regression toward 1500.")
    ax.set_ylabel("Elo rating", fontsize=9.5, color=t["ink2"])
    leg = ax.legend(frameon=False, loc="lower left", fontsize=9, ncol=len(teams))
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    ax.margins(x=0.06)
    return save(fig, "elo_over_time", theme)


# ---------------------------------------------------------------------------
# 2. Prediction error distribution
# ---------------------------------------------------------------------------

def plot_error_distribution(preds, theme="light"):
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(10, 5))
    style_axes(ax, t)

    model_err = preds["pred_margin"] - preds["actual_margin"]
    market_err = preds["spread_line"] - preds["actual_margin"]
    bins = np.arange(-45, 46, 3)

    # Step outlines rather than filled bars: two overlapping filled histograms
    # would hide each other, and the comparison IS the point of this figure.
    for err, color, label in ((model_err, t["series"][0], "Model"),
                              (market_err, t["series"][1], "Market")):
        ax.hist(err, bins=bins, histtype="step", linewidth=LINE_WIDTH,
                color=color, label=label, zorder=3)

    ax.axvline(0, color=t["axis"], linewidth=1.0, linestyle=(0, (4, 4)), zorder=1)

    m, k = margin_metrics(preds), margin_metrics(
        preds.rename(columns={"spread_line": "pred_margin", "pred_margin": "_m"}))
    counts, _ = np.histogram(model_err, bins=bins)
    peak = counts.max()
    # Direct labels: a short swatch in the series colour carries identity, the
    # text itself stays in ink.
    for j, (label, metrics, color) in enumerate(
            (("Model", m, t["series"][0]), ("Market", k, t["series"][1]))):
        y = peak * (0.96 - 0.10 * j)
        ax.plot([-43, -38], [y, y], color=color, linewidth=LINE_WIDTH,
                solid_capstyle="round", zorder=4)
        ax.annotate(f"{label}   MAE {metrics['mae']:.2f}", xy=(-36.5, y), fontsize=9.5,
                    color=t["ink"], va="center", ha="left", fontweight="bold", zorder=4)

    titled(ax, t, "Distribution of prediction errors",
           "Predicted minus actual home margin, test seasons 2022-2024.\n"
           "Both centre near zero; the market's distribution is the narrower one.")
    ax.set_xlabel("Prediction error (points)", fontsize=9.5, color=t["ink2"])
    ax.set_ylabel("Games", fontsize=9.5, color=t["ink2"])
    leg = ax.legend(frameon=False, loc="upper right", fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    return save(fig, "error_distribution", theme)


# ---------------------------------------------------------------------------
# 3. ATS by confidence bucket
# ---------------------------------------------------------------------------

def plot_ats_by_confidence(preds, theme="light"):
    t = THEMES[theme]
    table = ats_by_confidence(preds)
    rec = ats_record(preds)
    breakeven = rec["breakeven"]

    fig, ax = plt.subplots(figsize=(10, 5))
    style_axes(ax, t)

    x = np.arange(len(table))
    for xi, row in zip(x, table.itertuples()):
        rounded_bar(ax, xi, row.win_rate * 100, 0.62, t["series"][0])
        # +/-1 standard error. Without this the 6+ bucket reads as a strategy;
        # with it, the bar plainly overlaps breakeven.
        ax.errorbar(xi, row.win_rate * 100, yerr=row.std_error * 100, fmt="none",
                    ecolor=t["ink2"], elinewidth=1.4, capsize=5, capthick=1.4, zorder=4)
        # Surface-coloured box so the breakeven rule cannot strike through a
        # value label on the bars that sit closest to it.
        ax.annotate(f"{row.win_rate*100:.1f}%", xy=(xi, row.win_rate * 100 + row.std_error * 100),
                    xytext=(0, 7), textcoords="offset points", ha="center",
                    fontsize=9.5, fontweight="bold", color=t["ink"], zorder=6,
                    bbox=dict(facecolor=t["surface"], edgecolor="none", pad=1.5))
        ax.annotate(f"n={row.n_bets}", xy=(xi, 1.5), ha="center", fontsize=8.5,
                    color=t["surface"], zorder=5)

    # The rule stops short of the label rather than running to the axes edge,
    # so it cannot strike through its own text.
    ax.set_xlim(-0.62, len(table) - 1 + 1.02)
    ax.plot([-0.62, len(table) - 1 + 0.36], [breakeven * 100] * 2,
            color=t["ink2"], linewidth=1.4, linestyle=(0, (5, 3)), zorder=4)
    ax.annotate(f"{breakeven*100:.2f}%\nbreakeven\nat -110",
                xy=(len(table) - 1 + 0.44, breakeven * 100), ha="left", va="center",
                fontsize=8.5, color=t["ink2"], fontweight="bold", linespacing=1.4)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b} pts" for b in table["edge_bucket"]], color=t["ink2"], fontsize=9.5)
    ax.set_ylim(0, 72)
    titled(ax, t, "ATS win rate by disagreement with the market",
           "Test seasons 2022-2024. Whiskers show +/-1 standard error.\n"
           "If the model knew something the market did not, the right-hand bars would rise.")
    ax.set_xlabel("How far the model's margin departs from the line", fontsize=9.5, color=t["ink2"])
    ax.set_ylabel("ATS win rate (%)", fontsize=9.5, color=t["ink2"])
    return save(fig, "ats_by_confidence", theme)


if __name__ == "__main__":
    plt.rcParams["font.family"] = "sans-serif"

    games = load_games()
    preds, _ = run_backtest(games)
    test = select_seasons(preds, TEST_SEASONS, 0)

    # Three teams with genuinely different arcs over the window: a riser, a
    # faller, and a turnaround. Chosen to show the ratings responding to real
    # events rather than to flatter the model.
    teams = ["KC", "NE", "DET"]

    written = []
    for theme in ("light", "dark"):
        written.append(plot_elo_over_time(preds, teams, theme))
        written.append(plot_error_distribution(test, theme))
        written.append(plot_ats_by_confidence(test, theme))

    for p in written:
        print(f"wrote {os.path.relpath(p)}")
