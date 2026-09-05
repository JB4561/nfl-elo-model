"""
Data loading for the NFL Elo margin model.

Source: nflverse/nfldata `games.csv`
  https://github.com/nflverse/nfldata/blob/master/data/games.csv

Why this source: it carries game results AND the market spread in a single
table, so there is no join between two datasets. Joining a scores source to a
separate odds source is one of the most common places a sports backtest
silently breaks (mismatched team abbreviations, mismatched dates, duplicated
rows), and skipping that join removes the whole class of bug from v1.
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

# Local cache. We download once and reuse, so that (a) reruns are fast and
# (b) results are reproducible even if the upstream file changes underneath us.
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "games_raw.csv")

# Backtest window. 10 seasons is enough to make the ATS win rate mean something
# without reaching back into an era with materially different scoring/rules.
FIRST_SEASON = 2015
LAST_SEASON = 2024

# The only columns we need for v1. Everything else in the source file
# (weather, QB ids, moneylines, referee, ...) is deliberately dropped: those
# are v2 features and carrying them now would just invite scope creep.
COLUMNS = [
    "game_id",
    "season",
    "game_type",    # REG / WC / DIV / CON / SB -- lets us split regular season vs playoffs
    "week",
    "gameday",
    "gametime",
    "away_team",
    "home_team",
    "away_score",
    "home_score",
    "result",       # home_score - away_score (verified exact on 2015-2024)
    "spread_line",  # market line, POSITIVE = home favored, same sign as `result`
    "location",     # "Home" or "Neutral" -- matters for home-field advantage
]

# Three franchises relocated inside our window and nflverse gives them a new
# code afterwards. We map each old code to its current one so the franchise
# keeps a single, continuous Elo history across the move.
#
# Why: it is the same roster, coaching staff and front office -- a change of
# city is not a reason to erase what we have learned about the team. Leaving
# them split would silently reset those teams to the base rating mid-backtest.
# We already have a season-carryover parameter to express "teams change from
# year to year" in a principled way; making relocation a second, ad-hoc reset
# would double-count that effect for three arbitrary teams.
FRANCHISE_MAP = {
    "STL": "LA",    # Rams,     St. Louis  -> Los Angeles (2016)
    "SD": "LAC",    # Chargers, San Diego  -> Los Angeles (2017)
    "OAK": "LV",    # Raiders,  Oakland    -> Las Vegas   (2020)
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def download_games(force_refresh: bool = False) -> pd.DataFrame:
    """Fetch the raw nflverse games file, using a local cache.

    Set force_refresh=True to re-pull from the network (e.g. to pick up newly
    played games).
    """
    if os.path.exists(CACHE_PATH) and not force_refresh:
        return pd.read_csv(CACHE_PATH)

    df = pd.read_csv(GAMES_URL)
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
    return df


def load_games(
    first_season: int = FIRST_SEASON,
    last_season: int = LAST_SEASON,
    include_playoffs: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return a clean, chronologically sorted game table ready for the backtest.

    The output is the single source of truth for the rest of the project: the
    Elo loop will walk this frame top to bottom, in order, one game at a time.
    """
    df = download_games(force_refresh=force_refresh)

    df = df[COLUMNS].copy()
    df = df[(df["season"] >= first_season) & (df["season"] <= last_season)]

    if not include_playoffs:
        df = df[df["game_type"] == "REG"]

    # Drop games that have not been played yet. The source file includes future
    # scheduled games with null scores; feeding those to the model would be
    # meaningless, and a null score silently propagating into a rating update
    # is exactly the kind of bug that is hard to spot later.
    df = df[df["home_score"].notna() & df["away_score"].notna()]

    # Collapse relocated franchises onto a single code (see FRANCHISE_MAP).
    # Codes not in the map are left untouched.
    df["home_team"] = df["home_team"].replace(FRANCHISE_MAP)
    df["away_team"] = df["away_team"].replace(FRANCHISE_MAP)

    # Parse the date so sorting is a real time comparison, not string ordering.
    df["gameday"] = pd.to_datetime(df["gameday"])

    # ---- Chronological ordering: the backbone of the no-look-ahead guarantee.
    #
    # We sort by actual kickoff (date, then time). Because the Elo loop walks
    # this frame in order and only ever updates ratings AFTER predicting a game,
    # correct ordering here is what makes "we only used the past" true.
    #
    # A note on ties in the sort key: several games kick off at the same time,
    # so their relative order is arbitrary. That is harmless, because a team
    # never plays twice in the same time slot -- so no game in a slot can
    # possibly influence the correct prediction for another game in that slot.
    # Ordering only has to be right ACROSS slots, and it is.
    #
    # gametime can be missing on a few older rows, so it is the secondary key
    # only; the date does the real work.
    df = df.sort_values(["gameday", "gametime", "game_id"], kind="mergesort").reset_index(drop=True)

    return df


if __name__ == "__main__":
    games = load_games()
    print(f"Loaded {len(games)} games, {games['season'].min()}-{games['season'].max()}")
    print(f"Date range: {games['gameday'].min().date()} to {games['gameday'].max().date()}")
    print(f"Teams: {sorted(set(games['home_team']) | set(games['away_team']))}")
    print()
    print(games.head(5).to_string(index=False))
