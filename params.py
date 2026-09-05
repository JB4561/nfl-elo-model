"""
All tunable model parameters, in one place.

Kept in their own module (rather than at the top of elo.py) for one practical
reason: the tuning step later needs to sweep these values programmatically, and
importing a small config module is cleaner than reaching into the model file.

PROVENANCE: every parameter starts at FiveThirtyEight's published NFL Elo
setting, so each has a citable origin rather than being conjured. Values are
then tuned on the TRAINING seasons only, and any parameter that has been
fitted says so in its own comment, alongside the literature value it replaced.

Currently fitted: HOME_FIELD_ADVANTAGE_POINTS.
Still at literature values: everything else.

Reference: FiveThirtyEight, "How Our NFL Predictions Work"
https://fivethirtyeight.com/methodology/how-our-nfl-predictions-work/
"""

# ---------------------------------------------------------------------------
# Rating scale
# ---------------------------------------------------------------------------

# Every team starts here. The absolute number is arbitrary -- Elo only ever
# uses rating DIFFERENCES, so 1500 is convention, not a claim about anything.
BASE_RATING = 1500.0

# The logistic scale. A team 400 Elo above its opponent is expected to win
# 10-to-1. Like BASE_RATING this is a unit convention: it defines what "one
# Elo point" means. What actually matters is its ratio to K and ELO_PER_POINT.
ELO_SCALE = 400.0

# ---------------------------------------------------------------------------
# Update speed
# ---------------------------------------------------------------------------

# K controls how far a single game moves a rating -- the core tradeoff in the
# whole model:
#   K too HIGH -> ratings chase noise; one fluky blowout overwrites a season
#                 of evidence. Responsive but jumpy.
#   K too LOW  -> ratings are stable but stale; a team that genuinely improves
#                 (new QB) takes half a season to be believed.
# The NFL's 17-game season makes this bite harder than in, say, soccer: there
# is very little data per team per year, so we cannot afford to be very slow,
# but each game is also high-variance, so we cannot afford to be very fast.
K_FACTOR = 20.0

# ---------------------------------------------------------------------------
# Rating <-> points conversion
# ---------------------------------------------------------------------------

# How many Elo points equal one point of predicted game margin. This is the
# bridge between the rating system and an actual spread-comparable number:
#   predicted_margin = rating_difference / ELO_PER_POINT
# 25 Elo per point is 538's value. Larger => the same rating gap implies a
# SMALLER predicted margin (more conservative, predictions shrink toward 0).
ELO_PER_POINT = 25.0

# ---------------------------------------------------------------------------
# Home-field advantage
# ---------------------------------------------------------------------------

# Expressed in POINTS (not Elo) because points are the interpretable unit:
# "1.6 points of home advantage" is directly checkable against the market,
# whereas "40 Elo" is not. Converted to Elo internally via ELO_PER_POINT.
#
# FITTED on the training seasons (2016-2021) -- see tune.py. The literature
# value is 2.6 points (538's 65 Elo at 25 Elo/point), which left the model
# over-predicting home teams by +1.06 points per game. Sweeping this parameter
# and scoring on train only:
#
#     hfa=2.6 (literature)  mae 10.3736  bias +1.06
#     hfa=1.6 (chosen)      mae 10.3284  bias +0.02
#     hfa=1.3 (min MAE)     mae 10.3263  bias -0.29
#
# 1.6 was chosen over the MAE-optimal 1.3 because it costs 0.002 MAE and in
# exchange removes the directional lean entirely. Note how flat that curve is:
# HFA was genuinely mis-specified, but correcting it buys calibration, not
# meaningful accuracy.
#
# Why so far below the literature value: leaguewide home advantage collapsed
# mid-window (near zero in the empty-stadium 2020 season), and mean actual home
# margin across the training seasons is +1.68 points. The fit recovers that
# empirical quantity rather than inventing one.
#
# HONEST CAVEAT -- HOME ADVANTAGE IS NON-STATIONARY. Mean home margin is +1.68
# in the training seasons but +2.41 in the test seasons, so a value fitted on
# train does not transfer cleanly: train bias is +0.02, test bias is -0.73.
# The calibration this parameter was tuned to fix did not stay fixed.
#
# It is kept at 1.6 anyway, for two reasons. First, re-fitting it to the test
# seasons would destroy the holdout and make every subsequent number
# meaningless. Second, the market's own bias over the same test seasons is
# -0.70, nearly identical: the betting market under-predicted home teams in
# 2022-2024 by the same amount the model did. That residual reflects a genuine
# regime shift in the sport, not a defect in this parameter.
#
# A time-varying or rolling HFA would address this properly. That is a v2
# concern; a single fixed constant is the v1 design.
HOME_FIELD_ADVANTAGE_POINTS = 1.6

# ---------------------------------------------------------------------------
# Season carryover
# ---------------------------------------------------------------------------

# Fraction of a team's deviation from the mean that it KEEPS into the next
# season. 0.0 = full reset (everyone starts equal every year, throwing away
# real information); 1.0 = no regression (assumes a great 2019 team is still
# great in 2020, ignoring the draft, free agency and retirements).
# 2/3 retained (== 538's "reset one third of the way to the mean") is the
# baseline. Named for what it KEEPS, so a bigger number means more memory.
SEASON_CARRYOVER = 2.0 / 3.0

# ---------------------------------------------------------------------------
# Margin-of-victory multiplier
# ---------------------------------------------------------------------------

# Which MOV weighting to use. This is a genuine modelling choice, so it is a
# switch rather than a hardcoded formula -- we can run the backtest both ways
# and MEASURE whether the refinement earns its place.
#
#   "none" -- every win moves ratings the same amount regardless of margin.
#             Throws away real information (a 40-point win says more than a
#             1-point win), but it is the honest simplest baseline and makes
#             "does MOV weighting actually help?" a testable question.
#
#   "538"  -- log of the margin, damped by the favorite's rating edge:
#
#               log(|margin| + 1) * (B / (SLOPE * winner_elo_diff + B))
#
#             The log means the 7th point of margin matters less than the 1st
#             (diminishing returns -- a 35-point win is not 5x as informative
#             as a 7-point win). The second term is an AUTOCORRELATION
#             CORRECTION: it shrinks the update when a heavy favorite wins big
#             (we already knew they were good, so it is weak evidence) and
#             inflates it when an underdog wins big (genuinely surprising).
#             Without that correction, strong teams beating weak teams badly
#             would compound their own ratings upward in a feedback loop.
MOV_MULTIPLIER_MODE = "538"          # "none" | "538"

# Constants inside the 538 correction term. B is the value the term collapses
# to when the two teams are evenly rated (making the multiplier a plain log);
# SLOPE sets how hard the damping bites as the rating gap widens.
MOV_CORRECTION_B = 2.2
MOV_CORRECTION_SLOPE = 0.001

# ---------------------------------------------------------------------------
# Backtest split
# ---------------------------------------------------------------------------

# Parameters get tuned on TRAIN seasons only; the headline result is reported
# on TEST seasons, which the tuning never sees.
#
# IMPORTANT: this split governs which games count toward METRICS, not which
# games the model learns from. The Elo loop always runs continuously across
# all seasons -- a team's rating entering 2022 must reflect its 2021 games,
# or we would be handicapping the test period rather than testing it.
TRAIN_SEASONS = (2015, 2021)
TEST_SEASONS = (2022, 2024)

# Seasons to exclude from METRICS at the very start of the backtest. Every
# team begins at BASE_RATING with the model knowing nothing, so early games
# are scored against ratings that are pure prior. Excluding the first season
# from the reported numbers avoids penalising the model for a cold start.
# (Those games still RUN -- they are how the ratings warm up.)
BURN_IN_SEASONS = 1

# ---------------------------------------------------------------------------
# Against-the-spread evaluation
# ---------------------------------------------------------------------------

# Standard American odds on a point-spread bet. At -110 a bettor risks 110 to
# win 100, so wins must cover losses plus the vig:
#
#     breakeven win rate = 110 / (110 + 100) = 0.5238
#
# This is the number the ATS result must clear to represent a real edge.
# Beating 50% is not enough and is not evidence of anything.
ATS_ODDS_AMERICAN = -110
