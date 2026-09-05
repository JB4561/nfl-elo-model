"""
Level 1 unit checks for update_ratings — tests the function in isolation,
no backtest needed. Run: python test_update.py
"""
from elo import update_ratings


def show(label, home_before, away_before, new_home, new_away):
    home_change = new_home - home_before
    away_change = new_away - away_before
    print(f"\n{label}")
    print(f"  home: {home_before:.1f} -> {new_home:.2f}  (change {home_change:+.3f})")
    print(f"  away: {away_before:.1f} -> {new_away:.2f}  (change {away_change:+.3f})")
    print(f"  zero-sum check: home_change + away_change = {home_change + away_change:+.6f}  (should be ~0)")


# --- Case 1: two even (1500) teams, home wins 3-0 ---
# Expect: home gains a little, away loses EXACTLY that amount.
nh, na = update_ratings(1500, 1500, 3, 0)
show("Case 1: 1500 vs 1500, home wins 3-0", 1500, 1500, nh, na)

# --- Case 2: same game at a NEUTRAL site ---
# Expect: home gains slightly MORE than Case 1 (no HFA, so the win is more surprising).
nh_neutral, na_neutral = update_ratings(1500, 1500, 3, 0, neutral_site=True)
show("Case 2: 1500 vs 1500, home wins 3-0, NEUTRAL site", 1500, 1500, nh_neutral, na_neutral)
print(f"  --> compare home gain: neutral {nh_neutral-1500:+.3f} should be > non-neutral {nh-1500:+.3f}")

# --- Case 3: 1700 team beats 1300 team AT HOME (favorite wins) ---
# Expect: SMALL shift (favorite winning is weak evidence; MOV correction damps it).
nh, na = update_ratings(1700, 1300, 24, 10)
show("Case 3: 1700 (home) beats 1300, 24-10 (favorite wins)", 1700, 1300, nh, na)

# --- Case 4: 1300 team beats 1700 team ON THE ROAD (upset) ---
# Here the 1300 team is AWAY. home=1700 loses. Expect: LARGE shift.
# Critical bug check: the away (1300) team's new rating should end NEAR 1300+shift,
# NOT near 1700. If it comes out ~1700, the new_away line is using the wrong base rating.
nh, na = update_ratings(1700, 1300, 10, 24)
show("Case 4: 1300 (away) beats 1700 (home), 24-10 (UPSET)", 1700, 1300, nh, na)
print(f"  --> away team should end near ~1300s (started 1300), NOT near 1700. Got {na:.1f}")

# --- Case 5: explicit magnitude comparison, favorite-win vs upset ---
# Same rating gap, same margin, opposite outcome. Upset should move ratings MUCH more.
fav_h, fav_a = update_ratings(1700, 1300, 24, 10)   # favorite (home) wins big
ups_h, ups_a = update_ratings(1700, 1300, 10, 24)   # underdog (away) wins big
print("\nCase 5: favorite-win vs upset, same 400-Elo gap and same 14-pt margin")
print(f"  favorite wins: home moves {fav_h-1700:+.3f}")
print(f"  upset (away wins): home moves {ups_h-1700:+.3f}")
print(f"  --> |upset move| should be LARGER than |favorite-win move|")