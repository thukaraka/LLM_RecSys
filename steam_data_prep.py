#!/usr/bin/env python3
"""
Steam-200k data prep: games + hours of play indicate user preference (GTI).

- Load steam-200k.csv: user-id, game-title, behavior-name, value
  (behavior: 'purchase' value=1, 'play' value=hours)
- Define "enough interactions": user has more than 50 games (with play) — see MIN_GAMES_WITH_PLAY
- Randomly select 2 users with enough interactions → few-shot example users
- Select NUM_TARGET users (with enough interactions, excluding the 2) for evaluation

Output: user lists, per-user games+hours for prompts, ground truth (GTI) = games + hours = preference.
"""
import hashlib
import random
from pathlib import Path

import pandas as pd


def _deterministic_seed(seed, *args):
    """Reproducible seed from (seed, ...). Use instead of hash() which is not reproducible in Python 3."""
    h = hashlib.md5(str(seed).encode())
    for a in args:
        h.update(str(a).encode())
    return int(h.hexdigest(), 16) % (2**32 - 1)

BASE_DIR = Path(__file__).resolve().parent
STEAM_CSV = BASE_DIR / "Data" / "steam-200k.csv"
OUTPUT_DIR = BASE_DIR / "Data" / "steam_prep"
USER_SELECTION_SEED = 42

# "Enough interactions" = user has more than this many games (with play); optionally MIN_HOURS
MIN_GAMES_WITH_PLAY = 51   # more than 50 games (num_games_played > 50)
MIN_HOURS = 0              # no minimum hours; boundary is games-only
NUM_FEW_SHOT_USERS = 2
NUM_TARGET_USERS = 100
# User history: top N most hours + bottom N least hours = 10 games total for prompt
TOP_K_HOURS = 5
BOTTOM_K_HOURS = 5
NUM_RECOMMENDATIONS_FEW_SHOT = 10   # few-shot example recommendations = top 10 most played
MIN_HOURS_FOR_SAMPLE = 0.5   # only sample games user played at least this many hours


def load_steam(path=None):
    """Load steam-200k.csv; no header. Columns: user_id, game_title, behavior_name, value, (5th)."""
    path = path or STEAM_CSV
    df = pd.read_csv(
        path,
        header=None,
        names=["user_id", "game_title", "behavior_name", "value", "extra"],
        dtype={"user_id": str, "game_title": str, "behavior_name": str, "value": float},
        encoding="utf-8",
        on_bad_lines="warn",
    )
    # Clean: strip quotes from game_title if present
    df["game_title"] = df["game_title"].astype(str).str.strip().str.strip('"')
    return df


def user_interaction_stats(df):
    """Per user: total play hours, number of games with play, number of purchases."""
    play = df[df["behavior_name"] == "play"].copy()
    purchase = df[df["behavior_name"] == "purchase"].copy()
    hours = play.groupby("user_id").agg(
        total_hours=("value", "sum"),
        num_games_played=("game_title", "nunique"),
    ).reset_index()
    purchases = purchase.groupby("user_id").size().reset_index(name="num_purchases")
    stats = hours.merge(purchases, on="user_id", how="left").fillna(0)
    stats["num_purchases"] = stats["num_purchases"].astype(int)
    return stats


def select_users(stats, min_hours=MIN_HOURS, min_games=MIN_GAMES_WITH_PLAY, seed=USER_SELECTION_SEED):
    """Users with enough interactions; then 2 random few-shot, 100 target (excluding few-shot)."""
    enough = stats[
        (stats["total_hours"] >= min_hours) & (stats["num_games_played"] >= min_games)
    ].copy()
    user_list = enough["user_id"].tolist()
    random.seed(seed)
    few_shot = random.sample(user_list, min(NUM_FEW_SHOT_USERS, len(user_list)))
    remaining = [u for u in user_list if u not in few_shot]
    random.seed(seed + 1)
    target = sorted(random.sample(remaining, min(NUM_TARGET_USERS, len(remaining))))
    return few_shot, target, enough


def build_per_user_games(df, user_ids, min_hours_for_sample=MIN_HOURS_FOR_SAMPLE):
    """
    For each user: list of (game_title, hours) they played (value from 'play' rows).
    Used for prompt input (sample) and for GTI / ground truth (preference = games + hours).
    """
    play = df[df["behavior_name"] == "play"].copy()
    play = play[play["user_id"].isin(user_ids)]
    play = play[play["value"] >= min_hours_for_sample]
    # Per user: all (game_title, hours) sorted by hours desc
    per_user = {}
    for uid in user_ids:
        u = play[play["user_id"] == uid][["game_title", "value"]].drop_duplicates()
        u = u.sort_values("value", ascending=False)
        per_user[uid] = list(zip(u["game_title"].tolist(), u["value"].tolist()))
    return per_user


def select_top_and_bottom_games_for_prompt(per_user_games, top_k=TOP_K_HOURS, bottom_k=BOTTOM_K_HOURS):
    """
    For each user: user history = top_k most hours + bottom_k least hours, in alternating order.
    Order in prompt: first top played, then least played, then top, then least, etc.
    Returns list of (game, hours).
    """
    out = {}
    for uid, games_hours in sorted(per_user_games.items()):
        if not games_hours:
            out[uid] = []
            continue
        n = len(games_hours)
        top_part = games_hours[: min(top_k, n)]
        start_bottom = max(top_k, n - bottom_k)
        bottom_part = games_hours[start_bottom:] if start_bottom < n else []
        # Alternating: top1, least1, top2, least2, ...
        combined = []
        for i in range(max(len(top_part), len(bottom_part))):
            if i < len(top_part):
                combined.append(top_part[i])
            if i < len(bottom_part):
                combined.append(bottom_part[i])
        out[uid] = combined
    return out


def get_next_top_n_excluding(per_user_games, user_id, exclude_titles, n=NUM_RECOMMENDATIONS_FEW_SHOT):
    """
    For a user, return the next n game titles by hours played, excluding games in exclude_titles.
    So if history has top 5 + bottom 5, recommendations = next 10 most played (no overlap).
    """
    games_hours = per_user_games.get(user_id, [])
    exclude_set = set(exclude_titles)
    remaining = [(g, h) for g, h in games_hours if g not in exclude_set]
    top_n = remaining[: min(n, len(remaining))]
    return [g for g, h in top_n]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading Steam-200k...")
    df = load_steam()
    print(f"  Rows: {len(df)}, users: {df['user_id'].nunique()}")

    print("Computing per-user interaction stats (play hours, num games)...")
    stats = user_interaction_stats(df)
    print(f"  Users with at least one play: {len(stats)}")

    few_shot_users, target_users, enough_stats = select_users(
        stats, min_hours=MIN_HOURS, min_games=MIN_GAMES_WITH_PLAY
    )
    print(f"  Users with enough interactions (more than 50 games): {len(enough_stats)}")
    print(f"  Few-shot users (2, randomly selected): {few_shot_users}")
    print(f"  Target users (100): {len(target_users)}")

    all_eval_users = few_shot_users + target_users
    per_user_games = build_per_user_games(df, all_eval_users)
    # User history = 5 most hours played + 5 least hours played (per user)
    sample_for_prompt = select_top_and_bottom_games_for_prompt(
        per_user_games, top_k=TOP_K_HOURS, bottom_k=BOTTOM_K_HOURS
    )

    # --- Console: show how user history was selected (alternating: top, least, top, least...) ---
    print("\n" + "=" * 60)
    print("USER HISTORY (alternating: 1st top played, then 1st least, then 2nd top, 2nd least, ...)")
    print("=" * 60)
    for uid in few_shot_users + target_users[:3]:
        hist = sample_for_prompt.get(uid, [])
        if not hist:
            print(f"  User {uid}: no games")
            continue
        print(f"\n  User {uid}: {[(g, h) for g, h in hist]}")
    print("\n  ... (target users 4+ omitted for brevity; same structure)\n")

    # Few-shot examples: same alternating history order; recommendations = next 10 most played (no overlap)
    examples_steam = []
    print("=" * 60)
    print("FEW-SHOT: history (alternating top/least) -> recommendations (next 10 most played, no overlap)")
    print("=" * 60)
    for uid in few_shot_users:
        history_list = sample_for_prompt.get(uid, [])  # already alternating: top1, least1, top2, least2...
        history_titles = [g for g, h in history_list]
        rec_10 = get_next_top_n_excluding(
            per_user_games, uid, exclude_titles=history_titles, n=NUM_RECOMMENDATIONS_FEW_SHOT
        )
        if len(history_list) < 10 or len(rec_10) < 10:
            print(f"  User {uid}: skipped (need 10 for history and 10 for recommendations; got {len(history_list)}, {len(rec_10)})")
            continue
        history_10 = [{"game": g, "hours": h} for g, h in history_list]
        examples_steam.append({
            "user_id": uid,
            "user_history": history_10,
            "recommendation": rec_10,
        })
        print(f"\n  Few-shot user {uid}:")
        print(f"    History (alternating top/least): {[(g, h) for g, h in history_list]}")
        print(f"    Recommendations (next 10 most played, no overlap): {rec_10}")
    examples_data = {"random": examples_steam}

    # Save for notebook: few-shot list, target list, per-user full games (GTI), sample for prompt, examples
    import json
    with open(OUTPUT_DIR / "few_shot_users.json", "w") as f:
        json.dump(few_shot_users, f, indent=2)
    with open(OUTPUT_DIR / "target_users.json", "w") as f:
        json.dump(target_users, f, indent=2)
    # GTI = preference: list of {"game": title, "hours": h} per user
    gti = {uid: [{"game": g, "hours": h} for g, h in lst] for uid, lst in per_user_games.items()}
    with open(OUTPUT_DIR / "gti_preference.json", "w", encoding="utf-8") as f:
        json.dump(gti, f, indent=2, ensure_ascii=False)
    sample_serializable = {uid: [{"game": g, "hours": h} for g, h in lst] for uid, lst in sample_for_prompt.items()}
    with open(OUTPUT_DIR / "sample_for_prompt.json", "w", encoding="utf-8") as f:
        json.dump(sample_serializable, f, indent=2, ensure_ascii=False)
    with open(OUTPUT_DIR / "examples_steam.json", "w", encoding="utf-8") as f:
        json.dump(examples_data, f, indent=2, ensure_ascii=False)
    enough_stats.to_csv(OUTPUT_DIR / "user_stats_enough.csv", index=False)
    print(f"\nSaved to {OUTPUT_DIR}: few_shot_users.json, target_users.json, gti_preference.json, sample_for_prompt.json (user history: alternating top5/least5), examples_steam.json (few-shot: same alternating history, recommendations=next 10 most played), user_stats_enough.csv")
    return 0


if __name__ == "__main__":
    exit(main())
