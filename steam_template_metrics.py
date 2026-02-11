#!/usr/bin/env python3
"""
Calculate evaluation metrics for Steam game recommendations (similar to greedy_template_metrics.py for LastFM).

Flow:
- Target users: from Data/steam_prep/target_users.json (100 users).
- For each user, ground truth = all games they played in steam-200k MINUS the 10 games in
  sample_for_prompt.json (the "played games" shown in the prompt). Any gt game whose main title
  matches a sample game >= 80% is also excluded.
- Recommendations: game_1..game_10 from greedy_steam_results_extracted.csv. Each is matched to the
  steam-200k catalog (RapidFuzz >= 80%). Relevance: matched catalog game is in user's ground truth
  (main-title similarity >= 80%). Exposure grouping also uses >= 80%.
- Metrics per (model_name, prompt_type, template): HR@10, Precision@10, NDCG@10 (averaged over users),
  Gini and Entropy from aggregate exposure (main title only).
- Exposure counts: every slot contributes. Real games (catalog-matched or not) use normalized game
  title. Empty/hallucinated slots count as "" (included in Gini/Entropy). empty_slot_count and empty_slot_ratio reported.
- Hallucinated: games from Data/temperature/hallucinated.xlsx (column "title") are replaced with "" before metrics.
- All matching: catalogue and recommended game are converted to lowercase and special characters
  are removed (normalized form) before comparison; >= 80% for catalog match, ground-truth
  relevance, and exposure grouping.
"""

import json
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from scipy.stats import entropy
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
STEAM_200K = BASE_DIR / "Data" / "steam-200k.csv"
STEAM_PREP = BASE_DIR / "Data" / "steam_prep"
TEMPERATURE_DIR = BASE_DIR / "Data" / "temperature"
TARGET_USERS_FILE = STEAM_PREP / "target_users.json"
SAMPLE_FOR_PROMPT_FILE = STEAM_PREP / "sample_for_prompt.json"
EXTRACTED_CSV = BASE_DIR / "greedy_steam_results_extracted.csv"
OUTPUT_METRICS_CSV = BASE_DIR / "steam_template_metrics_results.csv"
# Hallucinated games: from Data/temperature/hallucinated.xlsx (column "title"); matching recs replaced with "" before metrics
HALLUCINATED_XLSX = TEMPERATURE_DIR / "hallucinated.xlsx"
HALLUCINATED_CUTOFF = 80

K = 10
CATALOG_CUTOFF = 80   # match recommended game to catalog: >= 80%
GT_CUTOFF = 80        # relevance and exposure grouping: >= 80% (accuracy and exposure)


def normalize_game_main_title(name: str) -> str:
    """
    Normalize for matching and exposure: lowercase and remove special characters.
    Both catalogue and recommended game are normalized before comparison.
    Order: strip quotes -> remove parenthetical/bracket content -> remove/replace
    special characters with space -> collapse spaces -> lowercase.
    """
    if not name or not isinstance(name, str):
        return ""
    name = str(name).replace('"', "").replace("'", "").strip()
    # Remove content in parentheses and brackets
    name = re.sub(r"\s*\([^)]*\)\s*", " ", name)
    name = re.sub(r"\s*\[[^\]]*\]\s*", " ", name)
    # Remove special characters: keep only letters, digits, spaces (replace rest with space)
    name = re.sub(r"[^\w\s]", " ", name, flags=re.UNICODE)
    # Collapse spaces and convert to lowercase
    name = re.sub(r"\s+", " ", name).strip().lower()
    return name


def is_empty_slot(rec) -> bool:
    """True if recommendation slot is empty (None, NaN, blank, "nan"/"none"). Recorded as "" in exposure."""
    if rec is None:
        return True
    if isinstance(rec, float) and rec != rec:
        return True
    try:
        if pd.isna(rec):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(rec, str):
        s = rec.strip()
        if not s or s.lower() in ("nan", "none", "n/a", "null"):
            return True
    return False


def load_hallucinated_games():
    """Load hallucinated game names from hallucinated.xlsx (column 'title'). Returns (original, normalized) or ([], [])."""
    if not HALLUCINATED_XLSX.exists():
        return [], []
    try:
        df = pd.read_excel(HALLUCINATED_XLSX, engine="openpyxl")
    except Exception as e:
        print(f"  Warning: could not load hallucinated games from {HALLUCINATED_XLSX.name}: {e}")
        return [], []
    if df.empty:
        return [], []
    for c in ["title", "Title", "game", "name", "Game", "Name"]:
        if c in df.columns:
            names = df[c].dropna().astype(str).str.strip()
            names = names[names != ""].unique().tolist()
            names = [g for g in names if str(g).lower() != "nan"]
            if names:
                return names, [normalize_game_main_title(g) for g in names]
    col = df.columns[0]
    names = df[col].dropna().astype(str).str.strip()
    names = names[names != ""].unique().tolist()
    names = [g for g in names if str(g).lower() != "nan"]
    if names:
        return names, [normalize_game_main_title(g) for g in names]
    return [], []


def is_hallucinated_match(rec, hall_original: list, hall_normalized: list, cutoff: int = HALLUCINATED_CUTOFF) -> bool:
    """True if rec matches any hallucinated game at >= cutoff (RapidFuzz on normalized)."""
    if not rec or not str(rec).strip() or not hall_normalized:
        return False
    norm_rec = normalize_game_main_title(str(rec).strip())
    if not norm_rec:
        return False
    best = process.extractOne(norm_rec, hall_normalized, scorer=fuzz.token_sort_ratio)
    if not best:
        return False
    _, score, _ = best
    return score >= cutoff


def replace_hallucinated_with_empty(rec_games: list, hall_original: list, hall_normalized: list) -> list:
    """Replace any recommendation matching hallucinated list at >= 80% with "". Call before metrics."""
    if not hall_normalized:
        return list(rec_games)
    out = []
    for rec in rec_games:
        if is_empty_slot(rec):
            out.append(rec)
            continue
        if is_hallucinated_match(rec, hall_original, hall_normalized):
            out.append("")
        else:
            out.append(rec)
    return out


def load_steam_catalog() -> list[str]:
    """Unique game names from steam-200k (catalog for matching)."""
    df = pd.read_csv(
        STEAM_200K,
        header=None,
        names=["user_id", "game_title", "behavior_name", "value", "extra"],
        encoding="utf-8",
        encoding_errors="replace",
    )
    catalog = df["game_title"].dropna().astype(str).str.strip().str.strip('"').unique().tolist()
    return [g for g in catalog if g and g.lower() != "nan"]


def load_target_users() -> list[str]:
    """Load target user IDs (100 users used in the greedy run)."""
    with open(TARGET_USERS_FILE, encoding="utf-8") as f:
        users = json.load(f)
    return [str(u) for u in users]


def load_sample_for_prompt() -> dict:
    """Load {user_id: [{"game": title, "hours": h}, ...]} - the 10 games shown in the prompt."""
    with open(SAMPLE_FOR_PROMPT_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return {str(uid): lst for uid, lst in data.items()}


def load_user_play_data(target_users: list[str]) -> dict:
    """For each user, set of game titles they played (from steam-200k play rows)."""
    df = pd.read_csv(
        STEAM_200K,
        header=None,
        names=["user_id", "game_title", "behavior_name", "value", "extra"],
        encoding="utf-8",
        encoding_errors="replace",
    )
    df["game_title"] = df["game_title"].astype(str).str.strip().str.strip('"')
    play = df[df["behavior_name"] == "play"].copy()
    play = play[play["user_id"].astype(str).isin(target_users)]
    gt_raw = {}
    for uid in target_users:
        games = play[play["user_id"].astype(str) == uid]["game_title"].dropna().unique()
        gt_raw[str(uid)] = set(g for g in games if g and str(g).lower() != "nan")
    return gt_raw


def remove_sampled_from_ground_truth(gt_all: set, sample_game_titles: list, cutoff: int = GT_CUTOFF) -> set:
    """
    Ground truth for eval = gt_all minus (1) exact sample titles, (2) any gt game whose
    main title matches any sample main title >= cutoff.
    """
    if not sample_game_titles:
        return gt_all
    sample_main = [normalize_game_main_title(g) for g in sample_game_titles if g]
    sample_main = [m for m in sample_main if m]
    if not sample_main:
        return gt_all
    kept = set()
    for g in gt_all:
        if not g:
            continue
        g_main = normalize_game_main_title(g)
        if not g_main:
            kept.add(g)
            continue
        if any(fuzz.token_sort_ratio(g_main, s_main) >= cutoff for s_main in sample_main):
            continue
        kept.add(g)
    return kept


def match_game_to_catalog(
    recommended: str,
    catalog_original: list[str],
    catalog_normalized: list[str],
    cutoff: int = CATALOG_CUTOFF,
):
    """
    Returns (matched_catalog_game_or_empty, best_score).
    Both recommended and catalogue are normalized (lowercase, special characters removed) before matching.
    """
    if not recommended or not str(recommended).strip():
        return "", 0
    norm_rec = normalize_game_main_title(str(recommended).strip())
    if not norm_rec:
        return "", 0
    if not catalog_normalized:
        return "", 0
    best = process.extractOne(norm_rec, catalog_normalized, scorer=fuzz.token_sort_ratio)
    if not best:
        return "", 0
    best_norm, best_score, best_idx = best
    if best_score < cutoff:
        return "", best_score
    return catalog_original[best_idx], best_score


def is_in_ground_truth(matched_game: str, gt_titles: set, cutoff: int = GT_CUTOFF) -> bool:
    """True if matched_game (catalog name) is relevant: main title matches any gt title >= cutoff (80%)."""
    if not matched_game or not gt_titles:
        return False
    rec_main = normalize_game_main_title(matched_game)
    if not rec_main:
        return False
    for gt in gt_titles:
        if not gt:
            continue
        gt_main = normalize_game_main_title(gt)
        if not gt_main:
            continue
        if fuzz.token_sort_ratio(rec_main, gt_main) >= cutoff:
            return True
    return False


# === Accuracy metrics (same equations as standard HR@K, Precision@K, NDCG@K) ===
# Relevance here: first-occurrence hit positions (matched game in ground truth, main-title >= 80%).

def hit_ratio_at_k(hit_positions: list, k: int = 10) -> int:
    """Hit Ratio@K = 1 if at least one relevant item in top-K, else 0."""
    return int(len(hit_positions) > 0)


def precision_at_k(hit_positions: list, k: int = 10) -> float:
    """Precision@K = (number of relevant items in top-K) / K; denominator = 10 per user."""
    return len(hit_positions) / k


def ndcg_at_k(hit_positions: list, num_gt: int, k: int = 10) -> float:
    """NDCG@K = DCG / IDCG. DCG uses actual positions of hits: sum(1/log2(i+2)) for position i (0-based)."""
    dcg = sum(1.0 / np.log2(i + 2) for i in hit_positions)
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(num_gt, k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# === Fairness/Diversity metrics (exposure distribution) ===
def gini_index(counts):
    """Gini coefficient of exposure inequality."""
    sorted_vals = np.sort(np.array(counts))
    n = len(sorted_vals)
    if n == 0 or sorted_vals.sum() == 0:
        return 0.0
    index = np.sum((2 * np.arange(1, n + 1) - n - 1) * sorted_vals)
    return round(index / (n * sorted_vals.sum()), 4)


def natural_entropy(counts):
    """Entropy of exposure diversity."""
    total = sum(counts)
    if total == 0:
        return 0.0
    p = np.array(counts) / total
    return round(entropy(p, base=np.e), 4)


def main():
    print("=" * 80)
    print("Steam Recommendation Evaluation Metrics")
    print("=" * 80)

    # 1. Load target users and sample-for-prompt
    target_users = load_target_users()
    print(f"Target users: {len(target_users)}")
    sample_for_prompt = load_sample_for_prompt()

    # 2. Build ground truth per user (played games minus sample)
    print("Loading user play data from steam-200k...")
    gt_raw = load_user_play_data(target_users)
    gt_by_user = {}
    for uid in target_users:
        gt_all = gt_raw.get(uid, set())
        sample_list = sample_for_prompt.get(uid, [])
        sample_titles = [x.get("game", "") for x in sample_list if isinstance(x, dict) and x.get("game")]
        gt_by_user[uid] = remove_sampled_from_ground_truth(gt_all, sample_titles, cutoff=GT_CUTOFF)
    print(f"  Ground truth built for {len(gt_by_user)} users")

    # 3. Load catalog (original names) and normalized list for matching
    print("Loading Steam catalog...")
    catalog = load_steam_catalog()
    catalog_normalized = [normalize_game_main_title(g) for g in catalog]
    print(f"  Catalog size: {len(catalog)} (matching uses normalized: lowercase, special chars removed; cutoff >= {CATALOG_CUTOFF}%)")

    # 4. Load hallucinated games (column "title" in Data/temperature/hallucinated.xlsx); matching recs replaced with "" before metrics
    print("Loading hallucinated games from hallucinated.xlsx...")
    hall_original, hall_normalized = load_hallucinated_games()
    if hall_original:
        print(f"  Loaded {len(hall_original)} hallucinated games; matching recs replaced with \"\" before metrics.")
    else:
        print("  No hallucinated.xlsx found or empty; no replacements.")

    # 5. Load recommendations
    print("Loading recommendations...")
    df = pd.read_csv(EXTRACTED_CSV, encoding="utf-8", encoding_errors="replace")
    df["user_id"] = df["user_id"].astype(str)
    df = df[df["user_id"].isin(target_users)]
    print(f"  Rows for target users: {len(df)}")

    # 6. Evaluate per (model_name, prompt_type, template)
    # DUPLICATE HANDLING:
    #   - Accuracy (HR@10, Precision@10, NDCG@10): If a game appears multiple times in the 10 recommendations,
    #     only the FIRST occurrence counts. Example: if "Game A" is at positions 1 and 5, only position 1 counts.
    #   - Exposure (Gini, Entropy): ALL occurrences count. Each slot contributes independently.
    #     Example: if "Game A" appears in 3 slots, it contributes 3 to the exposure distribution.
    # Gini/Entropy: over all recommended slots in config = (users with gt in config) × 10 (e.g. 100 users → 1000).
    # Accuracy: per user denominator = 10 (Precision@10); NDCG uses actual positions of hits; repeated items count only at first occurrence.
    metrics_rows = []
    for (model_name, prompt_type, template), group in tqdm(
        df.groupby(["model_name", "prompt_type", "template"]),
        desc="Configs",
    ):
        hr_list, prec_list, ndcg_list = [], [], []
        all_exposure = []  # one entry per slot (user × 10) → up to 1000 items for Gini/Entropy
        empty_slot_count = 0  # slots with no recommendation (punishment)

        for uid, recs in group.groupby("user_id"):
            # One row per user in this config
            row = recs.iloc[0]
            gt_titles = gt_by_user.get(uid, set())
            if not gt_titles:
                continue

            # Get top-10 recommended games (raw)
            rec_games = []
            for i in range(1, K + 1):
                g = row.get(f"game_{i}")
                if pd.isna(g) or not str(g).strip():
                    rec_games.append(None)
                else:
                    rec_games.append(str(g).strip())

            # Replace hallucinated games with "" before metrics (same as varying_temp_metrics)
            rec_games = replace_hallucinated_with_empty(rec_games, hall_original, hall_normalized)

            # Match each to catalog; build exposure. EMPTY / None / blank / "nan" / hallucinated = same: all recorded as ""
            # so Gini and Entropy count them as one category. ALL slots contribute (duplicates counted).
            matched_games = []
            for rec in rec_games:
                if is_empty_slot(rec):  # None, NaN, blank, "nan"/"none" → assign "" in exposure
                    matched_games.append(None)
                    empty_slot_count += 1
                    all_exposure.append("")
                    continue
                cat_game, _ = match_game_to_catalog(
                    rec, catalog, catalog_normalized, cutoff=CATALOG_CUTOFF
                )
                if cat_game:
                    matched_games.append(cat_game)
                    main_title = normalize_game_main_title(cat_game)
                    all_exposure.append(main_title if main_title else "")
                else:
                    matched_games.append(None)
                    rec_main = normalize_game_main_title(rec)
                    all_exposure.append(rec_main if rec_main else "")

            # ACCURACY METRICS: Repeated items in top-10 - keep only the FIRST occurrence for accuracy.
            # If a game appears at positions 1 and 5, only position 1 counts for HR/Precision/NDCG.
            # first_hit_positions = 0-based positions where a (first-occurrence) hit occurs; used for DCG.
            seen_main = set()
            first_hit_positions = []
            for i, mg in enumerate(matched_games):
                if mg is None:
                    continue
                main_title = normalize_game_main_title(mg)
                if not main_title or main_title in seen_main:  # skip duplicates: only first occurrence
                    continue
                seen_main.add(main_title)
                if is_in_ground_truth(mg, gt_titles, cutoff=GT_CUTOFF):
                    first_hit_positions.append(i)  # position i (0-based) for NDCG

            # Per user: denominator = 10 for Precision; NDCG uses positions in first_hit_positions.
            hr = hit_ratio_at_k(first_hit_positions, k=K)
            prec = precision_at_k(first_hit_positions, k=K)  # hits / 10
            ndcg = ndcg_at_k(first_hit_positions, num_gt=len(gt_titles), k=K)

            hr_list.append(hr)
            prec_list.append(prec)
            ndcg_list.append(ndcg)

        # EXPOSURE METRICS: Gini and Entropy on ALL exposure items. Empty/None/blank/hallucinated all recorded as ""
        # (one category). exposure_counts includes the count of "" so they are in Gini/Entropy.
        exposure_counts = list(Counter(all_exposure).values())
        gini = gini_index(exposure_counts) if exposure_counts else 0.0
        ent = natural_entropy(exposure_counts) if exposure_counts else 0.0
        total_slots = len(all_exposure)
        empty_slot_ratio = empty_slot_count / total_slots if total_slots else 0.0
        empty_string_count = all_exposure.count("")  # how many slots are "" (included in Gini/Entropy)

        metrics_rows.append({
            "model_name": model_name,
            "prompt_type": prompt_type,
            "template": template,
            "HR@10": round(np.mean(hr_list), 4) if hr_list else 0.0,
            "Precision@10": round(np.mean(prec_list), 4) if prec_list else 0.0,
            "NDCG@10": round(np.mean(ndcg_list), 4) if ndcg_list else 0.0,
            "Gini": gini,
            "Entropy": ent,
            "num_users": len(hr_list),
            "num_exposure_events": total_slots,
            "num_unique_exposed": len(Counter(all_exposure)),
            "empty_slot_count": empty_slot_count,
            "empty_string_count": empty_string_count,
            "empty_slot_ratio": round(empty_slot_ratio, 4),
        })

    # 7. Save
    out_df = pd.DataFrame(metrics_rows)
    out_df.to_csv(OUTPUT_METRICS_CSV, index=False)
    print(f"\nMetrics saved to: {OUTPUT_METRICS_CSV}")
    print(out_df.to_string(index=False))

    if out_df.shape[0] > 0:
        print("\nSummary:")
        for col in ["HR@10", "Precision@10", "NDCG@10", "Gini", "Entropy"]:
            if col in out_df.columns:
                print(f"  {col}: mean={out_df[col].mean():.4f}, min={out_df[col].min():.4f}, max={out_df[col].max():.4f}")


if __name__ == "__main__":
    main()
