"""
Extract game recommendations from results_greedy JSONL files.
Handles templates A, B, and C (including C with narrative/markdown).
Outputs CSV: model_name, prompt_type, user_id, template, response, game_1 ... game_10.
"""
import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd

# Model name -> folder under results_greedy (greedy run results)
MODEL_FOLDERS = {
    "gpt_4o": "steam_gpt_4o_results_revised",
    "gpt_4o_mini": "steam_gpt_4o_mini_results_revised",
    "mistral_7b": "steam_mistral_7b_results_revised",
    "mistral_large_2": "steam_mistral_large_2407_results_revised",
}

RESULTS_BASE = Path(__file__).resolve().parent / "Data"/"greedy_template"
OUTPUT_FILE = Path(__file__).resolve().parent / "greedy_steam_results_extracted.csv"


def clean_game_title(text: str) -> str:
    """Remove markdown, extra quotes, and trim."""
    if not text or not isinstance(text, str):
        return ""
    # Remove **bold** but keep the inner text
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = text.replace("**", "").replace("*", "")
    # Remove parenthetical suffix like *(Roguelike, ...)* or (description)
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text)
    text = re.sub(r"\s*\*\([^)]*\)\s*$", "", text)
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    return text.strip()


def _find_recommendations_json(text: str) -> Optional[str]:
    """Find a JSON object in text that contains 'recommendations' key. Handles ```json ... ``` and inline."""
    # Prefer block inside ```json ... ``` (use brace matching for correct extent)
    code_m = re.search(r"```(?:json)?\s*\{", text)
    if code_m:
        start = code_m.end() - 1  # position of '{'
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    # Find any {...} that contains "recommendations"
    idx = text.find('"recommendations"')
    if idx == -1:
        idx = text.find("'recommendations'")
    if idx == -1:
        idx = text.find("{")
    if idx == -1:
        return None
    # Start from the opening brace of this object (walk backward to find {)
    start = text.rfind("{", 0, idx + 1)
    if start == -1:
        start = idx
        while start > 0 and text[start] != "{":
            start -= 1
    if start == -1:
        return None
    depth = 0
    end = start
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if depth != 0:
        return None
    return text[start:end]


def extract_template_a(response: str) -> list[str]:
    """Template A: JSON with recommendations as list of {rank, title}."""
    games = []
    text = _find_recommendations_json(response)
    if not text:
        return games
    try:
        data = json.loads(text)
        recs = data.get("recommendations", [])
        if isinstance(recs, list):
            for r in recs:
                if isinstance(r, dict) and "title" in r:
                    games.append(clean_game_title(str(r["title"])))
                elif isinstance(r, str):
                    games.append(clean_game_title(r))
    except json.JSONDecodeError:
        pass
    return games[:10]


def extract_template_b(response: str) -> list[str]:
    """Template B: JSON with recommendations as list of strings."""
    games = []
    text = _find_recommendations_json(response)
    if not text:
        return games
    try:
        data = json.loads(text)
        recs = data.get("recommendations", [])
        if isinstance(recs, list):
            for r in recs:
                if isinstance(r, str):
                    games.append(clean_game_title(r))
                elif isinstance(r, dict) and "title" in r:
                    games.append(clean_game_title(str(r["title"])))
    except json.JSONDecodeError:
        pass
    return games[:10]


def extract_template_c(response: str) -> list[str]:
    """
    Template C: RECOMMENDATIONS:\n1) <Game 1>\n2) <Game 2>\n...\n10) <Game 10>
    Also handles narrative format: 1) **Hades** *(Roguelike, ...)* or 1) Hades - description.
    """
    by_rank = {}
    lines = response.split("\n")
    for line in lines:
        match = re.match(r"^\s*(\d+)\)\s*(.+)$", line)
        if match:
            num = int(match.group(1))
            if 1 <= num <= 10 and num not in by_rank:
                raw = match.group(2).strip()
                # Extract game title: **Game Name** *(desc)* -> Game Name; or plain "Game Name"
                bold = re.match(r"^\*\*(.+?)\*\*", raw)
                title = bold.group(1).strip() if bold else raw
                title = re.sub(r"\s*\*\([^)]*\)\s*$", "", title)
                title = re.sub(r"\s*\([^)]*\)\s*$", "", title)
                title = re.sub(r"\s*[-–—]\s*.*$", "", title)
                title = clean_game_title(title)
                if title:
                    by_rank[num] = title
    games = [by_rank[i] for i in range(1, 11) if i in by_rank]
    return games[:10]


def extract_games_from_response(response: str, template_set: str) -> list[str]:
    """Dispatch by template set. Returns up to 10 game titles."""
    if not response or (isinstance(response, str) and response.strip() in ("", "ERROR")):
        return [""] * 10
    template_set = (template_set or "").upper().strip()
    if template_set == "A":
        games = extract_template_a(response)
    elif template_set == "B":
        games = extract_template_b(response)
    else:
        # C or fallback
        games = extract_template_c(response)
    # Pad or trim to 10
    while len(games) < 10:
        games.append("")
    return games[:10]


def load_model_jsonl(folder_name: str) -> list[dict]:
    path = RESULTS_BASE / folder_name / "normal_t0.jsonl"
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main():
    all_rows = []
    for model_name, folder in MODEL_FOLDERS.items():
        data = load_model_jsonl(folder)
        print(f"{model_name} ({folder}): {len(data)} records")
        for rec in data:
            user_id = rec.get("user_id", "")
            template_set = rec.get("template_set", "")
            style = rec.get("style", "")
            response = rec.get("response", "")
            games = extract_games_from_response(response, template_set)
            row = {
                "model_name": model_name,
                "prompt_type": style,
                "user_id": user_id,
                "template": template_set,
                "response": response,
            }
            for i, g in enumerate(games, 1):
                row[f"game_{i}"] = g
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    # Column order: model_name, prompt_type, user_id, template, response, game_1 ... game_10
    col_order = ["model_name", "prompt_type", "user_id", "template", "response"] + [
        f"game_{i}" for i in range(1, 11)
    ]
    df = df[col_order]
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_FILE}")

    # Optional: also write a long-format JSONL with recommendations as a list (for downstream)
    out_jsonl = OUTPUT_FILE.with_suffix(".jsonl")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            games_list = [r[f"game_{i}"] for i in range(1, 11)]
            obj = {
                "model_name": r["model_name"],
                "prompt_type": r["prompt_type"],
                "user_id": r["user_id"],
                "template": r["template"],
                "response": r["response"],
                "recommendations": games_list,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Saved long format to {out_jsonl}")


if __name__ == "__main__":
    main()
