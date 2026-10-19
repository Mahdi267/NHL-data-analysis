# Logs used in the blog post (LLM parts)

Each `.jsonl` file has one record per call to Qwen (prompt, answer, token counts, duration, retrieved passages, test results).
Copies of the files in `logs/` (ignored by git). The `_executed.ipynb` files are the notebooks run from start to end.
Some of the runs were supervised by Claude to save time.

## Task 1 — fetch_game_llm and fetch_season_llm

| File | Content |
|---|---|
| chat_2026-10-15_20-47-48.jsonl | Game loop. Vague test 4 message (3 failed refinements), then explicit message (4/4 after 1 refinement) |
| chat_2026-10-16_13-19-58.jsonl | Season v0, 12-digit playoff IDs, refiner cut at 512 tokens |
| chat_2026-10-16_14-00-24.jsonl | Season v0 (2) "one digit each" (best version, 4/5), then the two v1 versions |
| chat_2026-10-19_16-43-45.jsonl | Full rerun of the notebook on 2026-10-19 (game 4/4, season 1/5, 1/5, 4/5) |
| llm_answers_Q1_executed.ipynb | Notebook of the 2026-10-19 rerun |

## Task 3 — tidy data with Qwen

| File | Content |
|---|---|
| chat_2026-10-17_10-07-50.jsonl | Prompt written with Claude's help (traps explained) |
| chat_2026-10-18_16-25-53.jsonl | Context variants C0 to C3 |
| chat_2026-10-18_19-45-31.jsonl | C0 to C3 again, plus goalie_stops (Qwen code and comparison, 40.3 %) |
| chat_2026-10-19_17-07-58.jsonl | Full rerun on 2026-10-19, same results |
| llm_answers_Q3_executed.ipynb | Notebook of the 2026-10-19 rerun |

## Task 4 — PandasAI

| File | Content |
|---|---|
| chat_2026-10-19_09-18-59.jsonl | Example of the assignment (Montreal and Toronto, wrist shots, 2020-21) |
| chat_2026-10-19_10-32-27.jsonl | Training questions, first pass (before the prompt changes) |
| chat_2026-10-19_12-12-39.jsonl | First 8 questions = training questions after the prompt changes. Last 8 questions = official evaluation. All records are labelled pandasai_eval_v0 |
