1. Add a multi-step prompt pipeline (no training needed)

- Trend report: fetch top 5–10 niche topics (Twitter/X search or public trend APIs).
- Content ideas: generate 10 tweet angles from trends + brand context.
- Final post: choose 3 ideas, generate 3 candidates with varied tones.
- A/B select the best candidate using a heuristic scorer (length, clarity, uniqueness, emoji/hashtag limits).
- image search and attachment to post
- Save all artifacts (trend→ideas→candidates→chosen) to `data.jsonl`.

3. Brand strategy integration

- Load `Brand Strategy Report.md` into a short “brand card” (voice, topics, boundaries).
- Use the brand card in every pipeline stage as context.

4. Metric loop (lightweight first)

- After 24–48h, fetch metrics for the last N posts; persist into `data.jsonl`.
- Compute simple reward = likes*2 + reposts*3 + views.
- Use reward to pick which prompts/styles to bias toward next day (no training yet).

5. Quality controls

- Hard constraints: tweet length, 0–2 emojis, 0–2 hashtags, no generic claims.
- Deduplicate against last 30 posts via embedding cosine sim threshold.
- Optional human veto window before posting (env flag).

6. Only then consider training

- If hosted quality/cost is good: try small hosted fine‑tune (few hundred examples).
- If you want control/cost savings: migrate to Runpod with `meta-llama/Meta-Llama-3.1-8B-Instruct` or `mistralai/Mistral-7B-Instruct`, LoRA via Unsloth/TRL. Keep the same dataset and pipeline.

7. Roadmap (later)

- Multi-platform adapters (LinkedIn, Threads) behind a common post interface.
- Simple image generation (1:1, minimal text) for posts that benefit from visuals.
- Weekly “what worked” report with examples and next-week plan.

Immediate edits

- `helpers/GRPO_Runpod.py`: add a `provider` switch (hosted vs local) and swap default model off Qwen.
- `reinforcement_learning_loop.py`: generate 3 candidates, score, post best, log all.
- `helpers/twitter_helpers.py`: stricter error handling; confirm metrics fields mapping.
- New: `helpers/pipeline.py` with functions: `get_trends()`, `gen_ideas()`, `gen_candidates()`, `score_candidates()`, `choose_and_post()`.
