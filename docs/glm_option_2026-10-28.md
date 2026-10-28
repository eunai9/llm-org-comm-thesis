# Option: add GLM as a generator and judge (Oct 28, 2026)

Status: not decided. Waiting for the user.

## Why GLM

- GLM is from Zhipu (Z.ai). It is a model family the project does not have yet.
- As a generator, it adds one more family to the model comparisons.
- As a judge, it can replace the judges we lost. DeepSeek was retired by NVIDIA. qwen fails the JSON check too often.
- It would give the judge-swap test (Q3) a third family that both writes and judges.

## Where it is free

| Route | Model | Notes |
|---|---|---|
| NVIDIA (our key) | `z-ai/glm-5.3-flash` | Same client and pacing as gpt-oss-20b. No new account. |
| Groq | none | Groq serves no GLM model. |
| Local Ollama | none that fit | The laptop has 7 GB RAM and no GPU. |
| OpenRouter | `glm-5.3-flash:free` | New account. Said to allow about 50 requests per day. Too few for generation. |
| Z.ai | GLM-4.7-Flash (free) | New account. No published rate limits. Free-tier terms have changed before. |

## Test on Oct 28, 2026, about 21:00 CET

- One small request to `z-ai/glm-5.3-flash` on NVIDIA.
- It waited 240 seconds for the first word. The answer then came in under a second. So the time is a queue wait, not model speed.
- gpt-oss-20b on the same key worked normally at the same time.
- It is a reasoning model. It returns a `reasoning_content` field, like gpt-oss.
- Our client times out before 240 s and retries, so a probe with our code finished no call.
- Not yet checked: reply quality and whether judge JSON is valid.

## What the wait means

At 4 minutes per call:

- The 239-reply judging would take about 16 hours.
- A full Q1 generation design (thousands of cells) would take weeks.

## Suggested next step

1. Measure the NVIDIA wait at 2–3 other times of day. Each check is one tiny request.
2. If the wait drops to seconds: rerun the probe (3 replies, 6 judge calls with the incoming message). Check reply quality and JSON.
3. If it stays at minutes: use GLM only for a small judge-only run, or try the Z.ai account.

Code change if adopted: add `"z-ai/glm-5.3-flash": "nvidia"` to `_MODEL_PROVIDER` in `src/thesis/analysis/judge_swap.py`, and maybe a longer client timeout.

The throwaway probe script is not in the repo. It used a scratch cache, so nothing was written to `runs/_cache`.

## Sources

- https://openrouter.ai/z-ai/glm-5.3-flash:free
- https://blogs.novita.ai/glm-free-api/
- https://www.developersdigest.tech/models/free-api/zai
- https://freellm.net/models/z-ai-zhipu-ai/glm-4-5-flash-retirement-announced
