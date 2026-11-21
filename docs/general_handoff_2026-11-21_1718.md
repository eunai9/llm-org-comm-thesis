# Session handoff — 2026-11-21 17:18

Read this first, then  sections 22–24 on .

## Where things live

- WSL repo : branch . Holds the code and . Run Python here.
- WSL repo : branch . Holds , figures, manifests, and handoffs. It has no venv.
- Convention used so far: code is committed on the branch, while the log, figures and manifests are committed on .
- OneDrive folder (Windows side) holds the deck , backups in , and .

## Done this session (details are in the commits, not repeated here)

- Tone judge, second judge gpt-oss-20b:  section 22, commits  and  on .
- Stakes and incoming tone inside each direction, for the qwen main set (all 1,440 replies):
  - Code: , commit  on the branch.
  - Write-up:  section 24, commit  on .
  - Figure: .
  - Manifest: .
  - The other 690 qwen replies were scored with gpt-oss-120b on Groq. All 1,440 tone scores are now in the response cache.
- Deck edits for "Final Result for Thesis", all saved to OneDrive except the last one:
  - Slide 4: qwen examples only, plus a "Why qwen" box.
  - qwen measures slide.
  - "Why gpt-oss-120b is the judge" slide.
  - Q2 slide, qwen only.
  - Open problems slide.
  - New slide 3: Personas and Situations.

## Open: deck save is pending

- The stakes and tone slide (new slide 7, after "Result — qwen3.8-27b on all three measures") is built and validated. It is NOT yet in the OneDrive deck.
  - The copy failed with "Device or resource busy", most likely because the deck is open in PowerPoint.
  - The built file is  in the previous session's scratchpad (Windows temp). That scratchpad may be gone. If so, rebuild the slide; see below.
- Before saving, check the deck's md5. It was  (equal to ) when last checked on 2026-11-21.
  - If the md5 differs, the user edited the deck by hand. Work from their current file and never overwrite their edits.
- A backup already exists: .
- Slide content:
  - Two clustered bar charts with down, peer and up bars.
  - Left chart: high stakes vs. routine. Tone score +0.37 / +0.67 / +0.60. Politeness log-odds −0.55 / −0.35 / −0.47.
  - Right chart: assertive message vs. neutral. Tone score −0.15 / +0.24 / +0.42. Politeness −0.33 / −0.48 / −0.42.
  - Notes under the charts, and a takeaway box.
  - Numbers come from the manifest above.
- Deck tooling:
  - python-pptx with .
  - Validate with the pptx skill's .
  - Render with PowerPoint COM through Windows PowerShell 5.1
Copyright (C) Microsoft Corporation. All rights reserved..
  - Styling: CONTENT layout; title shape "Text 0"; shape named "lead"; colors NAVY 1E2761, DOWN 3A5BA9, PEER 7E8AA0, UP C0601A; Calibri.

## Other open offers (not started)

- The layout footer "Q1 — examples and results" is wrong on the Q2 and open-problems slides (now slides 10 and 11 once slide 7 is in).
- The title slide still says "Progress update" and "October 21, 2026".
- The old all-model tone slide and the qwen measures slide have similar titles.
- Add section 24 and the per-item Q2 rubric numbers to .
- The stakes and tone result is not checked by results-verifier yet. Consider it before the result goes into a chapter.

## Gotchas

- Groq runs need  first. Run long jobs in tmux.
- The Windows-side  is not on PATH. Use WSL python.
- Windows-side writes can add CRLF. Check with diff before committing.

## Suggested skills

-  for any deck work.
-  before making or changing a chart.
-  before claiming the deck is saved or a result holds.
