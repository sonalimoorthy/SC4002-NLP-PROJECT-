# AI Collaboration Record — SC4002 Group Assignment

## 1. AI tool and model

| Item | Details |
|---|---|
| Platform | Claude (Anthropic), claude.ai web/app |
| Model | Claude Opus 5.5 |
| Used for | Stage 2 (AI implementation): diagnosing baseline weaknesses, designing and coding the improved model in the notebook, explaining results, Colab/GitHub workflow help |
| Not used for | Stage 1 baseline (written by the team) |
| Full chat history | [https://claude.ai/share/5b1eb5ea-4afe-43c3-88b2-6588b01f322b](https://claude.ai/share/5b1eb5ea-4afe-43c3-88b2-6588b01f322b) |

## 2. How the AI was used

1. The team's Stage 1 notebook (`main.ipynb`) was copied, and, as agreed within the team, the Stage 2 sections were added to the same notebook directly after the Stage 1 predictions.
2. The AI wrote the Stage 2 code: a baseline error analysis on the dev set, a table of improvement suggestions with accept / modify / reject decisions, and three incremental models (S2-A, S2-B, S2-C). The code was checked by the AI only on a synthetic dataset of the same format; all reported scores come from the team's own run.
3. The team ran the full notebook in Google Colab. No GPU was available on the free tier, so the run was on CPU (about 1–1.5 hours). The notebook was saved back to GitHub with all outputs.
4. The team shared the outputs with the AI, which helped interpret them (before/after diagnosis, error cases, limitations).
5. The team reviewed each AI suggestion and confirmed the accept / modify / reject decisions in Section 20 of the notebook. The report was written by the team without AI.

## 3. Prompt history

The prompts below are those relevant to the Stage 2 design, implementation and analysis, in order. Purely operational prompts (GitHub uploads, Colab runtime and timing questions) are omitted here but appear in the full transcript (Section 7). Prompts are paraphrased for clarity; the original wording is in the transcript.

| # | Prompt (team → AI) | AI response (summary) |
|---|---|---|
| 1 | Asked the AI to take a copy of the team's Stage 1 notebook (`main.ipynb`) and add a "Stage 2: AI Implementation" section directly after the Stage 1 prediction cells, so both stages live in one notebook. | Attempted to look up the repo; access was declined, so it stopped and asked for the notebook (upload or repo name) and the Stage 2 requirements. |
| 2 | Provided the assignment brief so the Stage 2 design would follow its constraints. | Planned Stage 2 within the brief's rules (no pretrained models except word vectors, ≤ 20M parameters): baseline error analysis; candidate improvements (GloVe, character embeddings, CRF, intent-guided slots, regularisation + early stopping) each marked accept / modify / reject; training on train only, checkpoint chosen on validation; intent accuracy and slot micro-F1 on validation and test (2 d.p.); comparison table, plots, ≥ 2 error cases. Noted the team must run training itself (no fabricated scores or logs). |
| 3 | Provided `main.ipynb` and asked the AI to implement Stage 2 on top of the existing pipeline. | Created `main_stage2.ipynb` with Stage 1 cells untouched. Added: baseline error analysis (verified its scoring matches the team's `get_scores`; flagged five weaknesses); a suggestions table (6 accepted, 3 modified, 5 rejected, e.g. BERT and data augmentation rejected as forbidden); three incremental models S2-A/S2-B/S2-C (~1.5M parameters each); dev-only model selection then one test run; comparison table, learning curves, before/after diagnosis and error cases. Code was tested only on a synthetic dataset of the same format, so no real scores were claimed. |
| 4 | Shared a PDF export of a partial run and asked the AI to interpret it. | Run was on CPU and stopped during S2-B. Confirmed baseline weaknesses (17.90% of dev utterances with unknown words; 172 invalid `I-` tags; best epoch 15/15) and S2-A's early gain. Recommended a GPU runtime. |
| 5 | Shared the PDF export of the completed run. | Run succeeded; S2-C selected on dev. Reported Stage 1 vs Stage 2 table (Section 4). Flagged that Sections 28–30 had no output in the PDF and should be checked. Noted limitations: S2-A/S2-B peaked near the 30-epoch limit; CPU, single seed. |
| 6 | Shared the Section 28 output (Stage 2 dev-set diagnosis). | Interpreted the before/after diagnosis: unknown-word gap reduced, invalid tags 172 → 0, missed slots 1,016 → 144; trade-off of more spurious / wrong-type slots; remaining hard cases (`general_quirky`, `calendar_set` vs `calendar_query`, creative-work names). Noted a rounding difference (16.03 vs 16.04). |
| 7 | Shared the Section 29 output (test-set error examples). | Reminded the team the report must be written without AI. Suggested candidate error cases with likely causes for the team to verify: "get sia's cheap thrills ready to play next" (unknown names → `general_quirky`), "what is mr. taxi's phone number…" (right span, wrong type), and a possible annotation-noise example. |
| 8 | Shared the Section 30 output (custom example utterances). | Explained the three predictions; "6 pm" mis-tagged because training data mostly spells numbers as words. Confirmed the notebook had run to completion. |
| 9 | Asked for a summary of the baseline weaknesses and how each was addressed. | Gave W1–W5 evidence, fixes and results (Section 4). |
| 10 | Asked whether the remaining weaknesses (spurious slots, `general_quirky`, media-name slots, digit tokens) could be reduced further. | Warned that changing the model after seeing test results would be tuning on test; assessed each limitation (Section 5) and recommended stopping. |
| 11 | Agreed to freeze the model at S2-C. | Stage 2 frozen; listed wrap-up steps. |

## 4. AI-proposed improvements and results (dev set unless stated)

| Weakness | Evidence (baseline) | Fix | Result |
|---|---|---|---|
| W1 Undertrained | Best epoch was the last (15/15); val loss still falling; SGD lr 0.05 | Adam (lr 0.001), gradient clipping, up to 30 epochs with early stopping on dev, dropout | S2-A (W1 + LSTM): intent 84.80%, slot F1 71.94% (baseline 70.98% / 43.31%) |
| W2 Weak encoder | Vanilla tanh RNN; intent uses only final hidden state | 2-layer BiLSTM; attention pooling for intent | Part of S2-A gain; attention contributes to S2-C's gain |
| W3 Unknown words | 17.90% of dev utterances contain OOV words; `<UNK>` vector never trained; slot F1 22.76% vs 47.45% | GloVe vectors, character-level features, word dropout | On OOV utterances: slot F1 22.76% → 64.29%, intent 57.14% → 75.55% |
| W4 Broken / missed spans | 172 invalid I- tags; 1,016 gold slots missed | CRF output layer | Invalid tags 172 → 0; missed 1,016 → 144; boundary errors 191 → 129; correct slots 693 → 1,571 |
| W5 No task sharing | Slot layer ignores intent | Sentence (intent) vector fed to slot layer at every word | Added together with attention + CRF in S2-C; S2-C slot F1 78.06% |

**Overall**

| Split | Metric | Stage 1 baseline | Stage 2 (S2-C) | Change |
|---|---|---|---|---|
| Dev | Intent accuracy | 70.98% | 87.01% | +16.03 |
| Dev | Slot F1 | 43.31% | 78.06% | +34.75 |
| Test | Intent accuracy | 69.20% | 85.64% | +16.44 |
| Test | Slot F1 | 41.92% | 76.60% | +34.68 |

Model size: about 1.37M parameters.

## 5. Suggestions rejected or not pursued

| AI suggestion | Decision | Reason |
|---|---|---|
| Optional S2-D: convert digits to words ("6" → "six") | Not pursued | Test results had already been seen; any further change would need to be selected on dev only and disclosed. Expected test gain was small. |
| Train S2-B/S2-C longer, or tune dropout / hidden size | Not pursued | Likely ~1 point gain at ~1 hour per run; risk of indirect test tuning. |
| Gazetteer of song / artist names | Rejected | External task data, forbidden by the brief. |
| Pretrained transformers (e.g. BERT) and data augmentation | Rejected (in the Section 20 decisions table) | Forbidden by the brief. |

**Remaining limitations (reported in the report):**
- Spurious slots rose 89 → 162; right-span-wrong-type errors rose 38 → 85 (trade-off for 878 extra correct slots).
- `general_quirky` intent still ~50% accurate (catch-all class, overlaps with `qa_factoid` / `news_query`, possible label noise).
- Song, audiobook and radio-station names remain the worst slot types.
- Numbers typed as digits ("6 pm") are still unknown words.

## 6. Team verification

- All AI-generated code was run end-to-end in Colab and outputs were saved in `main.ipynb`.
- Every number above was checked against the notebook outputs.
- Accept / modify / reject decisions for each AI suggestion are recorded in Section 20 of the notebook.
- The report was written in the team's own words.

## 7. Full transcript

Shared chat link (full, unedited conversation): [https://claude.ai/share/5b1eb5ea-4afe-43c3-88b2-6588b01f322b](https://claude.ai/share/5b1eb5ea-4afe-43c3-88b2-6588b01f322b)
