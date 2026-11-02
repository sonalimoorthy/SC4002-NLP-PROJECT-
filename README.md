# SC4002 Natural Language Processing — Group Assignment

## Project Overview

This project focuses on building a joint Natural Language Understanding (NLU) system that can understand an English assistant request by performing two tasks:

1. **Intent classification:** predict the user's intent from one of the 60 intent labels in the MASSIVE dataset (for example, `alarm_set`).
2. **Slot filling:** identify relevant spans in the utterance and assign each span one of the 55 slot types (for example, `[time: five am]`).

For every input utterance, the system must produce **both an intent label and its slot type–span pairs**. Both tasks must be handled by a single joint framework; implementing two entirely separate models is not permitted.

## Dataset

We use the **MASSIVE 1.0 English (en-US)** dataset and its original train, validation, and test partitions.

| Split | Partition | Number of examples | Purpose |
|---|---|---:|---|
| Training | `train` | 11,514 | Train model parameters and build the vocabulary |
| Validation | `dev` | 2,033 | Tune hyperparameters and select the best checkpoint |
| Test | `test` | 2,974 | Final evaluation after the model is frozen |
| **Total** | | **16,521** | |

The model must use `utt` as its input. The `intent` field provides the intent-classification label, while `annot_utt` provides the slot-filling annotations.

- Dataset download: <https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz>
- Official documentation: <https://github.com/alexa/massive>

**Dataset rules:** preserve the official splits and all examples; do not add or remove examples, re-split the data, or augment the training set with external or AI-generated task examples. Train using the official training split only.

## Project Stages

### Stage 1 — Independent Baseline (Without Generative AI)

Develop the initial joint intent-classification and slot-filling system independently, without generative AI assistance. This stage establishes the baseline against which the improved system will be compared.

The team will:

- Prepare the dataset and convert utterances and annotations into model inputs and labels.
- Implement a joint model that predicts intents and slot spans/types.
- Train using the training split and use the validation split for model selection.
- Record training and validation losses, evaluation results, elapsed training time, and learning curves.
- Save the best baseline checkpoint and evaluate it on the validation and test splits.

### Stage 2 — AI-Assisted Improvement

After the baseline has been completed and evaluated, collaborate with an AI tool to identify weaknesses and explore improvements. The team will decide which suggestions to accept, modify, or reject, and implement and evaluate the resulting improved system.

The team will:

- Analyse baseline errors and areas for improvement.
- Record the AI tool/model used and preserve the relevant AI collaboration history.
- Evaluate proposed changes and implement selected improvements.
- Train the improved system using the training split, select its checkpoint using validation results, and conduct final validation and test evaluation.
- Compare the Stage 2 system against the Stage 1 baseline.

## Team Responsibilities

The following allocation reflects the work currently assigned. Names for the remaining roles can be filled in once confirmed.

| Team member | Responsibility | Scope |
|---|---|---|
| **Gerald** | Data preprocessing | Prepare and inspect the MASSIVE dataset; preserve the official train/dev/test partitions; convert the raw utterances and annotations into suitable model inputs and labels; coordinate the processed data format with the modelling team. |
| **Sonali** | **Task 1 — Intent classification** (Stage 1 baseline, without AI) | Develop and experiment with the intent-classification component: predict one of the 60 MASSIVE intent labels for each utterance. Coordinate with Harshini so the intent output is integrated into the required joint framework. |
| **Harshini** | **Task 2 — Slot filling** (Stage 1 baseline, without AI) | Develop and experiment with the slot-filling component: identify slot spans and classify them into the 55 slot types. Coordinate with Sonali so the slot output is integrated into the required joint framework. |
| **Eishani** | AI implementation and evaluation — Stage 2 | Lead or coordinate the AI-assisted improvement stage: examine baseline weaknesses, assess AI suggestions, implement selected improvements, train the improved system, and evaluate its validation and test performance. Maintain the relevant AI collaboration records with the team. |
| **Swaytha** | Report (without AI) and team support | Coordinate the report preparation without using AI to write the assigned report content; gather verified methods, experiment results, and contributions from teammates; help other team members with integration, debugging, and outstanding tasks as needed. The AI-collaboration section should still accurately document the team's actual Stage 2 AI use and decisions. |

**Task allocation clarification:** **Sonali is responsible for Task 1 — Intent Classification. Harshini is responsible for Task 2 — Slot Filling.** Both are developing their respective Stage 1 baseline components independently, without generative AI assistance. Although the tasks have separate owners, the final system at both stages must produce the intent label and slot type–span pairs through a single joint framework; entirely separate models for the two tasks are not permitted. Gerald's preprocessing work should support both task owners. Stage 2 is the separate phase in which the team collaborates with AI to diagnose weaknesses and improve the joint system.

## Evaluation

All reported metrics must be expressed as percentages to **two decimal places**.

- **Intent accuracy:** percentage of utterances for which the predicted intent is correct.
- **Slot micro F1:** micro-averaged F1 based on exact `[slot type: slot span]` matches. A prediction counts as a true positive only when both the slot type and the slot span match.

Use the same evaluation functions and scoring rules for every experiment so that the baseline and improved systems can be compared consistently.

## Model and Experiment Requirements

- No pretrained models are allowed, except word vectors.
- The total parameter count for the entire system, including word vectors, must not exceed **20 million parameters**.
- Do not use models already trained or fine-tuned on MASSIVE or related tasks.
- Train or fine-tune the systems yourselves using only the official MASSIVE English training split.
- Use validation results for hyperparameter tuning and checkpoint selection. Keep the test split for final evaluation after the model is frozen.
- Do not manually enter scores, fabricate training logs, or clear notebook outputs before submission. Keep the experiment evidence reproducible.

## Repository Structure

The following is a suggested structure for organising the project files during development. The final submission archive must follow the required three-file format described below.

```text
SC4002-Group-Assignment/
├── README.md
├── main.ipynb                 # Both stages, training logs, outputs and evaluation
├── ai_collaboration.md        # AI tool/model, prompts and collaboration history
├── report.pdf                 # Maximum two pages, excluding the cover page
├── data/                      # Local dataset files; do not include in submission ZIP
├── checkpoints/               # Saved model checkpoints; organise as needed
└── src/                       # Optional supporting source code
```

## Submission Checklist

The final submission must be a ZIP archive named `Group_<ID>.zip` containing **only** these three required files:

- [ ] `main.ipynb` — source code for both stages, including training logs, outputs, and evaluation results.
- [ ] `ai_collaboration.md` or `ai_collaboration.txt` — chat history demonstrating collaboration with AI.
- [ ] `report.pdf` — a maximum of two pages, excluding the cover page.

The report should include the baseline and improved model pipelines, a comparison of validation and test results, AI collaboration and the team's decisions, lessons learned, and at least two error cases from the AI-improved system with analysis of likely causes and limitations.

Do **not** include the dataset archive, environments, cache files, API credentials, or other extra files in the final ZIP.

**Assignment deadline:** Friday, 13 November, 11:59 p.m. SGT.
