# Results / Результаты

Fine-tuning ruGPT3 + LoRA on 3303 chapter examples (train 2810 / val 226 / test 267). Google Colab, T4 GPU.
Дообучение ruGPT3 + LoRA на 3303 примерах-главах (train 2810 / val 226 / test 267). Google Colab, T4.

## Runs / Запуски

| Run | Model | Window | Epochs | Best val loss / perplexity |
|---|---|---|---|---|
| v1 | ruGPT3 small | 1024 | 3 | 3.148 / 23.3 |
| v2 | ruGPT3 small | 1024 | 8 (best step ~700) | 3.125 / 22.8 |
| v3 | ruGPT3 medium | 2048 | 6 (best step 528) | 2.825 / 16.9 |

Perplexity before and after fine-tuning (val / test): small 26.1 → 22.8 / 23.1 → 19.3; medium 18.9 → 16.9 / 16.6 → 14.5.
Perplexity of runs with different window length is only roughly comparable.

## Prompt following (50 test prompts, chapter up to 350 tokens) / Следование подсказке

| Model | Hero coverage | All heroes named | Junk | 3-gram repeats | Words per chapter |
|---|---|---|---|---|---|
| reference (real chapters) | 51% | 18% | 0% | 0.7% | 211 |
| small: base | 14% | 4% | 30% | 7.6% | 214 |
| small: v1 | 30% | 2% | 0% | 0.4% | 250 |
| small: v2 | 28% | 2% | 0% | 0.9% | 252 |
| medium: base | 29% | 4% | 38% | 4.4% | 209 |
| medium: v3 | 27% | 2% | 0% | 0.3% | 247 |

Fine-tuning removes junk and repeats and lowers perplexity. Hero coverage doubles for small, but does not improve for medium: following the prompt is the main open problem.
Дообучение убирает мусор и повторы и снижает перплексию. Охват героев у small вырос вдвое, у medium не вырос: следование подсказке остаётся главной проблемой.

## Figures / Графики

- `v3/training_curves.png` — loss and perplexity, run v3
- `v3/extra_charts.png` — example length, learning rate, perplexity before/after, val loss of all runs
- `prompt_following.png` — prompt-following metrics

Adapters, full corpus texts and API keys are not stored in the repository.
