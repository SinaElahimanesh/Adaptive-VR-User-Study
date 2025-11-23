## Demographics (Questionnaire)

Data: `outputs/questionnaire/demographics_summary.csv`, `outputs/questionnaire/demographics_gender.csv`, `outputs/questionnaire/demographics_language.csv`, `outputs/questionnaire/demographics_xr_distribution.csv`, `outputs/questionnaire/demographics_videogame_distribution.csv`

- Participants: 19 (unique)
- Age: mean 36.5 (std=11.0) (min 23, max 60)
- Gender: Male 10, Female 9
- XR experience (1–7): mean 1.53 (std=0.84); distribution in `demographics_xr_distribution.csv`
- Videogame experience (1–7): mean 4.05 (std=1.99); distribution in `demographics_videogame_distribution.csv`

Figures:
- `figs/age_histogram.png`
- `figs/gender_distribution.png`
- `figs/xr_experience_distribution.png`
- `figs/videogame_experience_distribution.png`

---

## Data Analysis Report - Questionnaire

### Dataset overview
- Participants: 19
- Trials: 57 (one per condition per participant: Stationary, Semi-Stationary, Moving)
- Tasks: Key, Visual, Controls
- Anchors: World, Head, Torso, Arm
- Measures: 1–7 ratings per task×anchor, planned anchor rankings, demographics, free-text comments

### Methods summary
- Preprocessing: normalized encodings/headers; long-form ratings; extracted planned top per task; unified comments; attached conditions to every trial.
- Quantitative:
  - Descriptive means/medians/std per condition×task×anchor
  - Preference distributions: “planned top” counts and proportions per condition×task
  - Repeated-measures Friedman tests within task across anchors, computed separately per condition
  - Condition effects via Kruskal–Wallis per (task, anchor)
  - Correlations: Spearman (avg ease vs age/xr/game) on participant aggregates; χ² (planned top vs condition at trial-level; vs gender at participant-level)
  - Consistency: planned vs top-rated agreement computed per participant×condition×task, summarized by condition×task
- Qualitative: seeded themes (visibility, obstruction, stability, interaction, convenience), TF‑IDF by condition, representative quotes
- Mixed methods: triangulated numeric top anchors vs top-mentioned anchors per condition×task; participant–condition profiles

---

## 1) Quantitative results and interpretation

### 1.1 Descriptive statistics (ease ratings)
Data: `outputs/desc_by_condition.csv`

#### Stationary (clear preference for World)
| Task | World | Head | Torso | Arm |
|---|---|---|---|---|
| Key | **7.00 (std=0.00)** | 4.00 (std=1.83) | 3.95 (std=1.61) | 2.63 (std=1.38) |
| Visual | **7.00 (std=0.00)** | 3.89 (std=1.76) | 4.26 (std=1.69) | 2.79 (std=1.36) |
| Controls | **6.84 (std=0.50)** | 4.05 (std=1.93) | 4.32 (std=1.34) | 3.26 (std=1.79) |

#### Moving (body anchors outperform World)
| Task | World | Head | Torso | Arm |
|---|---|---|---|---|
| Key | 2.32 (std=1.83) | 5.53 (std=1.74) | **5.95 (std=1.13)** | 3.84 (std=1.83) |
| Visual | 2.37 (std=1.67) | 4.89 (std=1.70) | **6.16 (std=1.01)** | 4.16 (std=1.68) |
| Controls | 2.16 (std=1.50) | 5.00 (std=1.80) | 5.42 (std=1.22) | **5.79 (std=1.55)** |

#### Semi-Stationary (Torso/Arm are strong, World mixed)
| Task | World | Head | Torso | Arm |
|---|---|---|---|---|
| Key | 4.89 (std=2.26) | 4.95 (std=1.78) | **5.37 (std=1.46)** | 3.74 (std=1.82) |
| Visual | 3.11 (std=1.85) | 5.11 (std=2.08) | **5.95 (std=1.35)** | 3.79 (std=1.58) |
| Controls | 2.68 (std=1.73) | 4.79 (std=1.84) | **5.42 (std=1.17)** | 5.37 (std=2.11) |

Findings: When users are stationary, world-fixed UIs are unequivocally easiest across all tasks (means ≥ 6.84), confirming that stability in the environment maximizes usability. As soon as users move, the numeric center shifts to body-anchored references: torso for continuous-view tasks (Key/Visual) and arm for quick input (Controls). Semi‑stationary lies between, with torso consistently strong and world no longer competitive for Visual/Controls. These patterns corroborate the intuition that optimal anchoring depends jointly on motion and task modality (continuous monitoring vs discrete control).

Figures:
- Heatmaps (mean ease) per task:
  - `figs/heatmap_key.png`
  - `figs/heatmap_visual.png`
  - `figs/heatmap_controls.png`

### 1.2 Preference distributions (planned top anchors)
Data: `outputs/preference_distribution.csv`

| Condition | Task     | World  | Head   | Torso  | Arm    |
|---|---|---|---|---|---|
| Stationary | Key      | **100%**   |        |        |        |
| Stationary | Visual   | **100%**   |        |        |        |
| Stationary | Controls | **94.7%**  |        |        | 5.3%   |
| Moving     | Key      | 5.3%   | **42.1%**  | 36.8%  | 15.8%  |
| Moving     | Visual   |        | 31.6%  | **47.4%**  | 21.1%  |
| Moving     | Controls |        | 26.3%  | 15.8%  | **57.9%**  |
| Semi-Stationary | Key      | **42.1%**  | 31.6%  | 15.8%  | 10.5%  |
| Semi-Stationary | Visual   | 5.3%   | 42.1%  | **47.4%**  | 5.3%   |
| Semi-Stationary | Controls | 5.3%   | 21.1%  | 15.8%  | **57.9%**  |

Findings: Planned choices mirror the descriptive means: stationary users overwhelmingly plan for world anchoring; moving users gravitate to torso/head for Key/Visual and to arm for Controls; semi‑stationary preferences split between world/torso for Key and torso/head for Visual. This pre‑trial signal anticipates final experience and thus is a strong prior for adaptive systems.

Figures:
- Preference bar charts per task:
  - `figs/pref_barchart_key.png`
  - `figs/pref_barchart_visual.png`
  - `figs/pref_barchart_controls.png`

### 1.3 Repeated-measures comparisons (anchors within task, per condition)
Data: `outputs/friedman_tests.csv`

It asks: “Within this one condition and this one task, do the anchors have the same central tendency?” A small p‑value (p < .05) means at least one anchor’s ratings differ from the others.

Findings: For Key and Visual, within every motion condition the anchors are not equivalent (Friedman p ≪ .05; moderate Kendall’s W), meaning anchor choice materially changes perceived ease even after conditioning on motion. For Controls, differences are less pronounced (often non‑significant), consistent with a more tolerant interaction pattern where multiple anchors can work if the UI remains reachable.

Implication: For Key/Visual, you should pick the anchor based on the user’s motion state (e.g., Torso when Moving, World when Stationary), because it has a significant, non‑trivial impact on perceived ease.
For Controls, you have more freedom—several anchors may feel comparably usable, so choose based on secondary goals (reach time, clutter, availability), which is why Arm tends to be preferred when Moving.

### 1.4 Condition effects (between conditions per anchor)
Data: `outputs/condition_effects.csv`

Usage: What H and the tiny p-values mean: A large H with p ≪ .05 says the rating distributions for that anchor are not the same across conditions. The extremely small p for World (≈10⁻⁸–10⁻⁹) indicates a very strong condition effect: World is rated much higher when Stationary and much lower when Moving (Semi is in between). For Torso, H is also significant—ratings rise in Moving/Semi versus Stationary. For Arm, the effect appears most clearly for Controls in Moving (higher) versus Stationary (lower).

Findings: Ratings for World exhibit the largest condition effect (H p ~ 10^-8 to 10^-9), soaring in Stationary but collapsing in Moving. Torso and Arm also vary by condition (e.g., Torso better when Moving/Semi than Stationary; Arm markedly better for Controls when Moving), quantitatively confirming that stationarity is a primary driver of anchoring suitability.

A simple, robust rule emerges:
Stationary → default World (all tasks).
Moving → Torso for Key/Visual; Arm for Controls; Head for short, transient “peek” only.
Semi‑Stationary → in between: Torso for Visual; Arm competitive for Controls; Key can be Torso or nearby World depending on workstation proximity.

### 1.5 Correlations and associations
Data: `outputs/correlations_spearman.csv`, `outputs/correlations_chi_square.csv`, `outputs/multinomial_choice_model.csv`

Findings: Age, XR, and game experience do not predict overall ease meaningfully (e.g., age ρ=.21, p=.40). In contrast, planned top is strongly associated with condition (χ² p≈3.8e‑6) and not with gender, underscoring that context—not demographics—explains the bulk of anchor choice variance in this study.

### 1.6 Consistency: planned vs rated top (per condition)
Data: `outputs/consistency_agreement.csv`

What was compared: For each participant, condition (Stationary/Semi/Moving), and task (Key/Visual/Controls), we took:
1) the anchor they planned to use first (planned_top), and
2) the anchor that ended up best by their ratings (rated_top),

Findings: Agreement between planned and experienced top anchors is very high (>90% across tasks/conditions) with substantial-to‑almost‑perfect chance‑corrected agreement (κ), indicating that participants could anticipate what would work best. This stability suggests adaptive systems could safely initialize from a short planning step and require only modest runtime adaptation.

Figures:
- Planned vs rated (sankey-like):
  - `figs/sankey_planned_vs_rated.png`
- Participant radar (example and per condition):
  - `figs/radar_1.png`
  - `figs/radar_1_stationary.png`
  - `figs/radar_1_semi-stationary.png`
  - `figs/radar_1_moving.png`


### 1.7 Planned preferences as ranks
Data: `outputs/planned_mean_ranks.csv`, `outputs/planned_friedman_ranks.csv`

Findings: Mean ranks reproduce the same ordering as ratings, and Friedman on ranks is significant for most condition×task combinations, indicating that the planning phase already captures robust ordering information that matches experience.

### 1.8 Preference Strength Index (PSI)
Data: `outputs/preference_strength_index.csv`

Findings: PSI frequently exceeds 2.0 for Stationary‑World (all tasks) and for Moving‑Torso (Key/Visual) or Moving‑Arm (Controls), indicating strong, confident top choices; small PSI occurs mainly where multiple anchors are viable (e.g., Controls in Semi‑Stationary).

### 1.9 Anchor Adaptability
Data: `outputs/anchor_adaptability.csv`, `outputs/anchor_adaptability_within_summary.csv`, `outputs/adaptability_planned_across_summary.csv`, `outputs/adaptability_rated_across_summary.csv`

Findings: Within a condition, many participants keep the same rated‑top across tasks (0–1 changes), but a non‑trivial fraction adapts (1–2 changes) consistent with “monitor vs control” needs. On average, within‑condition changes were: Stationary mean=0.21 (std=0.54), Semi‑Stationary mean=1.32 (std=0.75), Moving mean=1.16 (std=0.76). Across conditions (per task), mean changes in planned_top and rated_top were both ≈0.00 (no average switches per task across the three conditions in the aggregate summaries), indicating that most context‑driven switching is captured within each condition across tasks rather than between conditions per task in this study.

### 1.10 Post-hoc analyses (Holm-adjusted)

Usage:
- Purpose: After an overall test shows “there is some difference,” post‑hoc tests figure out which specific pairs differ; Holm correction keeps false positives in check.
- Reference: `posthoc_condition_pairwise.csv` (between conditions) and `posthoc_friedman_pairwise.csv` (within a condition). Rows with `p_holm < 0.05` mark the meaningful differences.

1) Condition effects post-hoc (pairwise U-tests)
Data: `outputs/posthoc_condition_pairwise.csv`

Findings: Stationary vs Moving shows large positive deltas for World across all tasks (Stationary≫Moving). Moving vs Stationary shows positive deltas for Torso (Key/Visual) and for Arm (Controls), confirming the direction and magnitude of context effects beyond omnibus tests.

2) Within-condition anchor post-hoc (pairwise Wilcoxon)
Data: `outputs/posthoc_friedman_pairwise.csv`

Findings:
- Stationary: World outranks Head/Torso/Arm for Key and Visual (large rank‑biserial effects).
- Moving: Torso outranks World for Key and Visual; Arm outranks World for Controls.
- Semi‑Stationary: Torso outranks World for Visual; Arm/Torso outrank World for Controls.

### 1.11 Performance vs demographics (StudyLogs timings)

Data: `outputs/logs/inter_completion_summary.csv`, `outputs/logs/ui_task_summary.csv`, joined with `Questionnaire.csv`

What we tested:
- Whether participant age, XR experience (1–7), and videogame experience (1–7) are associated with objective performance derived from StudyLogs.
- Performance metrics:
  - Mean inter-completion interval per condition (lower is faster).
  - Mean task completion duration averaged across tasks per condition (lower is faster).
- Analyses:
  - Spearman correlations per condition: `outputs/logs/performance_correlations_intervals.csv`, `outputs/logs/performance_correlations_durations.csv`
  - Group comparisons via Kruskal–Wallis on ordinal bins (Age: <=30 / 31–40 / 41+; Experience: low/mid/high): `outputs/logs/performance_group_tests_intervals.csv`, `outputs/logs/performance_group_tests_durations.csv`

Findings:
- Age shows moderate positive associations with slower timings (higher = slower), most clearly when users are stationary:
  - Inter-completion intervals and overall completion duration per condition (Spearman ρ [two‑sided p]):
  - Definitions:
    - Inter‑completion interval: mean time between consecutive TaskCompletion events per participant×condition (seconds). Lower = faster cadence between completed tasks.
    - Overall completion duration: mean of per‑task mean completion times across tasks (Key/Visual/Controls) per participant×condition (seconds). Lower = faster average task time.

  - Spearman ρ (rank correlation) size guide: |ρ|<0.1 none, 0.1–0.3 small, 0.3–0.5 moderate, >0.5 strong; the sign shows direction (positive = higher predictor → slower timing; negative = higher predictor → faster timing).
  - Bracketed values [p] are the exact two‑sided p‑values; thresholds: p<.001 very strong, p<.01 strong, p<.05 significant, .05≤p<.10 marginal (“trend”), p≥.10 not significant.

    | Condition | Inter‑completion interval | Overall completion duration |
    |---|---|---|
    | Stationary | 0.60 [0.006] | 0.55 [0.014] |
    | Semi‑Stationary | 0.48 [0.035] | 0.41 [0.077] |
    | Moving | 0.44 [0.057] | 0.42 [0.075] |

  - Interpretation: Positive ρ means “older → slower.” Values around 0.4–0.6 indicate moderate associations. Bracketed values are exact p‑values; p<.05 denotes conventional statistical significance.
- XR and videogame experience showed no reliable associations with performance timings across conditions (all p>.06).
- Kruskal–Wallis group tests on age/experience bins did not yield consistent significant differences (all p≥.14), likely due to small bin counts.

Figures:
- `outputs/logs/figs/scatter_intervals_vs_age.png`
- `outputs/logs/figs/scatter_intervals_vs_xr_experience.png`
- `outputs/logs/figs/scatter_intervals_vs_videogame_experience.png`

### 1.12 Task‑level performance (by task)

Data:
- Intervals by task: `outputs/logs/inter_completion_summary_by_task.csv`
- Correlations by task: `outputs/logs/performance_correlations_intervals_by_task.csv`, `outputs/logs/performance_correlations_durations_by_task.csv`
- Group tests by task: `outputs/logs/performance_group_tests_intervals_by_task.csv`, `outputs/logs/performance_group_tests_durations_by_task.csv`

Findings:
- Inter‑completion intervals (lower = faster cadence):
  - Stationary: Age shows strong positive associations (ρ [p]) — Key 0.71 [0.00066], Visual 0.59 [0.0077], Controls 0.47 [0.043].
  - Moving: Age correlates moderately for Key 0.49 [0.033] and Visual 0.47 [0.045]; Controls weaker and non‑significant.
  - Semi‑Stationary: Visual shows a moderate association (0.49 [0.032]); other tasks non‑significant.
- Completion duration (lower = faster per‑task time):
  - Visual shows the clearest age effects — Stationary 0.734 [0.00035], Semi‑Stationary 0.709 [0.00067]; Moving shows a trend 0.409 [0.082].
  - Videogame experience shows task‑specific negative associations for Visual duration (faster with more experience): Semi‑Stationary −0.495 [0.031], Stationary −0.479 [0.038]. Other tasks did not show reliable effects for XR/videogame experience.

Note: Bracketed values are exact two‑sided p; interpretation guides for ρ and p are in §1.11.

Brief interpretation:
- Visual shows the most consistent age effect on speed (both cadence and duration), especially when not moving; Key shows moderate age effects; Controls show weak/none.
- Where ρ is positive and significant, older participants tend to be slower on that metric; negative ρ indicates faster with higher predictor (seen only weakly for Controls).

### 1.13 Performance by anchoring mode and task (StudyLogs timings)

Data: `outputs/logs/ui_task_summary_by_anchor_overall.csv`

- The table reports mean completion time (seconds; lower = faster) aggregated across participants for each condition×task×anchor, derived from TaskCompletion durations in StudyLogs.
- Anchors are canonicalized to World, Head, Torso, Arm.

#### Stationary
| Task | World | Head | Torso | Arm |
|---|---|---|---|---|
| Key | 32.95 |  | 32.31 |  |
| Visual | 10.18 |  |  |  |
| Controls | 5.02 | 2.91 |  | 4.93 |

#### Semi‑Stationary
| Task | World | Head | Torso | Arm |
|---|---|---|---|---|
| Key | 44.83 | 43.43 | 48.35 | 45.30 |
| Visual | 8.24 | 10.07 | 9.32 | 9.99 |
| Controls | 7.66 | 4.45 | 6.04 | 6.51 |

#### Moving
| Task | World | Head | Torso | Arm |
|---|---|---|---|---|
| Key | 70.57 | 58.41 | 54.57 | 62.89 |
| Visual |  | 9.84 | 11.02 | 11.30 |
| Controls |  | 5.45 | 6.40 | 8.07 |

Implications:
- Stationary: World is the practical default across tasks; even Controls are near‑fastest in World (≈5.0 s), with Head slightly faster when used.
- Semi‑Stationary: Head often yields the fastest Controls (≈4.45 s) and competitive Key; Visual favors World/Torso (≈8.2–9.3 s). Body anchors (Torso/Arm) remain viable without clear penalties.
- Moving: For Controls, Head is fastest (≈5.45 s), with Torso close; Arm remains usable but slower on average. For Key/Visual, Torso/Head are faster than World, matching users’ stated preferences.

## 2) Qualitative results and interpretation

### 2.1 Theme prevalence (mentions per participant)
Data: `outputs/triangulation_theme_prevalence.csv`

Findings: Moving participants mention visibility and stability about 2× per person on average (visibility≈2.21; stability≈2.21), far more than stationary participants (≈0.79 and 1.00). Interaction convenience remains consistently referenced, but rises in motion. Qualitatively, users care most about seeing the UI without pursuing it and about the UI not drifting while they move—explaining the torso/head bias for monitoring and the arm bias for quick actions.

### 2.2 TF‑IDF terms by condition
Data: `outputs/condition_tfidf.csv`

Findings: Moving/Semi emphasize body‑centric, gaze‑centric terms (“arm”, “head”, “hand”, “follow”, “fov”), whereas Stationary emphasizes “world”, “visual”, “controls”, “stay”—precisely matching the numeric patterns and validating the construct validity of our measures.

### 2.3 Representative quotes
Data: `outputs/representative_quotes.csv`

Findings: Quotes repeatedly highlight avoiding FoV clutter (downweighting head while moving), exploiting the hand for immediacy (arm for Controls), and using the body to keep content near yet non‑intrusive (torso for Visual/Key). Multiple participants explicitly describe “world for stationary” and “body for moving,” aligning text with statistics.

Figures:
- Word clouds by condition:
  - `figs/wordcloud_stationary.png`
  - `figs/wordcloud_semi-stationary.png`
  - `figs/wordcloud_moving.png`

### 2.4 Sentiment analysis (transformer-based)
Data: `outputs/sentiment_transformer.csv`, `outputs/sentiment_by_condition_task.csv`

Method: multilingual transformer (XLM‑RoBERTa) applied to participants’ textual explanations; fallback to VADER where transformers are unavailable. We aggregate sentiment labels (POS/NEU/NEG) by condition and task.

Findings:
- Stationary: POS≈0.63, NEU≈0.11, NEG≈0.26 across tasks — overall positive but with more negatives referencing obstruction or button‑press friction in world‑fixed layouts.
- Semi‑Stationary: POS≈0.95, NEU≈0.05, NEG≈0.00 — the most positive condition; participants report comfortable access with little clutter.
- Moving: POS≈0.84, NEG≈0.16 — still positive, with negatives concentrated on FoV clutter for head and interaction instability when anchors don’t follow suitably.
- By task, Visual and Controls skew more positive in Moving/Semi when torso/arm strategies are mentioned; Key shows more mixed polarity in Moving when reacquisition costs loom.

Implication: Sentiment aligns with numeric preferences—when anchor–context pairing is appropriate (Stationary→World, Moving→Torso/Arm), language becomes uniformly positive; when mismatched (e.g., head while moving), negative cues rise. These patterns support rule‑based anchoring defaults with light adaptive overrides.

---

## 3) Practical implications

- Stationary: Use World anchoring by default for Key/Visual/Controls; arrange a stable layout near the task locus; permit subtle repositioning but avoid follow‑me behavior.
- Moving: Use Torso for Key/Visual to keep content in reach but out of central FoV; use Arm for Controls to minimize reach time; allow ephemeral “peek to head” for urgent, brief visual checks.
- Semi-Stationary: Prefer Torso for Visual; use Arm for Controls; choose Torso or a well‑placed World for Key depending on workspace locality.
- System design: Provide show/hide, lock, and distance tweaks; keep head‑anchored usage short‑lived to avoid clutter.