## Data Analysis Report - Logs


### 1) Key performance metrics 
Data: `ui_task_summary.csv`, `tidy_ui_events.csv`
- per participant×condition×task:
  - num_completions: number of TaskCompletion events.
  - mean/median/std duration (seconds) per completion.
  - accuracy: fraction of completions with Correct=True.

Findings:
- Accuracy is very high across the board (median ≈ 1.0), (I suppose that we let them to complete the task in all the conditions.) indicating the tasks were generally solved correctly under all conditions.
- Completion time patterns match the questionnaire results:
  - Controls are fastest in Stationary and remain relatively quick when Moving (median often 3–6 s), supporting Arm anchoring for at‑hand controls while moving.
  - Visual is mid‑range (≈7–13 s), with slightly longer times in Moving/Semi suggesting the cost of reacquiring information when the UI is not world‑fixed.
  - Key is slowest by far (often 25–70+ s, with some very long outliers in Moving), reflecting the complexity and location‑boundedness of the main task. Longer Key times in Moving make a strong case for torso/head anchoring to reduce reacquisition.
- Variability (std) is substantial for Key in Moving/Semi‑Stationary, consistent with heterogeneous strategies and anchor choices and supporting adaptive policies over one‑size‑fits‑all defaults.

Figures:
- Boxplot completion time by task×condition:
  - `figs/boxplot_duration_by_task_condition.png`
- Accuracy by task×condition:
  - `figs/bar_accuracy_by_task_condition.png`
- Median number of completions per participant:
  - `figs/bar_completions_by_task_condition.png`
- Heatmap mean completion time (s):
  - `figs/heatmap_mean_duration.png`


### 2) Motion capture summaries
Data: `motion_summary.csv`
- Capture rates were 10 Hz (Stationary) and 15 Hz (Semi/Moving) with long durations (≈4–13 minutes per condition), and fraction_valid ≈ 1.0 for most recordings.

Figures:
- Average motion recording duration by condition:
  - `figs/bar_motion_duration_by_condition.png`


### 3) Inter-completion intervals 
Data: `inter_completion_intervals.csv`, `inter_completion_summary.csv`

Findings:
- Across participants, mean intervals tend to be lowest in Stationary (≈9–18 s for many users), higher in Semi‑Stationary (≈15–25 s), and highest in Moving for several participants (often >20 s, with a few very large means reflecting pauses/outliers). This pattern suggests more idle/search or multitasking overhead when motion increases.
- Implication: favor anchors that reduce reacquisition while moving (Torso/Head for monitoring; Arm for quick actions) to shrink idle gaps between completions.

