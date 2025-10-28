from .io import load_questionnaire
from .preprocess import build_tidy_frames
from .quantitative import (
    compute_descriptive_stats,
    compute_preference_distribution,
    run_friedman_by_task,
    compute_condition_effects,
    compute_correlations,
    compute_consistency_agreement,
)
from .qualitative import (
    extract_themes,
    compare_comments_by_condition,
    map_justifications_by_anchor_and_task,
    select_representative_quotes,
)
from .metrics import (
    compute_preference_strength_index,
    compute_anchor_adaptability,
)
from .mixed import (
    triangulate_findings,
)
from .visuals import (
    plot_preference_barcharts,
    plot_heatmaps,
    plot_radar_profiles,
    plot_sankey_planned_vs_rated,
    plot_wordclouds,
)

