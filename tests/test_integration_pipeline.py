"""Integration smoke: synthetic EDP through physics hybrid detector."""

from __future__ import annotations

from wind_digital_twin.anomaly.physics_hybrid import fit_physics_hybrid
from wind_digital_twin.anomaly.scoring import threshold_from_training
from wind_digital_twin.config import DEFAULT_THRESHOLD_PERCENTILE
from wind_digital_twin.data.clean import (
    clean_turbine_df,
    get_failure_for_turbine,
    healthy_training_mask,
)
from wind_digital_twin.data.load_edp import load_edp_dataset
from wind_digital_twin.data.synthetic_edp import generate_synthetic_edp_dataset
from wind_digital_twin.eval.protocol import evaluate_turbine


def test_synthetic_edp_physics_hybrid_pipeline(tmp_path):
    generate_synthetic_edp_dataset(tmp_path, random_state=7)
    turbines, failures = load_edp_dataset(tmp_path)

    turbine_id = "T01"
    df = clean_turbine_df(turbines[turbine_id], min_power_kw=50.0)
    failure = get_failure_for_turbine(turbine_id, failures)
    healthy = df.loc[healthy_training_mask(df.index, failure, buffer_days=90)]

    pipeline = fit_physics_hybrid(healthy.iloc[:2000])
    train_scores = pipeline.score(healthy.iloc[2000:4000])
    assert len(train_scores) > 100

    threshold = threshold_from_training(
        train_scores, percentile=DEFAULT_THRESHOLD_PERCENTILE
    )
    test_scores = pipeline.score(df.iloc[4000:5000])
    result = evaluate_turbine(
        turbine_id,
        test_scores,
        threshold=threshold,
        failure=failure,
        persistence_samples=6,
        cooldown_hours=24,
    )
    assert result.has_gearbox_failure
    assert result.scored_days >= 0
