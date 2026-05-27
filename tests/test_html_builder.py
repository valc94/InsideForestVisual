from html_builder import (
    DatasetConfig,
    EvaluationConfig,
    Experiment,
    Rule,
    apply_missing_strategy,
    build_html,
    build_target_labels,
    california_housing_rows,
    compute_stats,
    evaluate_rule,
    evaluate_rule_set,
    invented_iris_experiments,
    invented_titania_experiments,
    invented_titanic_experiments,
    invented_housing_experiments,
    iris_rows,
    run_demo,
    run_titanic_exam,
    segment_stats,
    sample_rows_if_large,
    titania_rows,
    titanic_rows,
    tvd,
)


def test_compute_stats_and_scale_pipeline():
    rows = iris_rows(seed=7)
    stats = compute_stats(rows, ["sepal_length", "sepal_width"])
    assert "sepal_length" in stats
    assert stats["sepal_width"]["std"] > 0


def test_evaluate_rule_and_set():
    row = {"x": 2.0, "scaled": {"x": -0.4}}
    assert evaluate_rule(row, Rule("x", 1.9, 2.1, "raw"))
    assert evaluate_rule(row, Rule("x", -1.0, 0.0, "scaled"))
    assert evaluate_rule_set(row, [Rule("x", 1.9, 2.1, "raw")])


def test_tvd_simple_case():
    stats = {
        "onlyA": {"count": 10, "dist": {"A": 10, "B": 0}},
        "onlyB": {"count": 10, "dist": {"A": 0, "B": 10}},
        "intersection": {"count": 0, "dist": {"A": 0, "B": 0}},
    }
    assert tvd(stats, ["A", "B"], 1, 1) == 1.0


def test_segment_stats_shapes():
    rows = iris_rows(seed=1)
    exp = invented_iris_experiments()[0]
    # precompute scaled via html build path
    cfg = DatasetConfig("iris", ["sepal_length", "sepal_width", "petal_length", "petal_width"], "species")
    html = build_html(rows, cfg, EvaluationConfig(), [exp])
    assert "Ranking (TVD)" in html


def test_missing_strategy_drop_and_mean():
    rows = [{"a": 1.0}, {"a": None}, {"a": 3.0}]
    dropped = apply_missing_strategy(rows, ["a"], "drop")
    assert len(dropped) == 2
    filled = apply_missing_strategy(rows, ["a"], "mean")
    assert filled[1]["a"] == 2.0


def test_run_demo_creates_both_html(tmp_path):
    paths = run_demo(tmp_path)
    assert paths["iris"].exists()
    assert paths["titania"].exists()
    assert paths["titanic"].exists()
    assert "titania_sintetico" in paths["titania"].read_text(encoding="utf-8")
    assert "titanic_sintetico" in paths["titanic"].read_text(encoding="utf-8")


def test_invented_experiments_present():
    assert len(invented_iris_experiments()) >= 2
    assert len(invented_titania_experiments()) >= 2


def test_sampling_not_applied_when_under_threshold():
    rows = iris_rows(seed=3)
    sampled, info = sample_rows_if_large(rows, "species", threshold=2500, sample_size=500, strategy="stratified", seed=1)
    assert len(sampled) == len(rows)
    assert info["applied"] is False


def test_sampling_applied_when_over_threshold():
    rows = iris_rows(seed=3) * 30  # 4500 rows
    sampled, info = sample_rows_if_large(rows, "species", threshold=2500, sample_size=1200, strategy="stratified", seed=1)
    assert info["applied"] is True
    assert len(sampled) == 1200


def test_regression_deciles_labeling():
    rows = california_housing_rows(seed=2)[:200]
    labels, mode = build_target_labels(rows, "price_k", "regression_deciles", 10)
    assert mode == "regression_deciles"
    assert len(labels) == 200
    assert labels[0].startswith("Q")


def test_housing_html_generation():
    rows = california_housing_rows(seed=2)
    cfg = DatasetConfig(
        dataset_id="housing_test",
        feature_columns=["income_k", "rooms", "distance_km", "crime_index"],
        target_column="price_k",
        target_mode="regression_deciles",
        regression_bins=10,
    )
    html = build_html(rows, cfg, EvaluationConfig(min_samples_only_a=5, min_samples_only_b=5), invented_housing_experiments())
    assert "housing_test" in html
    assert "target_mode_info" in html


def test_titanic_experiments_present_and_rows():
    assert len(invented_titanic_experiments()) >= 2
    rows = titanic_rows(seed=9)
    assert len(rows) > 600
    assert {"survived", "fare", "pclass"}.issubset(rows[0].keys())


def test_run_titanic_exam_binary_output(tmp_path):
    out = run_titanic_exam(output_dir=tmp_path, require_all_classes_in_segment=True)
    text = out.read_text(encoding="utf-8")
    assert "titanic_sintetico_examen" in text
    assert "target_cardinality_info" in text
    assert "\"is_binary\": true" in text.lower()
