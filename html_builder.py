from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass
class Rule:
    field: str
    min: float
    max: float
    native: str = "raw"  # raw|scaled


@dataclass
class Experiment:
    id: int
    name: str
    description: str
    target_column: str
    rules: dict[str, list[Rule]]
    enabled: bool = True
    tags: list[str] = field(default_factory=list)


@dataclass
class DatasetConfig:
    dataset_id: str
    feature_columns: list[str]
    target_column: str
    id_column: str | None = None
    missing_strategy: str = "drop"
    scaling_mode_default: str = "zscore"  # raw|zscore|minmax|robust
    random_seed: int = 98765
    sampling_threshold: int = 2500
    sample_size: int = 2500
    sampling_strategy: str = "stratified"  # stratified|random
    target_mode: str = "classification"  # classification|regression_deciles
    regression_bins: int = 10


@dataclass
class EvaluationConfig:
    primary_metric: str = "tvd"
    min_samples_only_a: int = 1
    min_samples_only_b: int = 1
    ranking_order: str = "desc"


class ValidationError(ValueError):
    pass


def load_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, Any]] = []
        for row in reader:
            out: dict[str, Any] = {}
            for k, v in row.items():
                if v is None:
                    out[k] = None
                    continue
                v = v.strip()
                if v == "":
                    out[k] = None
                    continue
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
            rows.append(out)
    if not rows:
        raise ValidationError("CSV sin filas")
    return rows


def apply_missing_strategy(rows: list[dict[str, Any]], features: list[str], strategy: str) -> list[dict[str, Any]]:
    if strategy == "drop":
        return [r for r in rows if all(r.get(f) is not None for f in features)]
    if strategy in {"mean", "median"}:
        # median approximated for brevity as mean in this utility
        fills: dict[str, float] = {}
        for f in features:
            vals = [float(r[f]) for r in rows if isinstance(r.get(f), (int, float))]
            fills[f] = mean(vals) if vals else 0.0
        out = []
        for r in rows:
            nr = dict(r)
            for f in features:
                if nr.get(f) is None:
                    nr[f] = fills[f]
            out.append(nr)
        return out
    raise ValidationError(f"missing_strategy no soportado: {strategy}")


def sample_rows_if_large(
    rows: list[dict[str, Any]],
    target_column: str,
    threshold: int,
    sample_size: int,
    strategy: str,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Sample rows when dataset is very large.

    Sampling activates only when len(rows) > threshold.
    """
    original_n = len(rows)
    if original_n <= threshold:
        return rows, {
            "applied": False,
            "strategy": "none",
            "original_rows": original_n,
            "final_rows": original_n,
        }

    if sample_size <= 0:
        raise ValidationError("sample_size debe ser > 0")

    n = min(sample_size, original_n)
    rng = random.Random(seed)

    is_numeric_target = all(isinstance(r.get(target_column), (int, float)) for r in rows)
    effective_strategy = "random" if is_numeric_target and strategy == "stratified" else strategy

    if effective_strategy == "random":
        sampled = rng.sample(rows, n)
    elif effective_strategy == "stratified":
        by_class: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            cls = str(r[target_column])
            by_class.setdefault(cls, []).append(r)

        # proportional class allocation with deterministic rounding
        class_keys = sorted(by_class.keys())
        allocations: dict[str, int] = {}
        remaining = n
        for i, cls in enumerate(class_keys):
            if i == len(class_keys) - 1:
                allocations[cls] = remaining
            else:
                frac = len(by_class[cls]) / original_n
                k = max(1, int(round(n * frac)))
                k = min(k, len(by_class[cls]), remaining)
                allocations[cls] = k
                remaining -= k

        sampled = []
        for cls in class_keys:
            cls_rows = by_class[cls]
            k = min(allocations[cls], len(cls_rows))
            sampled.extend(rng.sample(cls_rows, k))

        # adjust in case of edge rounding mismatch
        if len(sampled) < n:
            missing = n - len(sampled)
            leftovers = [r for r in rows if r not in sampled]
            sampled.extend(rng.sample(leftovers, min(missing, len(leftovers))))
        elif len(sampled) > n:
            sampled = rng.sample(sampled, n)
    else:
        raise ValidationError(f"sampling_strategy no soportado: {strategy}")

    return sampled, {
        "applied": True,
        "strategy": effective_strategy,
        "original_rows": original_n,
        "final_rows": len(sampled),
        "threshold": threshold,
        "sample_size": sample_size,
    }


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def build_target_labels(rows: list[dict[str, Any]], target_column: str, mode: str, bins: int) -> tuple[list[str], str]:
    if mode == "classification":
        return [str(r[target_column]) for r in rows], "classification"
    if mode != "regression_deciles":
        raise ValidationError(f"target_mode no soportado: {mode}")
    if bins < 2:
        raise ValidationError("regression_bins debe ser >= 2")
    vals = [float(r[target_column]) for r in rows]
    sorted_vals = sorted(vals)
    edges = [_quantile(sorted_vals, i / bins) for i in range(bins + 1)]
    labels = []
    for v in vals:
        idx = bins - 1
        for i in range(bins):
            if edges[i] <= v <= edges[i + 1]:
                idx = i
                break
        labels.append(f"Q{idx+1}")
    return labels, "regression_deciles"


def compute_stats(rows: list[dict[str, Any]], features: list[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for f in features:
        vals = [float(r[f]) for r in rows]
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / len(vals)
        std = math.sqrt(var) or 1.0
        stats[f] = {"mean": mu, "std": std, "min": min(vals), "max": max(vals)}
    return stats


def add_scaled_values(rows: list[dict[str, Any]], features: list[str], stats: dict[str, dict[str, float]]) -> None:
    for r in rows:
        scaled = {}
        for f in features:
            scaled[f] = (float(r[f]) - stats[f]["mean"]) / stats[f]["std"]
        r["scaled"] = scaled


def evaluate_rule(row: dict[str, Any], rule: Rule) -> bool:
    val = row[rule.field] if rule.native == "raw" else row["scaled"][rule.field]
    return rule.min <= float(val) <= rule.max


def evaluate_rule_set(row: dict[str, Any], rule_list: list[Rule]) -> bool:
    if not rule_list:
        return False
    return all(evaluate_rule(row, rule) for rule in rule_list)


def segment_stats(rows: list[dict[str, Any]], exp: Experiment, classes: list[str], target_labels: list[str]) -> dict[str, Any]:
    only_a = {c: 0 for c in classes}
    only_b = {c: 0 for c in classes}
    inter = {c: 0 for c in classes}
    ca = cb = ci = 0

    for ix, row in enumerate(rows):
        m_i = evaluate_rule_set(row, exp.rules.get("intersection", []))
        m_a = evaluate_rule_set(row, exp.rules.get("only_cluster_a", []))
        m_b = evaluate_rule_set(row, exp.rules.get("only_cluster_b", []))
        in_a = m_i and m_a
        in_b = m_i and m_b

        cls = target_labels[ix]
        if in_a and in_b:
            ci += 1
            inter[cls] += 1
        elif in_a:
            ca += 1
            only_a[cls] += 1
        elif in_b:
            cb += 1
            only_b[cls] += 1

    return {
        "onlyA": {"count": ca, "dist": only_a},
        "onlyB": {"count": cb, "dist": only_b},
        "intersection": {"count": ci, "dist": inter},
    }


def tvd(stats: dict[str, Any], classes: list[str], min_a: int, min_b: int) -> float:
    count_a = stats["onlyA"]["count"]
    count_b = stats["onlyB"]["count"]
    if count_a < min_a or count_b < min_b:
        return 0.0
    delta = 0.0
    for c in classes:
        pa = stats["onlyA"]["dist"][c] / count_a
        pb = stats["onlyB"]["dist"][c] / count_b
        delta += abs(pa - pb)
    return 0.5 * delta





def _load_base_template() -> str:
    template_path = Path(__file__).with_name("Template_Base.html")
    if template_path.exists():
        return template_path.read_text(encoding="utf-8")
    return ""


def _render_html_from_template(template: str, payload: dict[str, Any], dcfg: DatasetConfig, row_count: int, sampling_info: dict[str, Any]) -> str:
    ranking_rows = "".join(
        f'<tr><td class="p-2">{r["id"]}</td><td class="p-2">{r["name"]}</td><td class="p-2">{r["tvd"]:.3f}</td><td class="p-2">{r["onlyA"]}|{r["intersection"]}|{r["onlyB"]}</td></tr>'
        for r in payload["ranking"]
    )
    return (template
        .replace("{{TITLE}}", "InsideForestVisual - Generado en Python")
        .replace("{{HEADER}}", "InsideForestVisual - HTML Generado")
        .replace("{{DATASET_ID}}", dcfg.dataset_id)
        .replace("{{ROW_COUNT}}", str(row_count))
        .replace("{{FEATURES}}", ", ".join(dcfg.feature_columns))
        .replace("{{SAMPLING_STATUS}}", "activo" if sampling_info["applied"] else "no aplicado")
        .replace("{{SAMPLING_STRATEGY}}", str(sampling_info["strategy"]))
        .replace("{{RANKING_ROWS}}", ranking_rows)
        .replace("{{PAYLOAD_JSON}}", json.dumps(payload, ensure_ascii=False))
    )


def build_html(rows: list[dict[str, Any]], dcfg: DatasetConfig, ecfg: EvaluationConfig, experiments: list[Experiment]) -> str:
    rows = apply_missing_strategy(rows, dcfg.feature_columns, dcfg.missing_strategy)
    rows, sampling_info = sample_rows_if_large(
        rows=rows,
        target_column=dcfg.target_column,
        threshold=dcfg.sampling_threshold,
        sample_size=dcfg.sample_size,
        strategy=dcfg.sampling_strategy,
        seed=dcfg.random_seed,
    )
    stats = compute_stats(rows, dcfg.feature_columns)
    add_scaled_values(rows, dcfg.feature_columns, stats)

    target_labels, effective_mode = build_target_labels(
        rows=rows, target_column=dcfg.target_column, mode=dcfg.target_mode, bins=dcfg.regression_bins
    )
    classes = sorted(set(target_labels))
    ranking = []
    for exp in experiments:
        if not exp.enabled:
            continue
        st = segment_stats(rows, exp, classes, target_labels)
        delta = tvd(st, classes, ecfg.min_samples_only_a, ecfg.min_samples_only_b)
        ranking.append((delta, exp, st))

    reverse = ecfg.ranking_order == "desc"
    ranking.sort(key=lambda x: x[0], reverse=reverse)

    payload = {
        "dataset_config": dcfg.__dict__,
        "rows": rows,
        "sampling_info": sampling_info,
        "target_mode_info": {"requested": dcfg.target_mode, "effective": effective_mode, "bins": dcfg.regression_bins},
        "evaluation_config": ecfg.__dict__,
        "experiments": [
            {
                "id": e.id,
                "name": e.name,
                "description": e.description,
                "target_column": e.target_column,
                "enabled": e.enabled,
                "tags": e.tags,
                "rules": {
                    k: [r.__dict__ for r in v] for k, v in e.rules.items()
                },
            }
            for _, e, _ in ranking
        ],
        "ranking": [
            {
                "id": e.id,
                "name": e.name,
                "tvd": round(delta, 6),
                "onlyA": st["onlyA"]["count"],
                "intersection": st["intersection"]["count"],
                "onlyB": st["onlyB"]["count"],
            }
            for delta, e, st in ranking
        ],
    }

    template = _load_base_template()
    if template:
        return _render_html_from_template(template, payload, dcfg, len(rows), sampling_info)

    # fallback autocontenido si falta el template externo
    return (
        "<!DOCTYPE html><html lang=\"es\"><head><meta charset=\"utf-8\" />"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />"
        "<title>InsideForestVisual - Generado en Python</title><script src=\"https://cdn.tailwindcss.com\"></script>"
        "</head><body class=\"bg-slate-950 text-slate-100\"><main class=\"max-w-6xl mx-auto p-6\">"
        f"<h1 class=\"text-2xl font-bold mb-2\">InsideForestVisual - HTML Generado</h1><p class=\"text-slate-300 mb-4\">Dataset: {dcfg.dataset_id} | Filas usadas: {len(rows)} | Features: {', '.join(dcfg.feature_columns)}</p>"
        "<pre id=\"payload\"></pre></main><script>const payload = "
        + json.dumps(payload, ensure_ascii=False)
        + ";document.getElementById('payload').textContent = JSON.stringify(payload, null, 2);</script></body></html>"
    )


def iris_rows(seed: int = 1234) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    cfg = [
        ("Setosa", 50, (5.01, 0.35), (3.43, 0.38), (1.46, 0.17), (0.25, 0.11)),
        ("Versicolor", 50, (5.94, 0.52), (2.77, 0.31), (4.26, 0.47), (1.33, 0.20)),
        ("Virginica", 50, (6.59, 0.64), (2.97, 0.32), (5.55, 0.55), (2.03, 0.27)),
    ]
    rows = []
    for sp, n, sl, sw, pl, pw in cfg:
        for i in range(n):
            rows.append({
                "id": f"{sp.lower()}-{i}",
                "sepal_length": round(rng.gauss(*sl), 2),
                "sepal_width": round(rng.gauss(*sw), 2),
                "petal_length": round(rng.gauss(*pl), 2),
                "petal_width": round(rng.gauss(*pw), 2),
                "species": sp,
            })
    return rows


def titania_rows(seed: int = 42) -> list[dict[str, Any]]:
    """Dataset sintético inventado: calidad de mineral Titania."""
    rng = random.Random(seed)
    rows = []
    classes = [
        ("Alpha", 120, 61, 0.7, 8.4, 0.5, 3.2, 0.4, 0.18, 0.05),
        ("Beta", 90, 58, 0.9, 7.1, 0.7, 2.1, 0.5, 0.28, 0.06),
        ("Gamma", 70, 64, 0.8, 9.5, 0.8, 4.2, 0.6, 0.12, 0.04),
    ]
    for cls, n, temp_mu, temp_sd, dens_mu, dens_sd, hard_mu, hard_sd, imp_mu, imp_sd in classes:
        for i in range(n):
            rows.append({
                "sample_id": f"{cls.lower()}-{i}",
                "temperature": round(rng.gauss(temp_mu, temp_sd), 2),
                "density": round(rng.gauss(dens_mu, dens_sd), 2),
                "hardness": round(rng.gauss(hard_mu, hard_sd), 2),
                "impurity_ratio": round(max(0.01, rng.gauss(imp_mu, imp_sd)), 3),
                "grade": cls,
            })
    return rows


def invented_iris_experiments() -> list[Experiment]:
    return [
        Experiment(1, "Setosa vs resto", "Separación por pétalo estrecho", "species", {
            "intersection": [Rule("petal_length", 1.0, 5.0, "raw")],
            "only_cluster_a": [Rule("petal_width", 0.1, 0.7, "raw")],
            "only_cluster_b": [Rule("petal_width", 1.0, 2.5, "raw")],
        }, tags=["baseline", "high_tvd"]),
        Experiment(2, "Transición media", "Corredor central", "species", {
            "intersection": [Rule("sepal_length", 4.8, 6.5, "raw")],
            "only_cluster_a": [Rule("sepal_width", -1.0, 0.0, "scaled")],
            "only_cluster_b": [Rule("sepal_width", 0.1, 1.8, "scaled")],
        }, tags=["stress"]),
    ]


def invented_titania_experiments() -> list[Experiment]:
    return [
        Experiment(1, "Pureza baja vs alta", "Frontera por impureza", "grade", {
            "intersection": [Rule("temperature", 57, 65, "raw")],
            "only_cluster_a": [Rule("impurity_ratio", 0.18, 0.40, "raw")],
            "only_cluster_b": [Rule("impurity_ratio", 0.01, 0.16, "raw")],
        }, tags=["quality"]),
        Experiment(2, "Matriz mecánica", "Dureza y densidad", "grade", {
            "intersection": [Rule("density", 6.4, 10.5, "raw")],
            "only_cluster_a": [Rule("hardness", 1.5, 3.0, "raw")],
            "only_cluster_b": [Rule("hardness", 3.1, 5.6, "raw")],
        }, tags=["materials", "robustness"]),
    ]


def california_housing_rows(seed: int = 101) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    for i in range(3500):
        income_k = max(8.0, rng.gauss(70, 20))
        rooms = max(1.0, rng.gauss(5.5, 1.4))
        distance_km = max(0.2, rng.gauss(18, 10))
        crime_index = max(0.1, rng.gauss(35, 12))
        # target continuo
        price_k = 40 + 4.2 * income_k + 18 * rooms - 2.1 * distance_km - 1.3 * crime_index + rng.gauss(0, 20)
        rows.append(
            {
                "house_id": f"h-{i}",
                "income_k": round(income_k, 2),
                "rooms": round(rooms, 2),
                "distance_km": round(distance_km, 2),
                "crime_index": round(crime_index, 2),
                "price_k": round(max(50, price_k), 2),
            }
        )
    return rows


def invented_housing_experiments() -> list[Experiment]:
    return [
        Experiment(1, "Suburbio premium vs riesgo", "Compara zonas con ingresos altos contra zonas con mayor crimen", "price_k", {
            "intersection": [Rule("rooms", 3.0, 8.0, "raw")],
            "only_cluster_a": [Rule("income_k", 85, 180, "raw"), Rule("crime_index", 5, 35, "raw")],
            "only_cluster_b": [Rule("income_k", 8, 80, "raw"), Rule("crime_index", 35, 90, "raw")],
        }, tags=["regression", "deciles"]),
        Experiment(2, "Cercanía urbana", "Efecto de distancia y densidad de habitaciones", "price_k", {
            "intersection": [Rule("income_k", -1.0, 2.5, "scaled")],
            "only_cluster_a": [Rule("distance_km", 0.2, 12.0, "raw")],
            "only_cluster_b": [Rule("distance_km", 20.0, 60.0, "raw")],
        }, tags=["regression"]),
    ]


def run_demo(output_dir: str | Path = ".") -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    iris = iris_rows()
    iris_cfg = DatasetConfig(
        dataset_id="iris_sintetico",
        feature_columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        target_column="species",
        id_column="id",
    )

    titania = titania_rows()
    tit_cfg = DatasetConfig(
        dataset_id="titania_sintetico",
        feature_columns=["temperature", "density", "hardness", "impurity_ratio"],
        target_column="grade",
        id_column="sample_id",
        scaling_mode_default="raw",
    )

    econfg = EvaluationConfig(primary_metric="tvd", min_samples_only_a=5, min_samples_only_b=5)

    iris_html = build_html(iris, iris_cfg, econfg, invented_iris_experiments())
    tit_html = build_html(titania, tit_cfg, econfg, invented_titania_experiments())

    # dataset adicional tipo regresión (target continuo)
    housing = california_housing_rows()
    housing_cfg = DatasetConfig(
        dataset_id="housing_sintetico",
        feature_columns=["income_k", "rooms", "distance_km", "crime_index"],
        target_column="price_k",
        id_column="house_id",
        target_mode="regression_deciles",
        regression_bins=10,
    )
    housing_html = build_html(housing, housing_cfg, econfg, invented_housing_experiments())

    iris_path = out / "iris_generated.html"
    tit_path = out / "titania_generated.html"
    house_path = out / "housing_generated.html"
    iris_path.write_text(iris_html, encoding="utf-8")
    tit_path.write_text(tit_html, encoding="utf-8")
    house_path.write_text(housing_html, encoding="utf-8")

    return {"iris": iris_path, "titania": tit_path, "housing": house_path}


if __name__ == "__main__":
    paths = run_demo("generated")
    print("Generados:")
    for k, p in paths.items():
        print(f" - {k}: {p}")
