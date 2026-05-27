# InsideForestVisual

`html_builder.py` genera reportes HTML con ranking de experimentos por segmentos (`onlyA` vs `onlyB`).

## Uso de `html_builder`

### Demo completa

```bash
python html_builder.py --mode demo
```

Genera:
- `generated/iris_generated.html`
- `generated/titania_generated.html`
- `generated/titanic_generated.html`
- `generated/housing_generated.html`

### Examen Titanic (binario: 2 salidas)

```bash
python html_builder.py --mode titanic_exam --output-dir generated --seed 77 --min-only-a 5 --min-only-b 5 --positive-label 1 --negative-label 0
```

Genera:
- `generated/titanic_exam_generated.html`

Este modo está pensado para targets con **2 salidas** (ej: `0/1`).
Si quieres exigir que ambos segmentos tengan presencia de ambas clases, usa:

```bash
python html_builder.py --mode titanic_exam --require-all-classes-in-segment
```

## Ejemplo Iris

```python
from html_builder import DatasetConfig, EvaluationConfig, build_html, iris_rows, invented_iris_experiments

rows = iris_rows(seed=1234)
cfg = DatasetConfig(
    dataset_id="iris_sintetico",
    feature_columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    target_column="species",
    id_column="id",
)
ecfg = EvaluationConfig(primary_metric="tvd", min_samples_only_a=5, min_samples_only_b=5)
html = build_html(rows, cfg, ecfg, invented_iris_experiments())
open("iris_ejemplo.html", "w", encoding="utf-8").write(html)
```

## Parámetros incrementados

Se ampliaron parámetros de entrada para el examen Titanic y evaluación binaria:

- `--mode` (`demo|titanic_exam`)
- `--output-dir`
- `--seed`
- `--min-only-a`
- `--min-only-b`
- `--positive-label`
- `--negative-label`
- `--require-all-classes-in-segment`

Además, `EvaluationConfig` ahora acepta:
- `binary_positive_label`
- `binary_negative_label`
- `require_all_classes_in_segment`

## Tests

```bash
PYTHONPATH=. pytest -q
```
