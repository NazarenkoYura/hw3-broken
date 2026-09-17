"""Стадия split: разбиение на train/val/test по группам (group_key).

Сплит по строкам здесь неверен: одна и та же тема встречается десятками
вопросов-парафразов, и при случайном разбиении её вопросы разъезжаются по
train и test. Тогда метрика на test завышена — это утечка, и её единственный
симптом — слишком хороший результат. Поэтому режем по группам: все примеры
одной группы целиком уходят в один сплит.
"""

import json
import random
import sys
import time
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from src.config import load_params
from src.contamination import report
from src.schema import Example, dump, iter_examples
from src.textnorm import normalize_group


def group_split(
    examples: list[Example], ratios: dict[str, float], seed: int
) -> dict[str, list[Example]]:
    """Разложить примеры по сплитам целыми группами, сохраняя доли.

    Группы перемешиваются, затем раскладываются от крупных к мелким в сплит с
    наибольшим остатком ёмкости (жадный алгоритм): доли держатся близко к
    заданным, а ни одна группа не пересекает границу сплитов.
    """
    buckets: dict[str, list[Example]] = {name: [] for name in ratios}

    groups: dict[str, list[Example]] = {}
    for ex in examples:
        groups.setdefault(normalize_group(ex.topic), []).append(ex)

    order = list(groups.items())
    random.Random(seed).shuffle(order)
    order.sort(key=lambda item: len(item[1]), reverse=True)  # крупные — первыми

    total = len(examples)
    # Целевой размер сплита; остаток = цель минус уже набранное.
    remaining = {name: total * ratios[name] for name in ratios}
    for _, rows in order:
        target = max(remaining, key=lambda name: remaining[name])
        buckets[target].extend(rows)
        remaining[target] -= len(rows)
    return buckets


def main() -> None:
    params = load_params()
    paths = params["paths"]
    cfg = params["split"]
    started = time.perf_counter()

    examples: list[Example] = list(iter_examples(paths["clean"]))
    if cfg["group_key"] != "topic":
        raise SystemExit(f"неизвестный split.group_key: {cfg['group_key']!r}")

    groups_total = len({normalize_group(ex.topic) for ex in examples})
    buckets = group_split(examples, cfg["ratios"], cfg["seed"])

    for name, rows in buckets.items():
        out = Path(paths[name])
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for ex in rows:
                fh.write(dump(ex) + "\n")

    nd = params["clean"]["near_dup"]
    rep = report(
        buckets["train"],
        buckets["test"],
        shingle_words=nd["shingle_words"],
        num_perm=nd["num_perm"],
        threshold=params["contamination"]["threshold"],
    )

    elapsed = round(time.perf_counter() - started, 2)
    metrics = {
        "version": params["collect"]["version"],
        "seed": cfg["seed"],
        "group_key": cfg["group_key"],
        "groups_total": groups_total,
        "sizes": {name: len(rows) for name, rows in buckets.items()},
        "groups": {
            name: len({normalize_group(ex.topic) for ex in rows}) for name, rows in buckets.items()
        },
        "ratios_actual": {
            name: round(len(rows) / len(examples), 4) for name, rows in buckets.items()
        },
        "contamination": rep,
    }
    mpath = Path(paths["metrics_split"])
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        "split: "
        + ", ".join(f"{name} {len(rows)}" for name, rows in buckets.items())
        + f" (групп {groups_total}, {elapsed} с)"
    )


if __name__ == "__main__":
    main()
