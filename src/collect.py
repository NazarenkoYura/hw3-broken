"""Стадия collect: источник → data/raw.jsonl.

ЗДЕСЬ студент подменяет сбор на свой. Ниже — разбор выгрузки Stack Exchange
(Cross Validated: сайт вопросов и ответов по статистике и машинному обучению).
Формат источника — три CSV-файла: Questions (Id, Title, Body), Answers (Id,
ParentId, Score, Body), Tags (Id, Tag).

Контракт стадии, а не её внутренности, держит остальной пайплайн: на выходе
JSONL со строками {"id", "topic", "messages": [system, user, assistant]}.

Скачанный чужой набор сам по себе сдачей не является (README, «Готовый датасет
как источник»). Поэтому стадия не перекладывает CSV в JSONL один в один, а
делает три вещи, и каждая видна числом в metrics/collect.json:

  1. сужает набор до перечисленных тем (collect.topics) — предметная область;
  2. сверяет ответ с голосами сообщества (collect.min_answer_score): вопрос без
     ответа с достаточным Score выбрасывается, а не переносится в обучение;
  3. разводит единственную связку «вопрос→ответ» на варианты инструкции
     (collect.system_prompts), чтобы модель не заучила одну формулировку.

Память держится под контролем: тяжёлые CSV читаются чанками, тела вопросов и
ответов материализуются только для отобранных строк.
"""

import hashlib
import json
import re
import sys
import time
from collections import Counter
from html import unescape
from pathlib import Path

import pandas as pd

from src.config import load_params, source_files

# Windows-консоль по умолчанию не в UTF-8: печать «→» или «≥» роняет стадию
# ещё до записи артефактов. Принудительный UTF-8 делает вывод одинаковым везде.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

CHUNK = 50_000

_SCRIPT = re.compile(r"(?is)<(script|style).*?</\1>")
_BREAK = re.compile(r"(?i)<br\s*/?>")
_BLOCK = re.compile(r"(?i)</(p|div|li|tr|h[1-6]|blockquote|pre)>")
_TAG = re.compile(r"<[^>]+>")
_SPACES = re.compile(r"[ \t\f\v]+")
_BLANKS = re.compile(r"\n\s*\n+")


def strip_html(text: str) -> str:
    """Тело SE-сообщения — HTML. Убираем разметку, оставляем читаемый текст."""
    if not text:
        return ""
    text = _SCRIPT.sub(" ", text)
    text = _BREAK.sub("\n", text)
    text = _BLOCK.sub("\n", text)
    text = _TAG.sub(" ", text)
    text = unescape(text)
    text = _SPACES.sub(" ", text)
    text = _BLANKS.sub("\n", text)
    return text.strip()


def pick_prompt(example_id: str, variants: list[str]) -> str:
    """Детерминированно выбрать вариант инструкции по id примера.

    Именно sha1, а не встроенный hash(): тот солится на каждый запуск процесса,
    и raw.jsonl переставал бы быть воспроизводимым.
    """
    digest = hashlib.sha1(example_id.encode("utf-8")).hexdigest()
    return variants[int(digest, 16) % len(variants)]


def _read_chunks(path: str, usecols: list[str]) -> pd.io.parsers.TextFileReader:
    """Чанковое чтение CSV Stack Exchange.

    Кириллица в выгрузке иногда сопровождается битыми байтами, поэтому
    errors="replace": строка не теряется, а нечитаемый байт заменяется.
    """
    return pd.read_csv(
        path,
        usecols=usecols,
        dtype={"Id": "int64", "ParentId": "int64", "Score": "float64"},
        encoding="utf-8",
        encoding_errors="replace",
        chunksize=CHUNK,
        on_bad_lines="skip",
        low_memory=False,
    )


def first_tag_by_question(path: str) -> dict[int, str]:
    """Id вопроса → его первичный тег (первый по порядку в источнике)."""
    first: dict[int, str] = {}
    for chunk in _read_chunks(path, ["Id", "Tag"]):
        for qid, tag in zip(chunk["Id"].tolist(), chunk["Tag"].tolist()):
            if qid not in first and isinstance(tag, str) and tag.strip():
                first[qid] = tag.strip()
    return first


def main() -> None:
    params = load_params()
    cfg = params["collect"]
    paths = params["paths"]
    sources = source_files(params)
    version = cfg["version"]

    n_rows = cfg["n_rows"][version]
    max_per_topic = cfg["max_per_topic"][version]
    variants = cfg["system_prompts"]
    if not variants:
        raise SystemExit("collect.system_prompts пуст: инструкцию брать неоткуда")
    topics = cfg["topics"]
    wanted = set(topics) if topics else None
    min_score = cfg["min_answer_score"]

    out = Path(paths["raw"])
    out.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()

    # 1. Теги: какому вопросу принадлежит какая тема.
    tag_of = first_tag_by_question(sources["tags"])

    # 2. Вопросы: тема + потолок на тему. Тело здесь не читаем — только Id и
    #    Title, иначе пришлось бы держать в памяти 100+ МБ.
    scanned = dropped_topic = dropped_cap = 0
    taken: Counter[str] = Counter()
    candidates: list[tuple[int, str, str]] = []
    for chunk in _read_chunks(sources["questions"], ["Id", "Title"]):
        for qid, title in zip(chunk["Id"].tolist(), chunk["Title"].tolist()):
            scanned += 1
            topic = tag_of.get(qid)
            if topic is None or (wanted is not None and topic not in wanted):
                dropped_topic += 1
                continue
            if taken[topic] >= max_per_topic:
                dropped_cap += 1
                continue
            taken[topic] += 1
            candidates.append((qid, topic, title if isinstance(title, str) else ""))
    candidate_ids = {qid for qid, _, _ in candidates}

    # 3. Тела вопросов — только для отобранных id.
    body_of: dict[int, str] = {}
    for chunk in _read_chunks(sources["questions"], ["Id", "Body"]):
        sub = chunk[chunk["Id"].isin(candidate_ids)]
        for qid, body in zip(sub["Id"].tolist(), sub["Body"].tolist()):
            body_of[qid] = body if isinstance(body, str) else ""

    # 4. Ответы — только для отобранных вопросов, и только прошедшие порог Score.
    #    На вопрос берём ответ с наибольшим Score (при равенстве — с меньшим Id).
    best: dict[int, tuple[float, int, str]] = {}
    for chunk in _read_chunks(sources["answers"], ["Id", "ParentId", "Score", "Body"]):
        sub = chunk[chunk["ParentId"].isin(candidate_ids)]
        for aid, pid, score, body in zip(
            sub["Id"].tolist(),
            sub["ParentId"].tolist(),
            sub["Score"].tolist(),
            sub["Body"].tolist(),
        ):
            if score is None or score < min_score:
                continue
            if not isinstance(body, str) or not body.strip():
                continue
            current = best.get(pid)
            if current is None or score > current[0] or (score == current[0] and aid < current[1]):
                best[pid] = (float(score), int(aid), body)

    # 5. Сборка JSONL в порядке Id. Ответа с достаточным Score нет — строка
    #    отбрасывается и попадает в счётчик.
    written = dropped_no_answer = 0
    prompts_used: set[str] = set()
    with out.open("w", encoding="utf-8") as fh:
        for qid, topic, title in sorted(candidates, key=lambda row: row[0]):
            if written >= n_rows:
                break
            if qid not in best:
                dropped_no_answer += 1
                continue
            _, _, answer_body = best[qid]
            user = (title.strip() + "\n\n" + strip_html(body_of.get(qid, ""))).strip()
            prompt = pick_prompt(str(qid), variants)
            prompts_used.add(prompt)
            record = {
                "id": f"cv_{qid}",
                "topic": topic,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": strip_html(answer_body)},
                ],
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    elapsed = round(time.perf_counter() - started, 2)
    metrics = {
        "version": version,
        "files": len(sources),
        "rows_scanned": scanned,
        "candidates": len(candidates),
        "rows_written": written,
        "dropped_topic_filter": dropped_topic,
        "dropped_topic_cap": dropped_cap,
        "dropped_no_answer": dropped_no_answer,
        "topics_filter": len(wanted) if wanted else 0,
        "max_per_topic": max_per_topic,
        "min_answer_score": min_score,
        "system_prompt_variants": len(prompts_used),
    }
    mpath = Path(paths["metrics_collect"])
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"collect: версия {version}, файлов {metrics['files']}, "
        f"просмотрено {scanned}, записано {written} "
        f"(тема -{dropped_topic}, потолок темы -{dropped_cap}, "
        f"без ответа -{dropped_no_answer}), "
        f"вариантов инструкции {len(prompts_used)}, "
        f"{elapsed} с → {out}"
    )


if __name__ == "__main__":
    main()
