#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt

RANGE_1_START = 164209
RANGE_1_END = 165076
RANGE_2_START = 165201
RANGE_2_END = 165332
EXPECTED_TASK_COUNT = 1000

RAG_INTENT_LABEL = "RAG or 초동수사"

# 예외 규칙:
# 164336은 wid가 비어 있어도 bns2로 정의
SPECIAL_GAME_BY_TASK_ID = {
    164336: "bns2",
}

def resolve_game(task_id: int, data: Dict[str, Any]) -> str:
    wid = data.get("wid")
    if wid is not None and str(wid).strip():
        return str(wid).strip()
    return SPECIAL_GAME_BY_TASK_ID.get(task_id, "UNKNOWN")


def is_valid_task(task_id: Any) -> bool:
    if task_id is None:
        return False
    try:
        tid = int(task_id)
    except Exception:
        return False
    return ((RANGE_1_START <= tid <= RANGE_1_END) or (RANGE_2_START <= tid <= RANGE_2_END))


def which_range(task_id: Any) -> Optional[str]:
    if task_id is None:
        return None
    try:
        tid = int(task_id)
    except Exception:
        return None
    if RANGE_1_START <= tid <= RANGE_1_END:
        return f"{RANGE_1_START}-{RANGE_1_END}"
    if RANGE_2_START <= tid <= RANGE_2_END:
        return f"{RANGE_2_START}-{RANGE_2_END}"
    return None


def safe_parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    try:
        if text.endswith("Z"):
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        if text.endswith(" KST"):
            core = text[:-4]
            dt = datetime.strptime(core, "%Y-%m-%d %H:%M:%S")
            return dt.replace(tzinfo=timezone(timedelta(hours=9)))
        return datetime.fromisoformat(text)
    except Exception:
        return None


def truncate_to_second(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return dt.replace(microsecond=0)


def same_to_second(a: Optional[str], b: Optional[str]) -> bool:
    da = truncate_to_second(safe_parse_dt(a))
    db = truncate_to_second(safe_parse_dt(b))
    if da is None or db is None:
        return False
    return da == db


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return d0 + d1


def flatten_choices(result: Dict[str, Any]) -> List[str]:
    return list(result.get("value", {}).get("choices", []) or [])


def flatten_ranker_ids(result: Dict[str, Any], key: str) -> List[str]:
    return list(result.get("value", {}).get("ranker", {}).get(key, []) or [])


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_dump(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_bar_chart(path: Path, labels: List[str], values: List[float], title: str, ylabel: str) -> None:
    plt.figure(figsize=(11, 6))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_line_chart(path: Path, labels: List[str], values: List[float], title: str, ylabel: str) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(labels, values, marker="o")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_grouped_bar_chart(
    path: Path,
    categories: List[str],
    series_names: List[str],
    series_values: List[List[float]],
    title: str,
    ylabel: str,
) -> None:
    plt.figure(figsize=(12, 6))
    n_cat = len(categories)
    n_series = len(series_names)
    x = list(range(n_cat))
    width = 0.8 / max(n_series, 1)

    for i, vals in enumerate(series_values):
        offsets = [v + (i - (n_series - 1) / 2) * width for v in x]
        plt.bar(offsets, vals, width=width, label=series_names[i])

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(x, categories, rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


@dataclass
class TaskMetrics:
    task_id: int
    task_range: str
    game: str
    inner_id: Optional[Any]
    wid: Optional[str]
    query: Optional[str]
    created_at: Optional[str]
    annotation_created_at: Optional[str]
    annotation_updated_at: Optional[str]
    completed_by: Optional[Any]
    lead_time_sec: Optional[float]
    is_annotated: bool
    is_cancelled: bool
    has_quality_review: bool
    has_triage: bool
    intent: Optional[str]
    triage: Optional[str]
    answer_quality: Optional[str]
    fallback_value: Optional[Any]
    has_fallback: bool
    used_docs_count: int
    used_noise_count: int
    unused_should_use_count: int
    unused_not_needed_count: int
    has_doc_comment: bool
    has_answer_comment: bool
    result_count: int
    is_reworked: bool
    rework_reason: Optional[str]
    is_reworked_by_ground_truth_time: bool


def parse_task(item: Dict[str, Any]) -> TaskMetrics:
    task_id = int(item["id"])
    data = item.get("data", {}) or {}
    anns = item.get("annotations", []) or []
    ann = anns[0] if anns else {}
    results = ann.get("result", []) or []

    intent = None
    triage = None
    answer_quality = None
    used_docs_count = 0
    used_noise_count = 0
    unused_should_use_count = 0
    unused_not_needed_count = 0
    has_doc_comment = False
    has_answer_comment = False

    for r in results:
        from_name = r.get("from_name")
        rtype = r.get("type")
        if from_name == "intent" and rtype == "choices":
            vals = flatten_choices(r)
            intent = vals[0] if vals else intent
        elif from_name == "triage" and rtype == "choices":
            vals = flatten_choices(r)
            triage = vals[0] if vals else triage
        elif from_name == "answer_quality" and rtype == "choices":
            vals = flatten_choices(r)
            answer_quality = vals[0] if vals else answer_quality
        elif from_name == "used_ranker" and rtype == "ranker":
            used_docs_count += len(flatten_ranker_ids(r, "used_todo"))
            used_noise_count += len(flatten_ranker_ids(r, "used_noise"))
        elif from_name == "unused_ranker" and rtype == "ranker":
            unused_should_use_count += len(flatten_ranker_ids(r, "unused_should_use"))
            unused_not_needed_count += len(flatten_ranker_ids(r, "unused_not_needed"))
        elif from_name == "doc_comment" and rtype == "textarea":
            has_doc_comment = True
        elif from_name == "answer_comment" and rtype == "textarea":
            has_answer_comment = True

    fallback_value = data.get("fallback")
    has_fallback = fallback_value is not None

    annotation_created_at = ann.get("created_at")
    annotation_updated_at = ann.get("updated_at")

    is_reworked_by_ground_truth_time = bool(
        anns and annotation_created_at and annotation_updated_at
        and not same_to_second(annotation_created_at, annotation_updated_at)
    )
    is_reworked = is_reworked_by_ground_truth_time
    rework_reason = "updated_after_initial_submit" if is_reworked else None

    game = resolve_game(task_id, data)

    return TaskMetrics(
        task_id=task_id,
        task_range=which_range(task_id) or "UNKNOWN",
        game=game,
        inner_id=item.get("inner_id"),
        wid=data.get("wid"),
        query=data.get("query"),
        created_at=data.get("created_at") or item.get("created_at"),
        annotation_created_at=annotation_created_at,
        annotation_updated_at=annotation_updated_at,
        completed_by=ann.get("completed_by"),
        lead_time_sec=ann.get("lead_time"),
        is_annotated=bool(anns),
        is_cancelled=bool(ann.get("was_cancelled", False)),
        has_quality_review=answer_quality is not None,
        has_triage=triage is not None,
        intent=intent,
        triage=triage,
        answer_quality=answer_quality,
        fallback_value=fallback_value,
        has_fallback=has_fallback,
        used_docs_count=used_docs_count,
        used_noise_count=used_noise_count,
        unused_should_use_count=unused_should_use_count,
        unused_not_needed_count=unused_not_needed_count,
        has_doc_comment=has_doc_comment,
        has_answer_comment=has_answer_comment,
        result_count=ann.get("result_count") or len(results),
        is_reworked=is_reworked,
        rework_reason=rework_reason,
        is_reworked_by_ground_truth_time=is_reworked_by_ground_truth_time,
    )


def build_task_rows(tasks: List[TaskMetrics]) -> List[Dict[str, Any]]:
    return [{
        "task_id": t.task_id,
        "task_range": t.task_range,
        "game": t.game,
        "inner_id": t.inner_id,
        "wid": t.wid,
        "intent": t.intent,
        "triage": t.triage,
        "answer_quality": t.answer_quality,
        "is_reworked": t.is_reworked,
        "is_reworked_by_ground_truth_time": t.is_reworked_by_ground_truth_time,
        "rework_reason": t.rework_reason,
        "lead_time_sec": t.lead_time_sec,
        "lead_time_min": round(t.lead_time_sec / 60, 2) if t.lead_time_sec is not None else None,
        "has_fallback": t.has_fallback,
        "fallback_value": t.fallback_value,
        "used_docs_count": t.used_docs_count,
        "used_noise_count": t.used_noise_count,
        "unused_should_use_count": t.unused_should_use_count,
        "unused_not_needed_count": t.unused_not_needed_count,
        "has_doc_comment": t.has_doc_comment,
        "has_answer_comment": t.has_answer_comment,
        "completed_by": t.completed_by,
        "annotation_created_at": t.annotation_created_at,
        "annotation_updated_at": t.annotation_updated_at,
        "query": t.query,
    } for t in tasks]


def build_daily_rows(tasks: List[TaskMetrics]) -> List[Dict[str, Any]]:
    daily = defaultdict(list)
    for t in tasks:
        dt = safe_parse_dt(t.annotation_updated_at) or safe_parse_dt(t.annotation_created_at)
        if dt:
            daily[dt.date().isoformat()].append(t)

    rows = []
    for day, xs in sorted(daily.items()):
        lead_times = [t.lead_time_sec for t in xs if t.lead_time_sec is not None]
        q = Counter(t.answer_quality for t in xs if t.answer_quality)
        rows.append({
            "date": day,
            "tasks": len(xs),
            "reworked_tasks": sum(1 for t in xs if t.is_reworked),
            "avg_lead_time_sec": mean(lead_times) if lead_times else None,
            "median_lead_time_sec": median(lead_times) if lead_times else None,
            "good": q.get("Good", 0),
            "fair": q.get("Fair", 0),
            "bad": q.get("Bad", 0),
            "fallback_tasks": sum(1 for t in xs if t.has_fallback),
        })
    return rows


def build_annotator_rows(tasks: List[TaskMetrics]) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)
    for t in tasks:
        grouped[str(t.completed_by)].append(t)
    rows = []
    for annotator, xs in sorted(grouped.items()):
        lead_times = [t.lead_time_sec for t in xs if t.lead_time_sec is not None]
        q = Counter(t.answer_quality for t in xs if t.answer_quality)
        rows.append({
            "completed_by": annotator,
            "tasks": len(xs),
            "reworked_tasks": sum(1 for t in xs if t.is_reworked),
            "avg_lead_time_sec": mean(lead_times) if lead_times else None,
            "median_lead_time_sec": median(lead_times) if lead_times else None,
            "good": q.get("Good", 0),
            "fair": q.get("Fair", 0),
            "bad": q.get("Bad", 0),
            "fallback_tasks": sum(1 for t in xs if t.has_fallback),
            "used_docs_avg": (sum(t.used_docs_count for t in xs) / len(xs)) if xs else None,
        })
    return rows


def build_label_distribution_rows(tasks: List[TaskMetrics]) -> List[Dict[str, Any]]:
    rows = []
    for group_name, counter in [
        ("intent", Counter(t.intent for t in tasks if t.intent)),
        ("triage", Counter(t.triage for t in tasks if t.triage)),
        ("answer_quality", Counter(t.answer_quality for t in tasks if t.answer_quality)),
        ("wid", Counter(t.wid for t in tasks if t.wid)),
        ("game", Counter(t.game for t in tasks if t.game)),
        ("task_range", Counter(t.task_range for t in tasks if t.task_range)),
        ("is_reworked", Counter(str(t.is_reworked) for t in tasks)),
        ("rework_reason", Counter(t.rework_reason for t in tasks if t.rework_reason)),
    ]:
        total = sum(counter.values())
        for label, count in counter.most_common():
            rows.append({"group": group_name, "label": label, "count": count, "rate": (count / total) if total else None})
    return rows


def build_range_split_rows(tasks: List[TaskMetrics]) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)
    for t in tasks:
        grouped[t.task_range].append(t)
    rows = []
    for range_name, xs in sorted(grouped.items()):
        lead_times = [t.lead_time_sec for t in xs if t.lead_time_sec is not None]
        q = Counter(t.answer_quality for t in xs if t.answer_quality)
        reviewed = sum(1 for t in xs if t.has_quality_review)
        reworked = sum(1 for t in xs if t.is_reworked)
        rows.append({
            "task_range": range_name,
            "tasks": len(xs),
            "reworked_tasks": reworked,
            "rework_rate": (reworked / len(xs)) if xs else None,
            "avg_lead_time_sec": mean(lead_times) if lead_times else None,
            "median_lead_time_sec": median(lead_times) if lead_times else None,
            "reviewed_tasks": reviewed,
            "good": q.get("Good", 0),
            "fair": q.get("Fair", 0),
            "bad": q.get("Bad", 0),
            "strict_pass_rate": (q.get("Good", 0) / reviewed) if reviewed else None,
            "lenient_pass_rate": ((q.get("Good", 0) + q.get("Fair", 0)) / reviewed) if reviewed else None,
            "fallback_rate": (sum(1 for t in xs if t.has_fallback) / len(xs)) if xs else None,
        })
    return rows


def build_quality_by_intent_rows(tasks: List[TaskMetrics]) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)
    for t in tasks:
        grouped[t.intent or "UNKNOWN"].append(t)
    rows = []
    for intent, xs in sorted(grouped.items()):
        reviewed = [t for t in xs if t.answer_quality]
        q = Counter(t.answer_quality for t in reviewed if t.answer_quality)
        reviewed_n = len(reviewed)
        reworked = sum(1 for t in xs if t.is_reworked)
        rows.append({
            "intent": intent,
            "tasks": len(xs),
            "reworked_tasks": reworked,
            "rework_rate": (reworked / len(xs)) if xs else None,
            "reviewed_tasks": reviewed_n,
            "good": q.get("Good", 0),
            "fair": q.get("Fair", 0),
            "bad": q.get("Bad", 0),
            "strict_pass_rate": (q.get("Good", 0) / reviewed_n) if reviewed_n else None,
            "lenient_pass_rate": ((q.get("Good", 0) + q.get("Fair", 0)) / reviewed_n) if reviewed_n else None,
            "fallback_rate": (sum(1 for t in xs if t.has_fallback) / len(xs)) if xs else None,
        })
    return rows


def build_game_summary_rows(tasks: List[TaskMetrics]) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)
    for t in tasks:
        grouped[t.game].append(t)

    rows = []
    for game, xs in sorted(grouped.items()):
        lead_times = [t.lead_time_sec for t in xs if t.lead_time_sec is not None]
        quality = Counter(t.answer_quality for t in xs if t.answer_quality)
        rag_tasks = [t for t in xs if t.intent == RAG_INTENT_LABEL]
        rag_triage = Counter(t.triage for t in rag_tasks if t.triage in ("yes", "no"))

        rows.append({
            "game": game,
            "tasks": len(xs),
            "avg_lead_time_sec": mean(lead_times) if lead_times else None,
            "median_lead_time_sec": median(lead_times) if lead_times else None,
            "good": quality.get("Good", 0),
            "fair": quality.get("Fair", 0),
            "bad": quality.get("Bad", 0),
            "rag_tasks": len(rag_tasks),
            "triage_yes_within_rag": rag_triage.get("yes", 0),
            "triage_no_within_rag": rag_triage.get("no", 0),
            "reworked_tasks": sum(1 for t in xs if t.is_reworked),
            "fallback_tasks": sum(1 for t in xs if t.has_fallback),
        })
    return rows


def build_game_quality_distribution_rows(tasks: List[TaskMetrics]) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)
    for t in tasks:
        grouped[t.game].append(t)

    rows = []
    for game, xs in sorted(grouped.items()):
        total = len(xs)
        q = Counter(t.answer_quality for t in xs if t.answer_quality)
        for label in ["Good", "Fair", "Bad"]:
            count = q.get(label, 0)
            rows.append({
                "game": game,
                "quality": label,
                "count": count,
                "rate_over_all_tasks": (count / total) if total else None,
            })
    return rows


def build_game_rag_triage_distribution_rows(tasks: List[TaskMetrics]) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)
    for t in tasks:
        grouped[t.game].append(t)

    rows = []
    for game, xs in sorted(grouped.items()):
        rag_tasks = [t for t in xs if t.intent == RAG_INTENT_LABEL]
        total_rag = len(rag_tasks)
        c = Counter(t.triage for t in rag_tasks if t.triage in ("yes", "no"))

        for label in ["yes", "no"]:
            count = c.get(label, 0)
            rows.append({
                "game": game,
                "triage": label,
                "count": count,
                "rate_within_rag_tasks": (count / total_rag) if total_rag else None,
                "rag_tasks": total_rag,
            })
    return rows


def build_summary(tasks: List[TaskMetrics]) -> Dict[str, Any]:
    total = len(tasks)
    annotated = [t for t in tasks if t.is_annotated]
    reviewed = [t for t in tasks if t.has_quality_review]
    triaged = [t for t in tasks if t.has_triage]
    lead_times = [t.lead_time_sec for t in tasks if t.lead_time_sec is not None]
    fallback_count = sum(1 for t in tasks if t.has_fallback)

    intent_counts = Counter(t.intent for t in tasks if t.intent)
    triage_counts = Counter(t.triage for t in tasks if t.triage)
    quality_counts = Counter(t.answer_quality for t in tasks if t.answer_quality)

    reviewed_n = len(reviewed)
    good = quality_counts.get("Good", 0)
    fair = quality_counts.get("Fair", 0)
    bad = quality_counts.get("Bad", 0)

    no_doc_used_tasks = sum(1 for t in tasks if t.used_docs_count == 0)
    unused_should_use_tasks = sum(1 for t in tasks if t.unused_should_use_count > 0)
    reworked_tasks = [t for t in tasks if t.is_reworked]
    rework_reason_counts = Counter(t.rework_reason for t in reworked_tasks if t.rework_reason)

    daily_counter = Counter()
    daily_groups = defaultdict(list)
    for t in tasks:
        dt = safe_parse_dt(t.annotation_updated_at) or safe_parse_dt(t.annotation_created_at)
        if dt:
            day = dt.date().isoformat()
            daily_counter[day] += 1
            if t.lead_time_sec is not None:
                daily_groups[day].append(t.lead_time_sec)

    daily_values = list(daily_counter.values())
    daily_avg_tagging_time = {day: mean(xs) if xs else None for day, xs in sorted(daily_groups.items())}
    annotator_counter = Counter(str(t.completed_by) for t in tasks if t.completed_by is not None)

    game_counts = Counter(t.game for t in tasks)
    game_avg_lead = {}
    game_quality_distribution = {}
    game_rag_triage_distribution = {}
    for game in sorted(game_counts.keys()):
        game_tasks = [t for t in tasks if t.game == game]
        xs = [t.lead_time_sec for t in game_tasks if t.lead_time_sec is not None]
        game_avg_lead[game] = mean(xs) if xs else None
        q = Counter(t.answer_quality for t in game_tasks if t.answer_quality)
        game_quality_distribution[game] = dict(q)

        rag_tasks = [t for t in game_tasks if t.intent == RAG_INTENT_LABEL]
        c = Counter(t.triage for t in rag_tasks if t.triage in ("yes", "no"))
        game_rag_triage_distribution[game] = {
            "yes": c.get("yes", 0),
            "no": c.get("no", 0),
            "rag_tasks": len(rag_tasks),
        }

    rag_tasks_total = [t for t in tasks if t.intent == RAG_INTENT_LABEL]
    rag_yes_no_counts = Counter(t.triage for t in rag_tasks_total if t.triage in ("yes", "no"))

    # 월별 반복 운영용 지표 구분
    keep_metrics = {
        "annotation_completion_rate": (len(annotated) / total) if total else None,
        "quality_review_coverage": (reviewed_n / total) if total else None,
        "triage_coverage": (len(triaged) / total) if total else None,
        "ground_truth_time_based_rework_rate": (len(reworked_tasks) / total) if total else None,
        "average_tagging_lead_time_sec": mean(lead_times) if lead_times else None,
        "median_tagging_lead_time_sec": median(lead_times) if lead_times else None,
        "average_daily_output": mean(daily_values) if daily_values else None,
        "median_daily_output": median(daily_values) if daily_values else None,
        "strict_pass_rate_good_only": (good / reviewed_n) if reviewed_n else None,
        "lenient_pass_rate_good_or_fair": ((good + fair) / reviewed_n) if reviewed_n else None,
        "fallback_rate": (fallback_count / total) if total else None,
        "missing_should_use_docs_rate": (unused_should_use_tasks / total) if total else None,
    }

    reference_metrics = {
        "expected_task_count": EXPECTED_TASK_COUNT,
        "filtered_task_count": total,
        "p90_tagging_lead_time_sec": percentile(lead_times, 0.90),
        "p95_tagging_lead_time_sec": percentile(lead_times, 0.95),
        "min_tagging_lead_time_sec": min(lead_times) if lead_times else None,
        "max_tagging_lead_time_sec": max(lead_times) if lead_times else None,
        "tasks_with_zero_used_docs": no_doc_used_tasks,
        "tasks_with_doc_comment": sum(1 for t in tasks if t.has_doc_comment),
        "tasks_with_answer_comment": sum(1 for t in tasks if t.has_answer_comment),
    }

    visualization_recommendations = {
        "required_charts": [
            "intent_distribution",
            "daily_task_counts",
            "daily_average_tagging_time",
            "game_task_counts",
            "game_average_tagging_time",
            "game_quality_distribution",
            "game_rag_triage_distribution",
            "core_kpi_snapshot",
        ],
        "why": {
            "intent_distribution": "월별 데이터 성격 변화 파악",
            "daily_task_counts": "작업량 변동과 병목일 확인",
            "daily_average_tagging_time": "난이도 상승일/비효율 발생일 확인",
            "game_task_counts": "게임별 물량 편차 확인",
            "game_average_tagging_time": "게임별 난이도 비교",
            "game_quality_distribution": "게임별 품질 편차 확인",
            "game_rag_triage_distribution": "게임별 초동수사 의존도 확인",
            "core_kpi_snapshot": "월별 핵심 KPI 요약",
        },
    }

    return {
        "scope": {
            "fixed_task_ranges": [f"{RANGE_1_START}-{RANGE_1_END}", f"{RANGE_2_START}-{RANGE_2_END}"],
            "expected_task_count": EXPECTED_TASK_COUNT,
            "filtered_task_count": total,
            "range_distribution": dict(Counter(t.task_range for t in tasks)),
        },
        "dataset_overview": {
            "total_tasks": total,
            "annotated_tasks": len(annotated),
            "annotation_completion_rate": (len(annotated) / total) if total else None,
            "quality_review_coverage": (reviewed_n / total) if total else None,
            "triage_coverage": (len(triaged) / total) if total else None,
            "fallback_rate": (fallback_count / total) if total else None,
            "cancelled_rate": (sum(1 for t in tasks if t.is_cancelled) / total) if total else None,
        },
        "quality_metrics": {
            "reviewed_tasks": reviewed_n,
            "strict_pass_rate_good_only": (good / reviewed_n) if reviewed_n else None,
            "lenient_pass_rate_good_or_fair": ((good + fair) / reviewed_n) if reviewed_n else None,
            "strict_rework_rate_bad_only": (bad / reviewed_n) if reviewed_n else None,
            "lenient_rework_rate_fair_or_bad": ((fair + bad) / reviewed_n) if reviewed_n else None,
            "quality_distribution": dict(quality_counts),
        },
        "rework_metrics": {
            "rework_definition": "Initial submit already contains triage, answer_quality, doc_comment, answer_comment. Rework is counted only when the annotation was later updated. Comparison is at second precision: created_at(second) != updated_at(second).",
            "reworked_tasks": len(reworked_tasks),
            "rework_rate_all_tasks": (len(reworked_tasks) / total) if total else None,
            "rework_reason_distribution": dict(rework_reason_counts),
            "ground_truth_time_based_reworked_tasks": len(reworked_tasks),
            "ground_truth_time_based_rework_rate": (len(reworked_tasks) / total) if total else None,
        },
        "time_metrics": {
            "average_tagging_lead_time_sec": mean(lead_times) if lead_times else None,
            "median_tagging_lead_time_sec": median(lead_times) if lead_times else None,
            "p90_tagging_lead_time_sec": percentile(lead_times, 0.90),
            "p95_tagging_lead_time_sec": percentile(lead_times, 0.95),
            "min_tagging_lead_time_sec": min(lead_times) if lead_times else None,
            "max_tagging_lead_time_sec": max(lead_times) if lead_times else None,
        },
        "speed_metrics": {
            "daily_task_counts": dict(sorted(daily_counter.items())),
            "daily_average_tagging_lead_time_sec": dict(sorted(daily_avg_tagging_time.items())),
            "average_daily_output": mean(daily_values) if daily_values else None,
            "median_daily_output": median(daily_values) if daily_values else None,
            "annotator_task_counts": dict(annotator_counter),
        },
        "game_metrics": {
            "game_task_counts": dict(sorted(game_counts.items())),
            "game_average_tagging_lead_time_sec": game_avg_lead,
            "game_quality_distribution": game_quality_distribution,
            "game_rag_triage_distribution": game_rag_triage_distribution,
        },
        "consistency_checks": {
            "rag_tasks_total": len(rag_tasks_total),
            "rag_triage_yes_total": rag_yes_no_counts.get("yes", 0),
            "rag_triage_no_total": rag_yes_no_counts.get("no", 0),
            "rag_triage_yes_no_sum": rag_yes_no_counts.get("yes", 0) + rag_yes_no_counts.get("no", 0),
        },
        "label_distributions": {
            "intent_distribution": dict(intent_counts),
            "triage_distribution": dict(triage_counts),
        },
        "retrieval_doc_usage_metrics": {
            "avg_used_docs_per_task": (sum(t.used_docs_count for t in tasks) / total) if total else None,
            "avg_unused_should_use_per_task": (sum(t.unused_should_use_count for t in tasks) / total) if total else None,
            "tasks_with_zero_used_docs": no_doc_used_tasks,
            "zero_used_docs_rate": (no_doc_used_tasks / total) if total else None,
            "tasks_with_missing_should_use_docs": unused_should_use_tasks,
            "missing_should_use_docs_rate": (unused_should_use_tasks / total) if total else None,
            "tasks_with_doc_comment": sum(1 for t in tasks if t.has_doc_comment),
            "tasks_with_answer_comment": sum(1 for t in tasks if t.has_answer_comment),
        },
        "monthly_operation_guidance": {
            "keep_as_core_kpis": keep_metrics,
            "use_as_reference_metrics": reference_metrics,
            "visualization_recommendations": visualization_recommendations,
        },
    }


def save_all_charts(outdir: Path, summary: Dict[str, Any]) -> None:
    charts_dir = outdir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    intent_dist = summary["label_distributions"]["intent_distribution"]
    save_bar_chart(
        charts_dir / "intent_distribution.png",
        list(intent_dist.keys()),
        list(intent_dist.values()),
        "Intent Distribution",
        "Tasks",
    )

    daily_counts = summary["speed_metrics"]["daily_task_counts"]
    save_line_chart(
        charts_dir / "daily_task_counts.png",
        list(daily_counts.keys()),
        list(daily_counts.values()),
        "Daily Throughput",
        "Tasks",
    )

    daily_avg = summary["speed_metrics"]["daily_average_tagging_lead_time_sec"]
    save_line_chart(
        charts_dir / "daily_average_tagging_time.png",
        list(daily_avg.keys()),
        list(daily_avg.values()),
        "Daily Average Tagging Lead Time",
        "Lead Time (sec)",
    )

    game_counts = summary["game_metrics"]["game_task_counts"]
    save_bar_chart(
        charts_dir / "game_task_counts.png",
        list(game_counts.keys()),
        list(game_counts.values()),
        "Task Count by Game",
        "Tasks",
    )

    game_avg = summary["game_metrics"]["game_average_tagging_lead_time_sec"]
    save_bar_chart(
        charts_dir / "game_average_tagging_time.png",
        list(game_avg.keys()),
        [0 if v is None else v for v in game_avg.values()],
        "Average Tagging Lead Time by Game",
        "Lead Time (sec)",
    )

    gq = summary["game_metrics"]["game_quality_distribution"]
    categories = list(gq.keys())
    good_vals = [gq[g].get("Good", 0) for g in categories]
    fair_vals = [gq[g].get("Fair", 0) for g in categories]
    bad_vals = [gq[g].get("Bad", 0) for g in categories]
    save_grouped_bar_chart(
        charts_dir / "game_quality_distribution.png",
        categories,
        ["Good", "Fair", "Bad"],
        [good_vals, fair_vals, bad_vals],
        "Quality Distribution by Game",
        "Tasks",
    )

    gr = summary["game_metrics"]["game_rag_triage_distribution"]
    categories = list(gr.keys())
    yes_vals = [gr[g].get("yes", 0) for g in categories]
    no_vals = [gr[g].get("no", 0) for g in categories]
    save_grouped_bar_chart(
        charts_dir / "game_rag_triage_distribution.png",
        categories,
        ["yes", "no"],
        [yes_vals, no_vals],
        "RAG Task Triage Distribution by Game",
        "Tasks",
    )

    core = summary["monthly_operation_guidance"]["keep_as_core_kpis"]
    core_labels = [
        "completion_rate",
        "review_coverage",
        "triage_coverage",
        "rework_rate",
        "strict_pass_rate",
        "lenient_pass_rate",
        "fallback_rate",
        "missing_should_use_docs_rate",
    ]
    core_values = [
        core["annotation_completion_rate"] or 0,
        core["quality_review_coverage"] or 0,
        core["triage_coverage"] or 0,
        core["ground_truth_time_based_rework_rate"] or 0,
        core["strict_pass_rate_good_only"] or 0,
        core["lenient_pass_rate_good_or_fair"] or 0,
        core["fallback_rate"] or 0,
        core["missing_should_use_docs_rate"] or 0,
    ]
    save_bar_chart(
        charts_dir / "core_kpi_snapshot.png",
        core_labels,
        core_values,
        "Core KPI Snapshot",
        "Rate",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--skip-count-check", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("Expected top-level JSON array")

    filtered_raw = [item for item in raw if is_valid_task(item.get("id"))]
    print(f"[INFO] 전체 태스크 수: {len(raw)}")
    print(f"[INFO] 필터 적용 태스크 수: {len(filtered_raw)}")

    if (not args.skip_count_check) and len(filtered_raw) != EXPECTED_TASK_COUNT:
        raise ValueError(
            f"Filtered task count is {len(filtered_raw)}, expected {EXPECTED_TASK_COUNT}. "
            "Use --skip-count-check to continue."
        )

    tasks = [parse_task(item) for item in filtered_raw]
    summary = build_summary(tasks)

    json_dump(outdir / "analysis_summary.json", summary)
    write_csv(outdir / "task_level_metrics.csv", build_task_rows(tasks))
    write_csv(outdir / "daily_throughput.csv", build_daily_rows(tasks))
    write_csv(outdir / "annotator_throughput.csv", build_annotator_rows(tasks))
    write_csv(outdir / "label_distributions.csv", build_label_distribution_rows(tasks))
    write_csv(outdir / "range_split_summary.csv", build_range_split_rows(tasks))
    write_csv(outdir / "quality_by_intent.csv", build_quality_by_intent_rows(tasks))
    write_csv(outdir / "game_summary.csv", build_game_summary_rows(tasks))
    write_csv(outdir / "game_quality_distribution.csv", build_game_quality_distribution_rows(tasks))
    write_csv(outdir / "game_rag_triage_distribution.csv", build_game_rag_triage_distribution_rows(tasks))
    save_all_charts(outdir, summary)

    print(f"[DONE] 분석 결과 저장 완료: {outdir}")
    print(f"[DONE] 차트 저장 완료: {outdir / 'charts'}")


if __name__ == "__main__":
    main()
