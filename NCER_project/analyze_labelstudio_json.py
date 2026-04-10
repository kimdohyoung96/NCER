#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze exported Label Studio JSON for QA / data production metrics.

Supported metrics (out of the box)
- 태깅 완료율
- 검수 완료율
- 검수 통과율 (strict = Good, lenient = Good+Fair)
- 재작업률 (strict = Bad, lenient = Fair+Bad)
- 평균/중앙 태깅 소요 시간
- fallback 비율
- intent / triage / answer_quality 분포
- 문서 활용도(used docs / unused_should_use / unused_not_needed)
- 일자별 처리량
- 주석자별 처리량 (completed_by 기준)

Usage
-----
python analyze_labelstudio_json.py --input labelstudio_export.json --outdir ./analysis_out

Outputs
-------
analysis_summary.json
task_level_metrics.csv
daily_throughput.csv
annotator_throughput.csv
label_distributions.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple


def safe_parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    value = value.strip()
    try:
        # Handle trailing Z
        if value.endswith("Z"):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        # Handle naive values like "2026-02-10 14:07:17 KST"
        if value.endswith(" KST"):
            core = value[:-4]
            dt = datetime.strptime(core, "%Y-%m-%d %H:%M:%S")
            return dt.replace(tzinfo=timezone.utc).astimezone(timezone.utc)
        return datetime.fromisoformat(value)
    except Exception:
        return None


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


@dataclass
class TaskMetrics:
    task_id: Any
    inner_id: Any
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


def parse_task(item: Dict[str, Any]) -> TaskMetrics:
    data = item.get("data", {}) or {}
    anns = item.get("annotations", []) or []

    # This export appears to have one final annotation per task, but keep it robust.
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

    return TaskMetrics(
        task_id=item.get("id"),
        inner_id=item.get("inner_id"),
        wid=data.get("wid"),
        query=data.get("query"),
        created_at=data.get("created_at") or item.get("created_at"),
        annotation_created_at=ann.get("created_at"),
        annotation_updated_at=ann.get("updated_at"),
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
    )


def to_serializable(value: Any) -> Any:
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


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


def build_summary(tasks: List[TaskMetrics]) -> Dict[str, Any]:
    total = len(tasks)
    annotated = [t for t in tasks if t.is_annotated]
    reviewed = [t for t in tasks if t.has_quality_review]
    triaged = [t for t in tasks if t.has_triage]

    lead_times = [t.lead_time_sec for t in tasks if t.lead_time_sec is not None]
    fallback_count = sum(t.has_fallback for t in tasks)

    intent_counts = Counter(t.intent for t in tasks if t.intent)
    triage_counts = Counter(t.triage for t in tasks if t.triage)
    quality_counts = Counter(t.answer_quality for t in tasks if t.answer_quality)

    good = quality_counts.get("Good", 0)
    fair = quality_counts.get("Fair", 0)
    bad = quality_counts.get("Bad", 0)
    reviewed_n = len(reviewed)

    # Rework/pass formulas are configurable by consumers; expose both.
    strict_pass_rate = (good / reviewed_n) if reviewed_n else None
    lenient_pass_rate = ((good + fair) / reviewed_n) if reviewed_n else None
    strict_rework_rate = (bad / reviewed_n) if reviewed_n else None
    lenient_rework_rate = ((fair + bad) / reviewed_n) if reviewed_n else None

    unused_should_use_tasks = sum(1 for t in tasks if t.unused_should_use_count > 0)
    no_doc_used_tasks = sum(1 for t in tasks if t.used_docs_count == 0)
    answer_comment_tasks = sum(1 for t in tasks if t.has_answer_comment)
    doc_comment_tasks = sum(1 for t in tasks if t.has_doc_comment)

    # Daily throughput by annotation creation date
    daily_counter = Counter()
    for t in tasks:
        dt = safe_parse_dt(t.annotation_created_at)
        if dt:
            daily_counter[dt.date().isoformat()] += 1

    # Annotator throughput
    annotator_counter = Counter(t.completed_by for t in tasks if t.completed_by is not None)

    return {
        "dataset_overview": {
            "total_tasks": total,
            "annotated_tasks": len(annotated),
            "annotation_completion_rate": (len(annotated) / total) if total else None,
            "quality_review_coverage": (reviewed_n / total) if total else None,
            "triage_coverage": (len(triaged) / total) if total else None,
            "fallback_rate": (fallback_count / total) if total else None,
        },
        "quality_metrics": {
            "reviewed_tasks": reviewed_n,
            "strict_pass_rate_good_only": strict_pass_rate,
            "lenient_pass_rate_good_or_fair": lenient_pass_rate,
            "strict_rework_rate_bad_only": strict_rework_rate,
            "lenient_rework_rate_fair_or_bad": lenient_rework_rate,
            "quality_distribution": dict(quality_counts),
        },
        "time_metrics": {
            "average_tagging_lead_time_sec": mean(lead_times) if lead_times else None,
            "median_tagging_lead_time_sec": median(lead_times) if lead_times else None,
            "p90_tagging_lead_time_sec": percentile(lead_times, 0.90),
            "p95_tagging_lead_time_sec": percentile(lead_times, 0.95),
            "min_tagging_lead_time_sec": min(lead_times) if lead_times else None,
            "max_tagging_lead_time_sec": max(lead_times) if lead_times else None,
        },
        "label_distributions": {
            "intent_distribution": dict(intent_counts),
            "triage_distribution": dict(triage_counts),
        },
        "document_usage_metrics": {
            "avg_used_docs_per_task": (sum(t.used_docs_count for t in tasks) / total) if total else None,
            "avg_unused_should_use_per_task": (sum(t.unused_should_use_count for t in tasks) / total) if total else None,
            "tasks_with_missing_should_use_docs": unused_should_use_tasks,
            "missing_should_use_docs_rate": (unused_should_use_tasks / total) if total else None,
            "tasks_with_zero_used_docs": no_doc_used_tasks,
            "zero_used_docs_rate": (no_doc_used_tasks / total) if total else None,
            "tasks_with_doc_comment": doc_comment_tasks,
            "tasks_with_answer_comment": answer_comment_tasks,
        },
        "throughput": {
            "daily_task_counts": dict(sorted(daily_counter.items())),
            "annotator_task_counts": dict(annotator_counter),
        },
        "recommended_additional_metrics": {
            "intent_share_of_total": "의도별 데이터 비중. 편향된 분포 여부 확인",
            "fallback_rate": "fallback 응답 비율. 검색/생성 실패 구간 탐지",
            "used_docs_zero_rate": "답변 생성 시 문서 미사용 비율. RAG 품질 저하 신호",
            "missing_should_use_docs_rate": "써야 할 문서를 놓친 비율. retrieval 리콜 이슈 탐지",
            "quality_by_intent": "의도별 Good/Fair/Bad 분포. 취약 카테고리 확인",
            "annotator_variance": "주석자별 처리량·품질 편차. 교육/가이드 정합성 확인",
            "p90_p95_lead_time": "평균보다 긴 꼬리 작업 탐지. 일정 산정 정확도 향상",
            "fallback_by_intent": "어떤 의도에서 fallback이 집중되는지 확인",
            "review_gap_rate": "의도는 태깅됐지만 triage/quality가 비어 있는 비율",
        },
    }


def build_task_rows(tasks: List[TaskMetrics]) -> List[Dict[str, Any]]:
    rows = []
    for t in tasks:
        rows.append({
            "task_id": t.task_id,
            "inner_id": t.inner_id,
            "wid": t.wid,
            "intent": t.intent,
            "triage": t.triage,
            "answer_quality": t.answer_quality,
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
        })
    return rows


def build_daily_rows(tasks: List[TaskMetrics]) -> List[Dict[str, Any]]:
    daily = defaultdict(list)
    for t in tasks:
        dt = safe_parse_dt(t.annotation_created_at)
        if dt:
            daily[dt.date().isoformat()].append(t)

    rows = []
    for day, xs in sorted(daily.items()):
        lead_times = [t.lead_time_sec for t in xs if t.lead_time_sec is not None]
        quality_counts = Counter(t.answer_quality for t in xs if t.answer_quality)
        rows.append({
            "date": day,
            "tasks": len(xs),
            "avg_lead_time_sec": mean(lead_times) if lead_times else None,
            "median_lead_time_sec": median(lead_times) if lead_times else None,
            "good": quality_counts.get("Good", 0),
            "fair": quality_counts.get("Fair", 0),
            "bad": quality_counts.get("Bad", 0),
            "fallback_tasks": sum(t.has_fallback for t in xs),
        })
    return rows


def build_annotator_rows(tasks: List[TaskMetrics]) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)
    for t in tasks:
        grouped[str(t.completed_by)].append(t)

    rows = []
    for annotator, xs in sorted(grouped.items(), key=lambda kv: kv[0]):
        lead_times = [t.lead_time_sec for t in xs if t.lead_time_sec is not None]
        quality_counts = Counter(t.answer_quality for t in xs if t.answer_quality)
        rows.append({
            "completed_by": annotator,
            "tasks": len(xs),
            "avg_lead_time_sec": mean(lead_times) if lead_times else None,
            "median_lead_time_sec": median(lead_times) if lead_times else None,
            "good": quality_counts.get("Good", 0),
            "fair": quality_counts.get("Fair", 0),
            "bad": quality_counts.get("Bad", 0),
            "fallback_tasks": sum(t.has_fallback for t in xs),
            "used_docs_avg": (sum(t.used_docs_count for t in xs) / len(xs)) if xs else None,
        })
    return rows


def build_label_distribution_rows(tasks: List[TaskMetrics]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for group_name, counter in [
        ("intent", Counter(t.intent for t in tasks if t.intent)),
        ("triage", Counter(t.triage for t in tasks if t.triage)),
        ("answer_quality", Counter(t.answer_quality for t in tasks if t.answer_quality)),
        ("wid", Counter(t.wid for t in tasks if t.wid)),
    ]:
        total = sum(counter.values())
        for label, count in counter.most_common():
            rows.append({
                "group": group_name,
                "label": label,
                "count": count,
                "rate": (count / total) if total else None,
            })
    return rows

# ---
# 범위 확정 
# ---

def is_valid_task(task_id: int) -> bool:
    return (
        (164209 <= task_id <= 165076) or
        (165201 <= task_id <= 165332)
    )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to exported Label Studio JSON")
    parser.add_argument("--outdir", required=True, help="Directory to write outputs")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("Expected top-level JSON array")

    # =========================
    # 🔥 필터 적용
    # =========================
    filtered_raw = [
        item for item in raw
        if item.get("id") and is_valid_task(item.get("id"))
    ]

    print(f"[INFO] 전체 태스크: {len(raw)}")
    print(f"[INFO] 필터 적용 후 태스크: {len(filtered_raw)}")

    # =========================
    # 🔥 필터된 데이터만 분석
    # =========================
    tasks = [parse_task(item) for item in filtered_raw]

    summary = build_summary(tasks)
    task_rows = build_task_rows(tasks)
    daily_rows = build_daily_rows(tasks)
    annotator_rows = build_annotator_rows(tasks)
    distribution_rows = build_label_distribution_rows(tasks)

    with (outdir / "analysis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=to_serializable)

    write_csv(outdir / "task_level_metrics.csv", task_rows)
    write_csv(outdir / "daily_throughput.csv", daily_rows)
    write_csv(outdir / "annotator_throughput.csv", annotator_rows)
    write_csv(outdir / "label_distributions.csv", distribution_rows)

    print(f"[DONE] Wrote analysis files to: {outdir}")

if __name__ == "__main__":
    main()
