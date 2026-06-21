#!/usr/bin/env python3
"""Training health monitor – polls TensorBoard event files and runs 5 health checks."""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sys
import time
import traceback
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

SCALAR_TAGS = [
    "mean_reward",
    "success_rate/train",
    "success_rate/eval",
    "approx_kl",
    "clip_fraction",
    "nonzero_adv_frac",
    "format_reward",
    "thinking_chars",
    "mean_response_len",
    "policy_entropy",
    "std_reward",
    "grad_norm",
]

TEXT_TAGS = ["text_0", "text_1", "text_2", "text_3"]

# PyTorch SummaryWriter stores text as ``text_N/text_summary``; scalars may differ.
SCALAR_ALIASES: dict[str, str] = {
    "policy_entropy": "entropy",
}
TEXT_ALIASES: dict[str, str] = {
    tag: f"{tag}/text_summary" for tag in TEXT_TAGS
}


def _load_yaml_editor():
    try:
        from ruamel.yaml import YAML

        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.indent(mapping=2, sequence=4, offset=2)
        return yaml, "ruamel"
    except ImportError:
        import yaml as pyyaml

        return pyyaml, "pyyaml"


def linregress_slope(points: list[tuple[float, float]]) -> float:
    """Simple least-squares slope (no scipy dependency)."""
    if len(points) < 2:
        return 0.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    n = len(xs)
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(n))
    den = sum((xs[i] - x_mean) ** 2 for i in range(n))
    return num / den if den else 0.0


def ngram_repetition_ratio(text: str, n: int = 4) -> float:
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(v for v in counts.values() if v > 2)
    return repeated / len(ngrams)


def strip_html_pre(raw: str) -> str:
    text = html.unescape(raw)
    text = re.sub(r"^<pre>", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"</pre>$", "", text.strip(), flags=re.IGNORECASE)
    return text.strip()


def extract_between(text: str, start_tag: str, end_tag: str) -> str | None:
    start = text.find(start_tag)
    if start < 0:
        return None
    start += len(start_tag)
    end = text.find(end_tag, start)
    if end < 0:
        return None
    return text[start:end].strip()


def is_strictly_increasing(values: list[float]) -> bool:
    return len(values) >= 2 and all(values[i] > values[i - 1] for i in range(1, len(values)))


@dataclass
class ScalarPoint:
    step: int
    value: float


@dataclass
class TextPoint:
    step: int
    text: str


@dataclass
class MonitorState:
    last_checked_step: int = 0
    reward_gap_history: deque = field(default_factory=lambda: deque(maxlen=5))
    kl_clip_violation_count: int = 0
    lr_reduction_count: int = 0
    adv_frac_low_count: int = 0
    eval_critical_count: int = 0
    temperature_boost_count: int = 0
    config_change_log: list = field(default_factory=list)
    missing_tags_logged: set = field(default_factory=set)


class TBReader:
    """Load scalar / text events from TensorBoard event files."""

    def __init__(self, log_dir: str | Path):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        self.log_dir = str(log_dir)
        self._ea = EventAccumulator(
            self.log_dir,
            size_guidance={
                "scalars": 0,
                "tensors": 0,
                "histograms": 0,
                "compressedHistograms": 0,
                "images": 0,
                "audio": 0,
            },
        )
        self._ea.Reload()
        self._available_scalar_tags: set[str] = set()
        self._available_tensor_tags: set[str] = set()
        self._refresh_tag_sets()

    def _refresh_tag_sets(self) -> None:
        tags = self._ea.Tags()
        self._available_scalar_tags = set(tags.get("scalars", []))
        self._available_tensor_tags = set(tags.get("tensors", []))

    def Reload(self) -> None:
        self._ea.Reload()
        self._refresh_tag_sets()

    def _resolve_scalar_tag(self, tag: str) -> str | None:
        if tag in self._available_scalar_tags:
            return tag
        alias = SCALAR_ALIASES.get(tag)
        if alias and alias in self._available_scalar_tags:
            return alias
        return None

    def _resolve_text_tag(self, tag: str) -> str | None:
        if tag in self._available_tensor_tags:
            return tag
        alias = TEXT_ALIASES.get(tag)
        if alias and alias in self._available_tensor_tags:
            return alias
        return None

    def get_scalars(self, tag: str) -> list[ScalarPoint]:
        resolved = self._resolve_scalar_tag(tag)
        if resolved is None:
            return []
        return [
            ScalarPoint(step=int(e.step), value=float(e.value))
            for e in self._ea.Scalars(resolved)
        ]

    def get_texts(self, tag: str) -> list[TextPoint]:
        resolved = self._resolve_text_tag(tag)
        if resolved is None:
            return []
        points: list[TextPoint] = []
        for event in self._ea.Tensors(resolved):
            tp = event.tensor_proto
            if not tp.string_val:
                continue
            raw = tp.string_val[0]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            points.append(TextPoint(step=int(event.step), text=strip_html_pre(raw)))
        return points

    def latest_step(self) -> int:
        max_step = 0
        for tag in SCALAR_TAGS:
            series = self.get_scalars(tag)
            if series:
                max_step = max(max_step, series[-1].step)
        for tag in TEXT_TAGS:
            series = self.get_texts(tag)
            if series:
                max_step = max(max_step, series[-1].step)
        return max_step

    def log_missing_tags(self, state: MonitorState) -> list[dict[str, Any]]:
        return _log_missing_tags(state, self._resolve_scalar_tag, self._resolve_text_tag)


def _log_missing_tags(
    state: MonitorState,
    resolve_scalar,
    resolve_text,
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for tag in SCALAR_TAGS:
        if tag in state.missing_tags_logged:
            continue
        if resolve_scalar(tag) is None:
            state.missing_tags_logged.add(tag)
            warnings.append(
                {
                    "level": "tag_missing",
                    "tag": tag,
                    "tag_type": "scalar",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
            )
    for tag in TEXT_TAGS:
        if tag in state.missing_tags_logged:
            continue
        if resolve_text(tag) is None:
            state.missing_tags_logged.add(tag)
            warnings.append(
                {
                    "level": "tag_missing",
                    "tag": tag,
                    "tag_type": "text",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
            )
    return warnings


class MockReader:
    """JSONL-backed reader for ``--test_mode``."""

    def __init__(self, mock_path: str | Path):
        self.scalars: dict[str, list[ScalarPoint]] = {t: [] for t in SCALAR_TAGS}
        self.texts: dict[str, list[TextPoint]] = {t: [] for t in TEXT_TAGS}
        with open(mock_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row["type"] == "scalar":
                    self.scalars[row["tag"]].append(
                        ScalarPoint(step=int(row["step"]), value=float(row["value"]))
                    )
                elif row["type"] == "text":
                    self.texts[row["tag"]].append(
                        TextPoint(step=int(row["step"]), text=row["value"])
                    )
        for series in self.scalars.values():
            series.sort(key=lambda p: p.step)
        for series in self.texts.values():
            series.sort(key=lambda p: p.step)

    def Reload(self) -> None:
        return

    def get_scalars(self, tag: str) -> list[ScalarPoint]:
        return list(self.scalars.get(tag, []))

    def get_texts(self, tag: str) -> list[TextPoint]:
        return list(self.texts.get(tag, []))

    def latest_step(self) -> int:
        max_step = 0
        for series in self.scalars.values():
            if series:
                max_step = max(max_step, series[-1].step)
        for series in self.texts.values():
            if series:
                max_step = max(max_step, series[-1].step)
        return max_step

    def _resolve_scalar_tag(self, tag: str) -> str | None:
        return tag if self.scalars.get(tag) else None

    def _resolve_text_tag(self, tag: str) -> str | None:
        return tag if self.texts.get(tag) else None

    def log_missing_tags(self, state: MonitorState) -> list[dict[str, Any]]:
        return _log_missing_tags(
            state, self._resolve_scalar_tag, self._resolve_text_tag
        )


def value_at_or_before(series: list[ScalarPoint], step: int) -> float | None:
    candidates = [p for p in series if p.step <= step]
    if not candidates:
        return None
    return candidates[-1].value


def recent_points(series: list[ScalarPoint], n: int, up_to_step: int) -> list[tuple[int, float]]:
    filtered = [p for p in series if p.step <= up_to_step]
    tail = filtered[-n:]
    return [(p.step, p.value) for p in tail]


def latest_text_at_or_before(series: list[TextPoint], step: int) -> str | None:
    candidates = [p for p in series if p.step <= step]
    if not candidates:
        return None
    return candidates[-1].text


class ConfigEditor:
    def __init__(self, config_path: str | Path, dry_run: bool = False):
        self.config_path = Path(config_path)
        self.dry_run = dry_run
        self.yaml, self.backend = _load_yaml_editor()

    def read(self) -> dict[str, Any]:
        with open(self.config_path, "r", encoding="utf-8") as f:
            if self.backend == "ruamel":
                return self.yaml.load(f)
            return self.yaml.safe_load(f)

    def write(self, data: dict[str, Any], step: int, comment: str) -> None:
        if self.dry_run:
            return
        backup = self.config_path.with_name(
            f"{self.config_path.name}.bak.{step}"
        )
        shutil.copy2(self.config_path, backup)
        with open(self.config_path, "w", encoding="utf-8") as f:
            if self.backend == "ruamel":
                self.yaml.dump(data, f)
            else:
                self.yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            f.write(f"\n# [monitor] {comment}\n")

    def update_w_length(
        self, step: int, state: MonitorState, reason: str
    ) -> str | None:
        data = self.read()
        old = float(data["training"]["r1_reward"]["w_length"])
        new = max(old * 0.5, 0.01)
        if new == old:
            return None
        action = f"w_length: {old:g} -> {new:g}"
        if not self.dry_run:
            data["training"]["r1_reward"]["w_length"] = new
            self.write(data, step, f"step={step} | {action} | reason: {reason}")
        state.config_change_log.append((step, "w_length", old, new))
        return action

    def update_learning_rate(
        self, step: int, state: MonitorState, reason: str
    ) -> str | None:
        data = self.read()
        old = float(data["training"]["learning_rate"])
        new = max(old * 0.5, 5e-6)
        if new == old:
            return None
        action = f"learning_rate: {old:g} -> {new:g}"
        if not self.dry_run:
            data["training"]["learning_rate"] = new
            self.write(data, step, f"step={step} | {action} | reason: {reason}")
        state.config_change_log.append((step, "learning_rate", old, new))
        return action

    def update_temperature(
        self, step: int, state: MonitorState, reason: str
    ) -> str | None:
        data = self.read()
        old = float(data["training"]["sampling"]["temperature"])
        new = min(old + 0.05, 1.50)
        if new == old:
            return None
        action = f"temperature: {old:g} -> {new:g}"
        if not self.dry_run:
            data["training"]["sampling"]["temperature"] = new
            self.write(data, step, f"step={step} | {action} | reason: {reason}")
        state.config_change_log.append((step, "temperature", old, new))
        return action


def append_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def check_reward_gap(
    reader: TBReader | MockReader,
    editor: ConfigEditor,
    state: MonitorState,
    step: int,
) -> dict[str, Any]:
    mean_r = value_at_or_before(reader.get_scalars("mean_reward"), step)
    acc = value_at_or_before(reader.get_scalars("success_rate/train"), step)
    result: dict[str, Any] = {
        "check": "reward_gap",
        "step": step,
        "level": "INFO",
        "triggered": False,
    }
    if mean_r is None or acc is None:
        result["skipped"] = True
        result["reason"] = "missing mean_reward or success_rate/train"
        return result

    gap = mean_r - acc
    state.reward_gap_history.append((step, gap))
    gaps = [g for _, g in state.reward_gap_history]
    result.update({"gap": round(gap, 4), "gap_trend": [round(g, 4) for g in gaps[-3:]]})

    if gap > 0.35 and len(gaps) >= 3 and is_strictly_increasing(gaps[-3:]):
        result["triggered"] = True
        result["level"] = "WARNING"
        action = editor.update_w_length(
            step, state, reason=f"reward_gap>{gap:.2f} for 3 steps"
        )
        result["action"] = action or "no_change"
    return result


def check_text_quality(
    reader: TBReader | MockReader,
    state: MonitorState,
    step: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for tag in ("text_0", "text_1"):
        series = reader.get_texts(tag)
        text = latest_text_at_or_before(series, step)
        if text is None:
            continue

        issues: list[str] = []
        format_closed = "</think>" in text
        if not format_closed:
            issues.append("FORMAT_NOT_CLOSED")

        thinking = extract_between(text, "<think>", "</think>")
        if thinking is None:
            thinking = text
        rep_ratio = ngram_repetition_ratio(thinking, n=4)
        if rep_ratio > 0.30:
            issues.append("THINKING_REPETITIVE")

        answer = extract_between(text, "<answer>", "</answer>")
        answer_tokens = len(answer.split()) if answer is not None else 0
        if answer is not None:
            if answer.strip() == "":
                issues.append("ANSWER_EMPTY")
            elif answer_tokens > 80:
                issues.append("ANSWER_TOO_LONG")
        elif "<answer>" in text or "</answer>" in text:
            issues.append("ANSWER_EMPTY")

        level = "WARNING" if issues else "INFO"
        results.append(
            {
                "check": "text_quality",
                "step": step,
                "level": level,
                "source_tag": tag,
                "format_closed": format_closed,
                "repetition_ratio": round(rep_ratio, 4),
                "answer_token_count": answer_tokens,
                "issues": issues,
            }
        )
    return results


def check_eval_trend(
    reader: TBReader | MockReader,
    state: MonitorState,
    step: int,
) -> dict[str, Any]:
    series = reader.get_scalars("success_rate/eval")
    points = recent_points(series, 3, step)
    result: dict[str, Any] = {
        "check": "eval_trend",
        "step": step,
        "level": "INFO",
    }
    if len(points) < 3:
        result["level"] = "eval_data_insufficient"
        result["eval_points"] = points
        return result

    slope = linregress_slope([(float(s), v) for s, v in points])
    result["eval_points"] = points
    result["slope"] = round(slope, 6)

    if slope >= -0.005:
        result["level"] = "INFO"
        state.eval_critical_count = 0
    elif slope > -0.02:
        result["level"] = "WARNING"
        state.eval_critical_count = 0
    else:
        result["level"] = "CRITICAL"
        state.eval_critical_count += 1
        result["consecutive_critical"] = state.eval_critical_count
        if state.eval_critical_count >= 2:
            result["recommendation"] = "STOP_TRAINING"
    return result


def check_training_instability(
    reader: TBReader | MockReader,
    editor: ConfigEditor,
    state: MonitorState,
    step: int,
) -> dict[str, Any]:
    kl_series = reader.get_scalars("approx_kl")
    clip_series = reader.get_scalars("clip_fraction")
    kl_pts = recent_points(kl_series, 3, step)
    clip_pts = recent_points(clip_series, 3, step)

    result: dict[str, Any] = {
        "check": "training_instability",
        "step": step,
        "level": "INFO",
        "triggered": False,
        "lr_change_count": state.lr_reduction_count,
    }
    if len(kl_pts) < 3 or len(clip_pts) < 3:
        result["skipped"] = True
        result["reason"] = "insufficient kl/clip history"
        return result

    kl_map = dict(kl_pts)
    clip_map = dict(clip_pts)
    common_steps = sorted(set(kl_map) & set(clip_map))[-3:]
    if len(common_steps) < 3:
        result["skipped"] = True
        result["reason"] = "insufficient aligned kl/clip steps"
        return result

    kl_values = [kl_map[s] for s in common_steps]
    clip_values = [clip_map[s] for s in common_steps]
    result["kl_values"] = [round(v, 4) for v in kl_values]
    result["clip_values"] = [round(v, 4) for v in clip_values]

    all_violating = all(k > 0.15 and c > 0.35 for k, c in zip(kl_values, clip_values))
    if all_violating:
        result["triggered"] = True
        result["level"] = "WARNING"
        if state.lr_reduction_count < 3:
            action = editor.update_learning_rate(
                step, state, reason="kl>0.15 and clip>0.35 for 3 steps"
            )
            if action:
                state.lr_reduction_count += 1
                result["action"] = action
            else:
                result["action"] = "no_change"
        else:
            result["action"] = "lr_reduction_limit_reached"
            result["level"] = "CRITICAL"
        state.kl_clip_violation_count = 0
        result["lr_change_count"] = state.lr_reduction_count
    return result


def check_gradient_efficiency(
    reader: TBReader | MockReader,
    editor: ConfigEditor,
    state: MonitorState,
    step: int,
) -> dict[str, Any]:
    series = reader.get_scalars("nonzero_adv_frac")
    window = recent_points(series, 20, step)
    result: dict[str, Any] = {
        "check": "gradient_efficiency",
        "step": step,
        "level": "INFO",
        "triggered": False,
        "window_size": len(window),
    }
    if len(window) < 5:
        result["skipped"] = True
        result["reason"] = "insufficient nonzero_adv_frac history (need >=5)"
        return result

    mean_adv = sum(v for _, v in window) / len(window)
    result["mean_adv_frac"] = round(mean_adv, 4)

    if mean_adv < 0.30:
        state.adv_frac_low_count += 1
    else:
        state.adv_frac_low_count = 0

    result["consecutive_low"] = state.adv_frac_low_count
    if state.adv_frac_low_count >= 2:
        result["triggered"] = True
        result["level"] = "WARNING"
        if state.temperature_boost_count < 2:
            action = editor.update_temperature(
                step, state, reason=f"mean_adv_frac={mean_adv:.2f} for 2 checks"
            )
            if action:
                state.temperature_boost_count += 1
                result["action"] = action
            else:
                result["action"] = "no_change"
        else:
            result["action"] = "temperature_boost_limit_reached"
    return result


def run_all_checks(
    reader: TBReader | MockReader,
    editor: ConfigEditor,
    state: MonitorState,
    step: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    records.extend(reader.log_missing_tags(state))

    check_fns = [
        lambda: check_reward_gap(reader, editor, state, step),
        lambda: check_text_quality(reader, state, step),
        lambda: check_eval_trend(reader, state, step),
        lambda: check_training_instability(reader, editor, state, step),
        lambda: check_gradient_efficiency(reader, editor, state, step),
    ]
    for fn in check_fns:
        try:
            out = fn()
            if isinstance(out, list):
                records.extend(out)
            else:
                records.append(out)
        except Exception as exc:
            records.append(
                {
                    "level": "check_error",
                    "check": getattr(fn, "__name__", "unknown"),
                    "step": step,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
            )
    for rec in records:
        rec.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
    return records


def _status_symbol(ok: bool, warn: bool = False, critical: bool = False) -> str:
    try:
        from rich.console import Console

        console = Console()
        if critical:
            return "[red]✗[/red]"
        if warn:
            return "[yellow]⚠[/yellow]"
        return "[green]✓[/green]"
    except ImportError:
        if critical:
            return "X"
        if warn:
            return "!"
        return "+"


def print_summary(step: int, records: list[dict[str, Any]]) -> None:
    by_check = {r.get("check"): r for r in records if "check" in r}
    text_recs = [r for r in records if r.get("check") == "text_quality"]

    parts: list[str] = []

    rg = by_check.get("reward_gap", {})
    if rg.get("triggered"):
        parts.append(f"! reward_gap={rg.get('gap', '?')} -> {rg.get('action', '')}")
    elif rg.get("skipped"):
        parts.append("? reward_gap=skip")
    else:
        parts.append(f"+ reward_gap={rg.get('gap', '?')}")

    if text_recs:
        bad = any(r.get("issues") for r in text_recs)
        if bad:
            issues = sorted({i for r in text_recs for i in r.get("issues", [])})
            parts.append(f"! text_{','.join(issues)}")
        else:
            parts.append("+ text_ok")
    else:
        parts.append("? text_skip")

    ev = by_check.get("eval_trend", {})
    if ev.get("level") == "eval_data_insufficient":
        parts.append("? eval_skip")
    elif ev.get("level") == "CRITICAL":
        rec = ev.get("recommendation", "")
        parts.append(f"X eval_slope={ev.get('slope', '?')}{' ' + rec if rec else ''}")
    elif ev.get("level") == "WARNING":
        parts.append(f"! eval_slope={ev.get('slope', '?')}")
    else:
        slope = ev.get("slope")
        parts.append(f"+ eval_slope={slope:+}" if isinstance(slope, (int, float)) else "+ eval_slope=?")

    kl = by_check.get("training_instability", {})
    if kl.get("triggered"):
        parts.append(f"! kl={kl.get('kl_values', ['?'])[-1]} -> {kl.get('action', '')}")
    elif kl.get("skipped"):
        parts.append("? kl=skip")
    else:
        klv = kl.get("kl_values", [None])
        parts.append(f"+ kl={klv[-1] if klv and klv[-1] is not None else 'ok'}")

    adv = by_check.get("gradient_efficiency", {})
    if adv.get("triggered"):
        parts.append(
            f"! adv_frac={adv.get('mean_adv_frac', '?')} -> {adv.get('action', '')}"
        )
    elif adv.get("skipped"):
        parts.append("? adv_frac=skip")
    else:
        parts.append(f"+ adv_frac={adv.get('mean_adv_frac', '?')}")

    line = f"[Step {step}] " + " | ".join(parts)

    try:
        from rich.console import Console

        Console().print(line)
    except ImportError:
        print(line)


def run_monitor(args: argparse.Namespace) -> None:
    if args.test_mode:
        if not args.mock_data:
            raise SystemExit("--test_mode requires --mock_data")
        reader: TBReader | MockReader = MockReader(args.mock_data)
    else:
        reader = TBReader(args.log_dir)

    editor = ConfigEditor(args.config_path, dry_run=args.dry_run)
    state = MonitorState()

    if args.test_mode:
        latest = reader.latest_step()
        check_step = (latest // args.step_interval) * args.step_interval or latest
        records = run_all_checks(reader, editor, state, check_step)
        append_jsonl(args.output_log, records)
        print_summary(check_step, records)
        return

    print(
        f"Monitor started | log_dir={args.log_dir} | config={args.config_path} | "
        f"interval={args.step_interval} steps / {args.poll_interval}s poll"
        + (" | DRY RUN" if args.dry_run else "")
    )
    while True:
        try:
            reader.Reload()
            latest_step = reader.latest_step()
            if latest_step - state.last_checked_step >= args.step_interval:
                check_step = (latest_step // args.step_interval) * args.step_interval
                if check_step > state.last_checked_step:
                    records = run_all_checks(reader, editor, state, check_step)
                    append_jsonl(args.output_log, records)
                    print_summary(check_step, records)
                    state.last_checked_step = check_step
            time.sleep(args.poll_interval)
        except KeyboardInterrupt:
            print("\nMonitor stopped.")
            break


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GRPO training health monitor")
    p.add_argument("--log_dir", type=str, default=None, help="TensorBoard log directory")
    p.add_argument("--config_path", type=str, required=True, help="Training YAML config path")
    p.add_argument("--output_log", type=str, default="monitor_log.jsonl")
    p.add_argument("--poll_interval", type=int, default=30)
    p.add_argument("--step_interval", type=int, default=10)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--test_mode", action="store_true")
    p.add_argument("--mock_data", type=str, default=None, help="JSONL mock data for test mode")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if not args.test_mode and not args.log_dir:
        print("error: --log_dir is required unless --test_mode is set", file=sys.stderr)
        sys.exit(2)
    run_monitor(args)


if __name__ == "__main__":
    main()
