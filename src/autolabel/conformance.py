"""라벨링 연동 확인 도구 (라벨링 확장 규격 1.0 7장). 외부 개발사가 납품 전에 스스로 확인하고, 플랫폼 검수에도 같은 것을 쓴다.

    python -m visionops_labeling.conformance classifier --adapter dotting --weights w.keras --images samples/ [--device cpu]
                                                        [--plugin dotting_attention ...]

어댑터는 entry point 그룹 visionops.adapters(분류 어댑터 규격 1.0), 플러그인은 visionops.labeling_signals 에서 찾는다.
error 가 하나라도 있으면 종료 코드 1. warn 은 확인이 필요한 항목, info 는 참고 수치다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import plugins as plug
from .tasks.classify.model import BINARY_NAMES, ModelInfo
from .tasks.classify.tta import TRANSFORMS, apply, expand

ADAPTER_GROUP = "visionops.adapters"
ATOL = 1e-3  # 묶음 크기·반복 호출 결과 차이 허용치(GPU 연산 오차)


@dataclass
class Check:
    name: str
    level: str      # error · warn · info
    ok: bool
    message: str = ""


@dataclass
class Report:
    subject: str
    checks: list[Check] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)  # 참고 수치(이미지당 처리 시간 등)

    def add(self, name: str, ok: bool, message: str = "", level: str = "error") -> None:
        self.checks.append(Check(name, level, ok, message))

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks if c.level == "error")

    def to_text(self) -> str:
        lines = [f"[{self.subject}] {'통과' if self.ok else '실패'}"]
        for c in self.checks:
            mark = "OK " if c.ok else {"error": "ERR", "warn": "WRN"}.get(c.level, "INF")
            lines.append(f"  {mark} {c.name}" + (f" — {c.message}" if c.message else ""))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"subject": self.subject, "ok": self.ok, "checks": [c.__dict__ for c in self.checks]}


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.allclose(a, b, atol=ATOL))


def check_classifier(adapter: Any, model: Any, images: list[np.ndarray]) -> Report:
    """분류 어댑터·로드한 모델이 라벨링 쪽 규격을 지키는지: 점수·임베딩 모양·값, 묶음 크기·반복 호출 일관성, 입력 불변, TTA 선언."""
    report = Report(f"classifier:{getattr(adapter, 'name', '?')}")
    info = ModelInfo.from_adapter(adapter)
    names = list(getattr(model, "class_names", []))
    report.add("task", getattr(adapter, "task", None) == "classify", f"task={getattr(adapter, 'task', None)!r}")
    report.add("class_names", bool(names) and len(set(names)) == len(names), f"{names}")
    if len(names) == 2:
        report.add("binary_names", names == BINARY_NAMES, f"이진 분류는 {BINARY_NAMES} 순서여야 합니다: {names}")
    try:
        expand(info.tta)
        report.add("tta_names", True, f"{list(info.tta)}")
    except ValueError as error:
        report.add("tta_names", False, f"{error} (가능: {list(TRANSFORMS)})")
    report.add("max_batch", info.max_batch >= 1, f"max_batch={info.max_batch}")
    if not images:
        report.add("images", False, "확인할 이미지가 없습니다")
        return report
    try:
        _check_calls(report, model, images, info, names)
    except Exception as error:  # noqa: BLE001 — 어댑터 예외도 확인 결과로 알린다
        report.add("exception", False, f"{type(error).__name__}: {error}")
    return report


def _check_calls(report: Report, model: Any, images: list[np.ndarray], info: ModelInfo, names: list[str]) -> None:
    copies = [a.copy() for a in images]
    batch = images[:info.max_batch]
    started = time.perf_counter()
    scores = np.asarray(model.predict(batch))
    predict_seconds = time.perf_counter() - started
    report.add("predict_shape", scores.shape == (len(batch), len(names)), f"{scores.shape}, 기대 ({len(batch)}, {len(names)})")
    report.add("predict_dtype", scores.dtype == np.float32, f"{scores.dtype}")
    finite = bool(np.isfinite(scores).all())
    report.add("predict_range", finite and scores.min() >= -1e-6 and scores.max() <= 1 + 1e-6,
               f"min {float(np.nanmin(scores)):.4f}, max {float(np.nanmax(scores)):.4f}" if scores.size else "빈 결과")
    if len(names) > 2 and finite:
        report.add("softmax_rows", bool(np.allclose(scores.sum(1), 1.0, atol=ATOL)), "다중 분류 점수 행 합은 1 이어야 합니다",
                   level="warn")
    report.add("predict_repeat", _same(scores, np.asarray(model.predict(batch))), "같은 입력 두 번 결과가 같아야 합니다")
    singles = np.concatenate([np.asarray(model.predict([a])) for a in batch]) if batch else scores
    report.add("predict_batch", _same(scores, singles), "batch 1 과 max_batch 결과가 같아야 합니다")
    emb = np.asarray(model.embed(batch))
    report.add("embed_shape", emb.ndim == 2 and emb.shape[0] == len(batch) and emb.shape[1] > 0, f"{emb.shape}")
    report.add("embed_dtype", emb.dtype == np.float32, f"{emb.dtype}")
    report.add("embed_finite", bool(np.isfinite(emb).all()), "NaN·Inf 가 없어야 합니다")
    report.add("embed_repeat", _same(emb, np.asarray(model.embed(batch))), "같은 입력 두 번 결과가 같아야 합니다")
    single_emb = np.concatenate([np.asarray(model.embed([a])) for a in batch])
    report.add("embed_batch", _same(emb, single_emb), "batch 1 과 max_batch 결과가 같아야 합니다")
    report.add("input_unchanged", all(np.array_equal(a, b) for a, b in zip(images, copies)), "입력 이미지를 바꾸면 안 됩니다")
    report.add("predict_seconds", True, f"{predict_seconds:.3f}초 / {len(batch)}장", level="info")
    # TTA 선언: 변형마다 예측 클래스 유지 비율(정답을 바꾸지 않는 변환이면 높아야 한다)
    try:
        for variant in expand(info.tta):
            other = np.asarray(model.predict([apply(a, variant) for a in batch]))
            if other.shape != scores.shape:
                report.add(f"tta:{variant}", False, f"변형 이미지 점수 모양 {other.shape} 이 원본 {scores.shape} 과 다릅니다")
                continue
            kept = float(np.mean(other.argmax(1) == scores.argmax(1)))
            report.add(f"tta:{variant}", True, f"예측 클래스 유지 {kept:.0%} (README 에 적은 값 이상인지 확인)", level="info")
    except ValueError:
        pass
    report.metrics["predict_seconds_per_image"] = predict_seconds / len(batch)  # check_plugin 이 처리 시간 비교에 쓴다


def check_plugin(plugin: Any, model: Any, images: list[np.ndarray],
                 predict_seconds_per_image: float | None = None) -> Report:
    """신호 플러그인 규격: 속성, 결과 수·값·detail, 반복 일관성, 입력 불변, 실패 시 처리, 처리 시간."""
    report = Report(f"plugin:{getattr(plugin, 'name', '?')}")
    problems = plug.plugin_problems(plugin)
    report.add("attributes", not problems, "; ".join(problems))
    if problems or not images:
        if not images:
            report.add("images", False, "확인할 이미지가 없습니다")
        return report
    scores = np.asarray(model.predict(images))
    copies, score_copy = [a.copy() for a in images], scores.copy()
    started = time.perf_counter()
    records = plug.run([plugin], model, images, scores)
    seconds = time.perf_counter() - started
    errors = [r[plugin.name]["error"] for r in records if "error" in r[plugin.name]]
    report.add("compute", not errors, errors[0] if errors else f"{len(records)}장")
    if not errors:
        again = plug.run([plugin], model, images, scores)
        same = all(abs(a[plugin.name]["value"] - b[plugin.name]["value"]) <= ATOL for a, b in zip(records, again))
        report.add("repeat", same, "같은 입력 두 번 value 가 같아야 합니다")
        values = [r[plugin.name]["value"] for r in records]
        report.add("values", True, f"min {min(values):.4f}, max {max(values):.4f}", level="info")
    report.add("input_unchanged", all(np.array_equal(a, b) for a, b in zip(images, copies))
               and np.array_equal(scores, score_copy), "images·scores 를 바꾸면 안 됩니다")
    per_image = seconds / len(images)
    report.metrics["seconds_per_image"] = per_image
    message = f"이미지당 {per_image * 1000:.1f}ms"
    if predict_seconds_per_image:
        # 플러그인은 자동라벨링 묶음마다 돈다. 원본 예측의 2배(+5ms)를 넘으면 처리량에 영향이 크다
        report.add("seconds", per_image <= 2 * predict_seconds_per_image + 0.005,
                   message + f" (predict 이미지당 {predict_seconds_per_image * 1000:.1f}ms 의 2배 이내 권장)", level="warn")
    else:
        report.add("seconds", True, message, level="info")
    return report


# ── 명령 ───────────────────────────────────────────────────────────────────────
def _entry(group: str, name: str) -> Any:
    from importlib.metadata import entry_points

    for ep in entry_points(group=group):
        if ep.name == name:
            obj = ep.load()
            return obj() if isinstance(obj, type) else obj
    raise SystemExit(f"entry point {group}:{name} 를 찾지 못했습니다. 패키지(wheel)가 설치되어 있는지 확인하세요.")


def _images(folder: Path, decode: str, limit: int) -> list[np.ndarray]:
    from .images import read_image

    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
    return [read_image(str(p), 200_000_000, decode) for p in files[:limit]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m visionops_labeling.conformance")
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("classifier", help="분류 어댑터 + (선택) 신호 플러그인 확인")
    c.add_argument("--adapter", required=True, help="어댑터 이름 (entry point visionops.adapters)")
    c.add_argument("--weights", required=True, type=Path)
    c.add_argument("--images", required=True, type=Path, help="표본 이미지 폴더 (최대 --limit 장)")
    c.add_argument("--device", default="cpu")
    c.add_argument("--plugin", action="append", default=[], help="신호 플러그인 이름 (여러 번)")
    c.add_argument("--limit", type=int, default=32)
    c.add_argument("--json", type=Path, help="결과를 JSON 으로도 저장")
    args = parser.parse_args(argv)

    adapter = _entry(ADAPTER_GROUP, args.adapter)
    info = ModelInfo.from_adapter(adapter)
    model = adapter.load(args.weights, args.device, {})
    images = _images(args.images, info.decode, args.limit)
    reports = [check_classifier(adapter, model, images)]
    seconds = reports[0].metrics.get("predict_seconds_per_image")
    for name in args.plugin:
        reports.append(check_plugin(_entry(plug.ENTRY_POINT_GROUP, name), model, images, seconds))
    for report in reports:
        print(report.to_text())
    if args.json:
        args.json.write_text(json.dumps([r.to_dict() for r in reports], ensure_ascii=False, indent=1), encoding="utf-8")
    close = getattr(model, "close", None)
    if close:
        close()
    return 0 if all(r.ok for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
