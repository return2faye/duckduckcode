from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
import sqlite3
from typing import Any

from .runner import DEFAULT_DATABASE, _initialize_database

DEFAULT_REPORT = Path(".duckduckcode/eval-reports/eval-report.html")


def generate_report(
    database_path: Path = DEFAULT_DATABASE,
    output_path: Path = DEFAULT_REPORT,
    batch_id: str | None = None,
) -> Path:
    if not database_path.is_file():
        raise RuntimeError(f"Evaluation database '{database_path}' does not exist")
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        _initialize_database(connection)
        selected_batch = batch_id or _latest_batch(connection)
        rows = connection.execute(
            "SELECT evaluations.*, cases.case_json FROM evaluations "
            "JOIN cases ON cases.id = evaluations.case_id "
            "WHERE batch_id = ? ORDER BY evaluations.id",
            (selected_batch,),
        ).fetchall()
    if not rows:
        raise RuntimeError(f"Evaluation batch '{selected_batch}' was not found")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render(selected_batch, rows), encoding="utf-8")
    return output_path.resolve()


def _latest_batch(connection: sqlite3.Connection) -> str:
    row = connection.execute(
        "SELECT batch_id FROM evaluations ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        raise RuntimeError("Evaluation database has no runs")
    return str(row[0])


def _render(batch_id: str, rows: list[sqlite3.Row]) -> str:
    passed = sum(row["passed"] for row in rows)
    tokens = sum(row["token_usage"] + row["judge_token_usage"] for row in rows)
    duration = sum(row["duration_seconds"] for row in rows)
    max_tokens = max(1, *(row["token_usage"] for row in rows))
    cards = "".join(_case_card(row, max_tokens) for row in rows)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DuckDuckCode eval report</title>
<style>
:root{{--bg:#0b1020;--panel:#151c31;--muted:#98a2b8;--text:#edf2ff;--ok:#48d597;--bad:#ff6b7a;--accent:#7aa2ff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}}
main{{max-width:1100px;margin:auto;padding:32px 20px}}h1,h2,p{{margin-top:0}}.muted{{color:var(--muted)}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:24px 0}}
.metric,.case{{background:var(--panel);border:1px solid #28324f;border-radius:12px;padding:16px}}.metric strong{{display:block;font-size:22px}}
.case{{margin:16px 0}}.head{{display:flex;justify-content:space-between;gap:16px;align-items:start}}.pass{{color:var(--ok)}}.fail{{color:var(--bad)}}
.bar{{height:7px;background:#27304a;border-radius:9px;overflow:hidden;margin:8px 0 14px}}.bar i{{display:block;height:100%;background:var(--accent)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}}.compact,.event{{border-left:3px solid var(--accent);padding:8px 10px;margin:8px 0;background:#10162a}}.event.error{{border-color:var(--bad)}}
details{{margin-top:10px}}summary{{cursor:pointer;color:#b9c8ef}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#0d1325;padding:12px;border-radius:8px;max-height:420px;overflow:auto}}
</style></head><body><main>
<h1>DuckDuckCode evaluation</h1><p class="muted">batch {escape(batch_id)}</p>
<section class="metrics"><div class="metric"><span>Pass rate</span><strong>{passed}/{len(rows)}</strong></div><div class="metric"><span>Total tokens</span><strong>{tokens:,}</strong></div><div class="metric"><span>Duration</span><strong>{duration:.1f}s</strong></div><div class="metric"><span>Compactions</span><strong>{sum(row['compactions'] for row in rows)}</strong></div></section>
{cards}
</main></body></html>"""


def _case_card(row: sqlite3.Row, max_tokens: int) -> str:
    bench = _json(row["case_json"], {})
    metadata = bench.get("metadata", {})
    tool_events = _json(row["tool_events"], [])
    compactions = [
        event
        for event in _json(row["compaction_events"], [])
        if event.get("status") == "completed"
    ]
    tests = _json(row["test_results"], [])
    validation = _json(row["validation_errors"], [])
    verdict = "PASS" if row["passed"] else "FAIL"
    css = "pass" if row["passed"] else "fail"
    score = "—" if row["score"] is None else row["score"]
    width = row["token_usage"] / max_tokens * 100
    return f"""<article class="case">
<div class="head"><div><h2>{escape(row['case_id'])}</h2><span class="muted">{escape(str(metadata.get('suite', '')))} · {escape(str(metadata.get('category', '')))}</span></div><strong class="{css}">{verdict} · {score}/4</strong></div>
<p>{escape(row['reason'])}</p><div class="bar" title="Agent tokens"><i style="width:{width:.1f}%"></i></div>
<div class="grid"><div><b>Models</b><br>{escape(row['agent_model'])}<br><span class="muted">Judge: {escape(row['judge_model'])}</span></div><div><b>Usage</b><br>{row['token_usage']:,} agent + {row['judge_token_usage']:,} judge<br><span class="muted">{row['duration_seconds']:.1f}s</span></div><div><b>Context</b><br>{row['compactions']} compactions<br><span class="muted">{escape(row['status'])}</span></div></div>
{_compaction_html(compactions)}
{_events_html(tool_events)}
{_details('Required tests', tests)}{_details('Validation errors', validation)}{_details('Final answer', row['final_answer'])}{_details('Workspace diff', row['workspace_diff'])}{_details('Runtime errors', row['error'])}
</article>"""


def _compaction_html(events: list[dict[str, Any]]) -> str:
    if not events:
        return '<p class="muted">No context compaction.</p>'
    rendered = "".join(_compaction_event_html(event) for event in events)
    return f"<details open><summary>Context compaction</summary>{rendered}</details>"


def _compaction_event_html(event: dict[str, Any]) -> str:
    before = int(event.get("before_tokens", 0))
    after = int(event.get("after_tokens", 0))
    saved = before - after
    percentage = saved / before * 100 if before else 0
    return (
        f'<div class="compact"><b>Turn {event.get("turn", "?")}: '
        f"{before:,} → {after:,} estimated tokens · "
        f"saved {saved:,} ({percentage:.1f}%)</b>"
        f'<pre>{escape(str(event.get("summary", "")))}</pre></div>'
    )


def _events_html(events: list[dict[str, Any]]) -> str:
    if not events:
        return ""
    rendered = "".join(_tool_event_html(event) for event in events)
    calls = sum(event.get("type") == "tool_call" for event in events)
    return f"<details><summary>Tool trace ({calls} calls)</summary>{rendered}</details>"


def _tool_event_html(event: dict[str, Any]) -> str:
    event_type = str(event.get("type", "event"))
    name = event.get("name", "")
    status = "error" if event.get("is_error") else str(event.get("decision", ""))
    detail = {
        key: event[key] for key in ("call_id", "arguments", "content") if key in event
    }
    rendered_detail = (
        f"<pre>{escape(json.dumps(detail, ensure_ascii=False, indent=2, default=str))}</pre>"
        if detail
        else ""
    )
    css = "event error" if event.get("is_error") else "event"
    suffix = f" · {escape(status)}" if status else ""
    label = (
        f'turn {event.get("turn", "?")} · {escape(event_type)} · '
        f"{escape(str(name))}{suffix}"
    )
    if event_type == "tool_result":
        return (
            f'<details class="{css}"><summary><b>{label}</b></summary>'
            f"{rendered_detail}</details>"
        )
    return f'<div class="{css}"><b>{label}</b>{rendered_detail}</div>'


def _details(title: str, value: Any) -> str:
    if value in ("", [], None):
        return ""
    text = (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, indent=2)
    )
    return f"<details><summary>{escape(title)}</summary><pre>{escape(text)}</pre></details>"


def _json(value: str, default: Any) -> Any:
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def main() -> None:
    parser = argparse.ArgumentParser(description="Render DuckDuckCode eval results.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--batch")
    args = parser.parse_args()
    try:
        print(generate_report(args.database, args.output, args.batch))
    except RuntimeError as exc:
        parser.exit(2, f"duckduckcode-eval-report: error: {exc}\n")


if __name__ == "__main__":
    main()
