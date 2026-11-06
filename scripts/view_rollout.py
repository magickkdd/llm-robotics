"""Render and locally serve an image-inspired LIBERO rollout inspector.

Usage: python scripts/view_rollout.py outputs/<budgeted-run> --serve
"""

from __future__ import annotations

import argparse
import datetime
import html
import json
import math
import re
import subprocess
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote
import os
import numpy as np


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _safe_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def _canonical_model(value: str) -> str:
    for provider in ("openai/", "anthropic/", "google/"):
        if value.startswith(provider):
            return value[len(provider):]
    return value


def _episode_context(source: Path, episode: int) -> tuple[Path, Path, dict, dict]:
    source = source.resolve()
    if source.name == "inspect":
        inspect_dir = source
        if re.fullmatch(r"episode_\d+", source.parent.name):
            episode_dir = source.parent
            run_dir = episode_dir.parent
        else:
            run_dir = source.parent
            episode_dir = run_dir
    elif (source / f"episode_{episode:02d}" / "inspect").is_dir():
        run_dir = source
        episode_dir = run_dir / f"episode_{episode:02d}"
        inspect_dir = episode_dir / "inspect"
    elif (source / "inspect").is_dir():
        run_dir = source
        episode_dir = source
        inspect_dir = source / "inspect"
    else:
        raise FileNotFoundError(f"Cannot find episode {episode:02d}/inspect under {source}")

    logs = sorted(path for path in inspect_dir.glob("*.json") if path.is_file())
    if not logs:
        raise FileNotFoundError(f"No Inspect Robots eval log found in {inspect_dir}")
    log = _load_json(logs[-1])
    summary_path = run_dir / "summary.json"
    summary = _load_json(summary_path) if summary_path.is_file() else {}
    results = summary.get("results") or []
    row = next((item for item in results if item.get("episode") == episode), {})
    if not row and summary.get("status"):
        row = summary
    return run_dir, episode_dir, log, row


def _transcript_messages(log: dict) -> list[dict]:
    sample = (log.get("samples") or [{}])[0]
    trial_transcripts = sample.get("policy_transcripts") or []
    messages = []
    for trial in trial_transcripts:
        if isinstance(trial, list):
            messages.extend(message for message in trial if isinstance(message, dict))
    return messages


def _parse_calls(messages: list[dict]) -> list[dict]:
    calls = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or tool_call
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"note": arguments}
            if not isinstance(arguments, dict):
                arguments = {}
            calls.append({
                "name": function.get("name", "tool"),
                "arguments": arguments,
                "step": _observation_step(messages, message),
            })
    return calls


def _observation_step(messages: list[dict], target: dict) -> int:
    try:
        index = messages.index(target)
    except ValueError:
        return 0
    for message in reversed(messages[:index]):
        content = message.get("content")
        if isinstance(content, list):
            text = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
        else:
            text = str(content or "")
        match = re.search(r"\(step (\d+)\)", text)
        if match:
            return int(match.group(1))
    return 0


def _initial_position(messages: list[dict]) -> list[float] | None:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        text = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
        match = re.search(r"state\[eef_state\]: x=([\d.+-]+) y=([\d.+-]+) z=([\d.+-]+)", text)
        if match:
            return [float(match.group(index)) for index in range(1, 4)]
    return None


def _trajectory(inspect_dir: Path, log: dict, initial: list[float] | None) -> list[dict]:
    sample = (log.get("samples") or [{}])[0]
    metadata = (sample.get("trial_metadata") or [{}])[0]
    relative = metadata.get("actions")
    if not relative:
        return []
    actions_path = inspect_dir / relative
    rows = _load_jsonl(actions_path)
    labels = next((row.get("labels") for row in rows if row.get("kind") == "header"), None) or []
    if not all(axis in labels for axis in ("x", "y", "z")):
        messages = _transcript_messages(log)
        points = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            text = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
            match = re.search(r"state\[sites\]: \[([^\]]+)\]", text)
            step_match = re.search(r"\(step (\d+)\)", text)
            if match:
                values = [float(value.strip()) for value in match.group(1).split(",")]
                if len(values) >= 6:
                    points.append({"x": values[3], "y": values[4], "z": values[5],
                                   "t": int(step_match.group(1)) if step_match else len(points)})
        if points:
            return points
        return []
    x_index, y_index, z_index = (labels.index(axis) for axis in ("x", "y", "z"))
    points = []
    if initial is not None:
        points.append({"x": initial[0], "y": initial[1], "z": initial[2], "t": 0})
    for row in rows:
        action = row.get("action")
        if not isinstance(action, list) or len(action) <= max(x_index, y_index, z_index):
            continue
        points.append({"x": action[x_index], "y": action[y_index], "z": action[z_index],
                       "t": int(row.get("t", len(points)))})
    return points


def _inspect_directory(run_dir: Path, episode_dir: Path) -> Path:
    inspect_dir = episode_dir / "inspect"
    if not inspect_dir.is_dir() and (run_dir / "inspect").is_dir():
        return run_dir / "inspect"
    return inspect_dir


def _video_path(run_dir: Path, episode_dir: Path, result: dict) -> Path | None:
    candidate = Path(result["video"]) if result.get("video") else None
    if candidate and candidate.is_file():
        return candidate.resolve()
    videos = sorted(episode_dir.rglob("rollout.mp4"))
    if videos:
        return videos[0].resolve()
    existing = next(iter(sorted(run_dir.glob("**/rollout.mp4"))), None)
    if existing:
        return existing.resolve()
    inspect_dir = _inspect_directory(run_dir, episode_dir)
    frame_dirs = sorted((inspect_dir / "frames").glob("*")) if (inspect_dir / "frames").is_dir() else []
    if not frame_dirs:
        return None
    frame_dir = frame_dirs[-1]
    video_path = run_dir / "rollout.mp4"
    python = os.environ.get("LIBERO_PYTHON", str(Path.home() / "miniconda3/envs/libero/bin/python"))
    encoder = Path(__file__).with_name("encode_rollout_frames.py")
    try:
        subprocess.run([python, str(encoder), str(frame_dir), str(video_path)],
                       check=True, capture_output=True, text=True, timeout=120)
        return video_path.resolve()
    except (OSError, subprocess.SubprocessError):
        return None


def _dashboard_data(run_dir: Path, episode_dir: Path, log: dict, result: dict) -> dict:
    inspect_dir = _inspect_directory(run_dir, episode_dir)
    messages = _transcript_messages(log)
    calls = _parse_calls(messages)
    trajectory = _trajectory(inspect_dir, log, _initial_position(messages))
    video = _video_path(run_dir, episode_dir, result)
    video_url = quote(video.relative_to(run_dir).as_posix(), safe="/") if video else None
    protocol = _load_json(run_dir / "protocol.json") if (run_dir / "protocol.json").is_file() else {}
    eval_info = log.get("eval") or {}
    sample = (log.get("samples") or [{}])[0]
    status = result.get("status") or sample.get("status") or log.get("status", "unknown")
    scorer_metrics = ((log.get("results") or {}).get("metrics") or {})
    task_score = scorer_metrics.get("success_at_end")
    if task_score is not None:
        outcome = "Success" if float(task_score) >= 1.0 else "Task Failure"
    elif status in {"success", "task_success", "failure", "task_failure", "error", "api_error_usage_unknown", "incomplete_budget"}:
        outcome = "Success" if status in {"success", "task_success"} else "Task Failure" if status == "task_failure" else "Failure" if status in {"failure", "error", "api_error_usage_unknown", "incomplete_budget"} else status
    else:
        outcome = status.replace("_", " ").title()
    cost = result.get("estimated_usd")
    if cost is None:
        usage_file = run_dir / "api" / "api_usage.jsonl"
        usage = _load_jsonl(usage_file)
        cost = sum(row.get("estimated_usd", 0) for row in usage)
    step_count = result.get("steps")
    if step_count is None:
        step_count = (log.get("stats") or {}).get("total_steps", len(trajectory))
    call_count = result.get("api_attempts", len(calls))
    return {
        "model": protocol.get("model") or (eval_info.get("policy_config") or {}).get("model", "LLM policy"),
        "effort": protocol.get("reasoning_effort") or (eval_info.get("policy_config") or {}).get("effort") or "—",
        "task": protocol.get("task_id", sample.get("scene_metadata", {}).get("libero_task_id", "—")),
        "state": sample.get("scene_metadata", {}).get("init_state", "—"),
        "outcome": outcome,
        "steps": int(step_count or 0),
        "calls": int(call_count or 0),
        "cost": f"${float(cost):.3f}" if cost is not None else "—",
        "instruction": sample.get("instruction", "LIBERO rollout"),
        "video": video_url,
        "calls_data": calls,
        "trajectory": trajectory,
        "error": sample.get("error") or log.get("error"),
    }


def _socket_opening_dashboard_data(source: Path) -> dict:
    rows = _load_jsonl(source / "trajectory.jsonl")
    points = []
    for row in rows:
        position = row.get("socket_opening_center_world_m")
        if not isinstance(position, list) or len(position) != 3:
            continue
        points.append({
            "x": float(position[0]),
            "y": float(position[1]),
            "z": float(position[2]),
            "t": float(row.get("sim_time_s", len(points))),
        })
    result_path = source / "result.json"
    result = _load_json(result_path) if result_path.is_file() else {}
    metadata_path = source / "metadata.json"
    metadata = _load_json(metadata_path) if metadata_path.is_file() else {}
    return {
        "model": "MuJoCo UR20",
        "effort": "Ground-truth trace",
        "task": "Socket search",
        "state": metadata.get("state_id", "—"),
        "outcome": str(result.get("status", "recording")).replace("_", " ").title(),
        "steps": len(points),
        "calls": 0,
        "cost": "—",
        "instruction": "Socket opening-center world trajectory",
        "video": None,
        "calls_data": [],
        "trajectory": points,
        "trajectory_label": "Socket opening center",
        "error": None,
    }


def render_dashboard(data: dict) -> str:
    escaped = html.escape
    video_panel = (
        f'<video id="rollout-video" controls playsinline src="{escaped(data["video"])}"></video>'
        if data["video"] else '<div class="empty-video">No rollout video was saved for this episode.</div>'
    )
    rollout_options = "".join(
        f'<option value="{escaped(item["url"])}" data-model="{escaped(_canonical_model(item.get("model", "")))}" '
        f'data-date="{escaped(item.get("date", ""))}" data-scene="{escaped(item.get("scene", ""))}" '
        f'data-outcome="{escaped(item.get("outcome", ""))}"'
        f'{" selected" if item["selected"] else ""}>'
        f'{escaped(item["label"])}</option>'
        for item in data.get("rollouts", [])
    )
    def filter_options(key: str, label: str) -> str:
        values = sorted({(_canonical_model(item.get(key, "")) if key == "model" else item.get(key, ""))
                         for item in data.get("rollouts", []) if item.get(key, "")},
                        reverse=key == "date")
        choices = ''.join(f'<option value="{escaped(value)}">{escaped(value)}</option>' for value in values)
        return f'<label>{label}<select class="rollout-filter" data-filter="{key}"><option value="">All</option>{choices}</select></label>'

    rollout_selector = (
        '<div class="run-picker"><div class="filter-row">'
        '<label for="rollout-search">Search<input id="rollout-search" type="search" placeholder="run / task / state / status"></label>'
        f'{filter_options("model", "Model")}{filter_options("date", "Date")}'
        f'{filter_options("scene", "Scene")}{filter_options("outcome", "Outcome")}'
        f'</div><div class="select-row"><label for="rollout-picker">Matching rollouts</label>'
        f'<select id="rollout-picker">{rollout_options}</select><span id="rollout-count" class="hint"></span></div></div>'
        if rollout_options else ""
    )
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LIBERO Rollout Inspector</title>
<style>
:root{color-scheme:dark;--bg:#07111c;--panel:#0e1c2a;--panel2:#0a1723;--line:#294156;--blue:#72c8ff;--muted:#9aabba;--white:#f3f7fb;--orange:#ff956f}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--white);font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1440px;margin:auto;padding:26px 28px 48px}.eyebrow{font-size:11px;letter-spacing:.16em;font-weight:800;color:var(--blue)}h1{font-size:38px;line-height:1.1;letter-spacing:-.035em;margin:14px 0 16px}.subtitle{color:var(--blue);font-weight:750;font-size:16px;margin-bottom:34px}
.stats{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);border-radius:10px;background:#101f2e;overflow:hidden;margin-bottom:34px}.stat{padding:14px 18px;text-align:center;border-right:1px solid var(--line)}.stat:last-child{border:0}.label{display:block;color:#aac0d2;text-transform:uppercase;letter-spacing:.06em;font-size:10px;margin-bottom:5px}.value{font-size:17px;font-weight:800}.failure{color:var(--orange)}.success{color:#77e0aa}
.run-picker{display:grid;gap:10px;margin:0 0 18px;color:var(--muted);font-size:12px}.filter-row,.select-row{display:flex;align-items:end;gap:10px;flex-wrap:wrap}.run-picker label{display:grid;gap:5px}.run-picker input,.run-picker select{background:var(--panel);color:var(--white);border:1px solid var(--line);border-radius:7px;padding:9px 12px;min-width:150px}.run-picker input{width:min(300px,80vw)}.run-picker select{max-width:min(760px,100%)}.select-row #rollout-picker{flex:1;min-width:min(420px,90vw)}
.columns{display:grid;grid-template-columns:1fr 1fr;gap:20px;align-items:start}.section-title{font-size:16px;margin:0 0 15px;font-weight:780}.video-wrap{border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#000;aspect-ratio:16/9;display:flex;align-items:center;justify-content:center}video{width:100%;height:100%;object-fit:contain}.empty-video{color:var(--muted);padding:24px}.hint{font-size:12px;color:var(--muted);margin:10px 0 0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:15px}.plot-head{display:flex;align-items:center;justify-content:space-between;color:var(--muted);font-size:11px;margin-bottom:10px}.toggle{border:1px solid var(--line);border-radius:8px;padding:7px 11px;color:#c0d1df;font-size:11px}.plot{display:block;width:100%;height:330px;background:radial-gradient(ellipse at 48% 48%,#12283b 0%,#091724 74%);border:1px solid #29465e;border-radius:9px}.legend{display:flex;gap:12px;align-items:center;border:1px solid #29465e;background:#0b1a27;border-radius:8px;padding:7px 10px;font-size:10px;color:#bdcbd8}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--blue);margin-right:5px}.diamond{display:inline-block;width:8px;height:8px;background:#ffae69;transform:rotate(45deg);margin-right:6px}.plot-controls{display:flex;gap:12px;align-items:center;margin:12px 2px 0}.plot-controls button{background:#193249;border:1px solid #41627b;color:#e3edf6;border-radius:8px;padding:10px 13px;font-weight:700;white-space:nowrap}.progress{flex:1;min-width:0}.progress-label{font-size:11px;color:#b6ccdc;font-weight:700;margin-bottom:5px}input[type=range]{width:100%;accent-color:var(--blue)}.plan-note{border-left:3px solid var(--blue);background:#0b1a27;color:#d0dce7;margin-top:10px;padding:12px;font-size:12px;min-height:48px}
.timeline{margin-top:16px}.timeline-head{display:flex;align-items:center;justify-content:space-between;color:var(--blue);font-weight:800;margin-bottom:12px}.timeline-list{max-height:420px;overflow:auto;display:grid;gap:10px;padding-right:5px}.event{background:#0a1723;border:1px solid #294156;border-radius:9px;padding:13px;cursor:pointer}.event:hover,.event.active{border-color:#58b7f4;background:#102131}.event-top{display:flex;gap:10px;align-items:center;color:var(--muted);font-size:11px}.time{min-width:72px;text-align:center;border:1px solid #59b6f4;border-radius:5px;color:var(--blue);font-weight:750;padding:3px}.event-title{color:#edf5fc;font-weight:750}.event-note{margin:9px 0 0;color:#d0dce7;line-height:1.55}.target{font-size:10px;color:var(--muted);margin-top:8px}.error{margin-top:16px;color:#ffc1ad;background:#2b1b1a;border:1px solid #70463e;border-radius:8px;padding:11px;overflow-wrap:anywhere}
@media(max-width:900px){main{padding:22px 16px}.columns{grid-template-columns:1fr}h1{font-size:32px}.stats{grid-template-columns:repeat(2,1fr)}.stat:nth-child(2){border-right:0}.stat:nth-child(-n+2){border-bottom:1px solid var(--line)}}
</style></head><body><main>
<div class="eyebrow">03 / ROLLOUT EVIDENCE</div><h1>Rollout inspector</h1>
__ROLLOUT_SELECTOR__
<div class="subtitle" id="subtitle"></div>
<div class="stats"><div class="stat"><span class="label">Outcome</span><span id="outcome" class="value"></span></div><div class="stat"><span class="label">Steps</span><span id="steps" class="value"></span></div><div class="stat"><span class="label">Calls</span><span id="calls-count" class="value"></span></div><div class="stat"><span class="label">Cost</span><span id="cost" class="value"></span></div></div>
<div class="columns"><section><h2 class="section-title">Rollout video</h2><div class="video-wrap">__VIDEO__</div><p class="hint">Agent view and wrist view are shown side by side when available.</p></section>
<section><h2 class="section-title">3D spatial plan</h2><div class="card"><div class="plot-head"><span>Position marker</span><span class="toggle">● __TRAJECTORY_LABEL__&nbsp;&nbsp; ◆ Recorded path</span></div><svg id="plot" class="plot" viewBox="0 0 600 330" role="img" aria-label="__TRAJECTORY_LABEL__ trajectory plot"></svg><div class="plot-controls"><button id="play">Play 3D timeline</button><div class="progress"><div class="progress-label">Trajectory progress</div><input id="progress" type="range" min="0" max="100" value="0"></div></div><div id="plan-note" class="plan-note"></div></div>
<div class="card timeline"><div class="timeline-head"><span>View full reasoning timeline</span><span id="call-total" class="hint"></span></div><div id="timeline" class="timeline-list"></div></div></section></div>
<div id="error" class="error" hidden></div></main>
<script>
const data=__DATA__;
const video=document.getElementById('rollout-video');
const events=data.calls_data||[], points=data.trajectory||[];
document.getElementById('subtitle').textContent=data.trajectory_label?`${data.trajectory_label} · World-frame MuJoCo ground truth`:`${data.model} / ${data.effort} · Task ${String(data.task).padStart(2,'0')} · state ${data.state}`;
document.getElementById('outcome').textContent=data.outcome;
document.getElementById('outcome').classList.add(data.outcome==='Success'?'success':'failure');
document.getElementById('steps').textContent=data.steps;
document.getElementById('calls-count').textContent=data.calls;
document.getElementById('cost').textContent=data.cost;
document.getElementById('call-total').textContent=`(${events.length} calls)`;
if(data.error){const error=document.getElementById('error');error.hidden=false;error.textContent=data.error;}
const svg=document.getElementById('plot');
const ns='http://www.w3.org/2000/svg';
function el(tag,attrs={}){const node=document.createElementNS(ns,tag);for(const [key,value] of Object.entries(attrs))node.setAttribute(key,value);return node;}
const plotPoints=points.length?points:[{x:0,y:0,z:0}];
const worldX=plotPoints.map(point=>point.x),worldY=plotPoints.map(point=>point.y),worldZ=plotPoints.map(point=>point.z);
const bounds={minX:Math.min(...worldX),maxX:Math.max(...worldX),minY:Math.min(...worldY),maxY:Math.max(...worldY),minZ:Math.min(0,...worldZ),maxZ:Math.max(...worldZ)};
for(const key of ['x','y']){const span=bounds[`max${key.toUpperCase()}`]-bounds[`min${key.toUpperCase()}`];if(span<.01){bounds[`min${key.toUpperCase()}`]-=.01;bounds[`max${key.toUpperCase()}`]+=.01;}}
const marginX=Math.max(.01,(bounds.maxX-bounds.minX)*.12),marginY=Math.max(.01,(bounds.maxY-bounds.minY)*.12);
bounds.minX-=marginX;bounds.maxX+=marginX;bounds.minY-=marginY;bounds.maxY+=marginY;
const rawProject=point=>[point.x-point.y,(point.x+point.y)*.35-point.z*1.6];
const boundsPoints=[...plotPoints,{x:bounds.minX,y:bounds.minY,z:0},{x:bounds.minX,y:bounds.maxY,z:0},{x:bounds.maxX,y:bounds.minY,z:0},{x:bounds.maxX,y:bounds.maxY,z:0},{x:bounds.minX,y:bounds.minY,z:bounds.maxZ}].map(rawProject);
const minScreenX=Math.min(...boundsPoints.map(point=>point[0])),maxScreenX=Math.max(...boundsPoints.map(point=>point[0]));
const minScreenY=Math.min(...boundsPoints.map(point=>point[1])),maxScreenY=Math.max(...boundsPoints.map(point=>point[1]));
const centerX=(minScreenX+maxScreenX)/2,centerY=(minScreenY+maxScreenY)/2;
const scale=Math.min(470/Math.max(maxScreenX-minScreenX,.01),225/Math.max(maxScreenY-minScreenY,.01));
function project(point){const raw=rawProject(point);return [300+(raw[0]-centerX)*scale,168+(raw[1]-centerY)*scale];}
const maxStep=Math.max(1,...points.map(point=>Number(point.t)||0));
function drawPlot(progress=1){svg.replaceChildren();
 if(points.length<1){const text=el('text',{x:300,y:175,fill:'#9aabba','text-anchor':'middle','font-size':13});text.textContent='No executed trajectory in this rollout';svg.append(text);return;}
 const floorCorners=[{x:bounds.minX,y:bounds.minY,z:0},{x:bounds.maxX,y:bounds.minY,z:0},{x:bounds.maxX,y:bounds.maxY,z:0},{x:bounds.minX,y:bounds.maxY,z:0}].map(project);
 svg.append(el('polygon',{points:floorCorners.map(point=>point.join(',')).join(' '),fill:'#173248',opacity:.56,stroke:'#4e7189','stroke-width':1}));
 for(let index=1;index<4;index++){const fraction=index/4,x=bounds.minX+(bounds.maxX-bounds.minX)*fraction,y=bounds.minY+(bounds.maxY-bounds.minY)*fraction;const alongX=[project({x,y:bounds.minY,z:0}),project({x,y:bounds.maxY,z:0})],alongY=[project({x:bounds.minX,y,z:0}),project({x:bounds.maxX,y,z:0})];for(const line of [alongX,alongY])svg.append(el('line',{x1:line[0][0],y1:line[0][1],x2:line[1][0],y2:line[1][1],stroke:'#36536a','stroke-width':1,opacity:.7}));}
 const axisBase={x:bounds.minX,y:bounds.minY,z:0};
 const axes=[{end:{x:bounds.maxX,y:bounds.minY,z:0},color:'#e98273',label:'X'},{end:{x:bounds.minX,y:bounds.maxY,z:0},color:'#6ed7a3',label:'Y'},{end:{x:bounds.minX,y:bounds.minY,z:bounds.maxZ||.1},color:'#72c8ff',label:'Z'}];
 for(const axis of axes){const origin=project(axisBase),end=project(axis.end);svg.append(el('line',{x1:origin[0],y1:origin[1],x2:end[0],y2:end[1],stroke:axis.color,'stroke-width':2}));const label=el('text',{x:end[0]+5,y:end[1]+3,fill:axis.color,'font-size':12,'font-weight':700});label.textContent=axis.label;svg.append(label);}
 const currentStep=Math.max(0,Math.min(maxStep,progress*maxStep));
 let visible=points.filter(point=>(Number(point.t)||0)<=currentStep);
 if(!visible.length)visible=[points[0]];
 const nextIndex=points.findIndex(point=>(Number(point.t)||0)>currentStep);
 if(nextIndex>0){const previous=points[nextIndex-1],next=points[nextIndex],span=(Number(next.t)||0)-(Number(previous.t)||0);const fraction=span>0?(currentStep-(Number(previous.t)||0))/span:0;visible=[...visible,{x:previous.x+(next.x-previous.x)*fraction,y:previous.y+(next.y-previous.y)*fraction,z:previous.z+(next.z-previous.z)*fraction,t:currentStep}];}
 const projected=visible.map(project);
 if(projected.length>1){svg.append(el('polyline',{points:projected.map(p=>p.join(',')).join(' '),fill:'none',stroke:'#72c8ff','stroke-width':3,'stroke-linecap':'round','stroke-linejoin':'round'}));}
 for(let index=0;index<projected.length;index+=Math.max(1,Math.floor(projected.length/18))){svg.append(el('circle',{cx:projected[index][0],cy:projected[index][1],r:3,fill:'#9bd9ff',opacity:.7}));}
 const currentPoint=visible[visible.length-1],current=projected[projected.length-1],ground=project({x:currentPoint.x,y:currentPoint.y,z:0});
 svg.append(el('line',{x1:current[0],y1:current[1],x2:ground[0],y2:ground[1],stroke:'#ffd166','stroke-width':2,'stroke-dasharray':'4 4',opacity:.9}));
 svg.append(el('ellipse',{cx:ground[0],cy:ground[1],rx:7,ry:3,fill:'#ffd166',opacity:.78}));
 svg.append(el('circle',{cx:current[0],cy:current[1],r:10,fill:'#72c8ff',stroke:'#e7f7ff','stroke-width':2}));
 svg.append(el('circle',{cx:current[0]-3,cy:current[1]-3,r:3,fill:'#fff',opacity:.9}));
 const coords=el('text',{x:Math.min(500,current[0]+14),y:Math.max(18,current[1]-12),fill:'#f5fbff','font-size':11,'font-weight':700,'paint-order':'stroke','stroke':'#091724','stroke-width':3});coords.textContent=`x ${currentPoint.x.toFixed(3)}  y ${currentPoint.y.toFixed(3)}  z ${currentPoint.z.toFixed(3)} m`;svg.append(coords);
}
drawPlot();
const timeline=document.getElementById('timeline');
if(!events.length){timeline.innerHTML='<div class="event"><div class="event-top">No model tool calls recorded</div><div class="event-note">The run ended before the model could issue an action.</div></div>';}
events.forEach((event,index)=>{const args=event.arguments||{},card=document.createElement('article');card.className='event';
 const eventStep=Number(event.step||0);const time=video&&video.duration?eventStep/maxStep*video.duration:eventStep/20;const stamp=`${Math.floor(time/60)}:${(time%60).toFixed(1).padStart(4,'0')}`;
 const targets=args.targets?JSON.stringify(args.targets):'';
 card.innerHTML=`<div class="event-top"><span class="time">${stamp}</span><span class="event-title">Plan ${index+1}</span><span>${event.name}</span></div><div class="event-note"></div><div class="target"></div>`;
 card.querySelector('.event-note').textContent=args.note||args.reason||args.hindsight||'';
 card.querySelector('.target').textContent=targets?`Target: ${targets}`:'';
 card.addEventListener('click',()=>{document.querySelectorAll('.event.active').forEach(node=>node.classList.remove('active'));card.classList.add('active');const fraction=eventStep/maxStep;if(video&&video.duration)video.currentTime=Math.min(time,video.duration);document.getElementById('progress').value=fraction*100;drawPlot(fraction);document.getElementById('plan-note').textContent=args.note||args.reason||'Selected action';});
 timeline.append(card);
});
document.getElementById('plan-note').textContent=points.length?`${points.length} ${data.trajectory_label||'end-effector'} samples · time in simulation seconds.`:'No trajectory was recorded for this episode.';
document.getElementById('progress').addEventListener('input',event=>{const fraction=Number(event.target.value)/100;if(video&&video.duration)video.currentTime=fraction*video.duration;drawPlot(fraction);});
document.getElementById('play').addEventListener('click',()=>{if(video){video.play();return;}const started=performance.now();const animate=now=>{const fraction=Math.min(1,(now-started)/6000);document.getElementById('progress').value=fraction*100;drawPlot(fraction);if(fraction<1)requestAnimationFrame(animate);};requestAnimationFrame(animate);});
function syncPlayback(){if(!video||!video.duration)return;const fraction=video.currentTime/video.duration;document.getElementById('progress').value=fraction*100;drawPlot(fraction);}
if(video){video.addEventListener('timeupdate',syncPlayback);video.addEventListener('seeked',syncPlayback);}
const picker=document.getElementById('rollout-picker');
const search=document.getElementById('rollout-search');
const filters=[...document.querySelectorAll('.rollout-filter')];
function filterRollouts(){if(!picker)return;const query=(search?.value||'').trim().toLowerCase();let visible=[];for(const option of picker.options){const matchesQuery=!query||option.text.toLowerCase().includes(query);const matchesFilters=filters.every(filter=>!filter.value||option.dataset[filter.dataset.filter]===filter.value);option.hidden=!(matchesQuery&&matchesFilters);if(!option.hidden)visible.push(option);}document.getElementById('rollout-count').textContent=`${visible.length} matching`;if(visible.length&&picker.selectedOptions[0]?.hidden)picker.value=visible[0].value;}
if(search)search.addEventListener('input',filterRollouts);filters.forEach(filter=>filter.addEventListener('change',filterRollouts));filterRollouts();
const params=new URLSearchParams(location.search);
if(search)search.value=params.get('q')||'';
filters.forEach(filter=>{filter.value=params.get(filter.dataset.filter)||'';});
function persistPickerState(){const state=new URLSearchParams();if(search?.value)state.set('q',search.value);filters.forEach(filter=>{if(filter.value)state.set(filter.dataset.filter,filter.value);});return state.toString();}
if(picker)picker.addEventListener('change',()=>{const query=persistPickerState();location.href=picker.value+(query?`?${query}`:'');});
</script></body></html>""".replace("__VIDEO__", video_panel).replace("__DATA__", _safe_json(data)).replace("__ROLLOUT_SELECTOR__", rollout_selector).replace("__TRAJECTORY_LABEL__", html.escape(data.get("trajectory_label", "End effector")))


def _rollout_entries(root: Path) -> list[tuple[Path, int, dict, dict]]:
    entries = []
    seen = set()
    inspect_dirs = sorted(
        path for path in root.rglob("inspect")
        if path.is_dir() and not any(part.startswith(("_trash_", "_archive_"))
                                      for part in path.relative_to(root).parts)
    )
    for inspect_dir in inspect_dirs:
        if not list(inspect_dir.glob("*.json")):
            continue
        match = re.fullmatch(r"episode_(\d+)", inspect_dir.parent.name)
        if match:
            episode = int(match.group(1))
            run_dir = inspect_dir.parent.parent
        else:
            episode = 0
            run_dir = inspect_dir.parent
        key = (run_dir.resolve(), episode)
        if key in seen:
            continue
        try:
            resolved_run, resolved_episode, log, result = _episode_context(run_dir, episode)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        entries.append((resolved_run, episode, log, result))
        seen.add(key)
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="budgeted run directory, episode directory, or Inspect log directory")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--serve", action="store_true", help="serve the report locally until stopped")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8301)
    args = parser.parse_args()

    source = args.source.resolve()
    if (source / "trajectory.jsonl").is_file():
        report = source / "socket_opening_trajectory.html"
        report.write_text(
            render_dashboard(_socket_opening_dashboard_data(source)), encoding="utf-8"
        )
        print(f"Viewer report: {report}")
        if args.serve:
            catalog_root = source.parent
            handler = lambda *handler_args, **handler_kwargs: SimpleHTTPRequestHandler(
                *handler_args, directory=str(catalog_root), **handler_kwargs
            )
            server = ThreadingHTTPServer((args.host, args.port), handler)
            report_url = quote(
                (source.name + "/" + report.name), safe="/"
            )
            print(f"Open http://{args.host}:{server.server_port}/{report_url}")
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
        return
    direct_source = (
        (source / f"episode_{args.episode:02d}" / "inspect").is_dir()
        or (source / "inspect").is_dir()
        or source.name == "inspect"
    )
    if direct_source:
        run_dir, _, _, _ = _episode_context(source, args.episode)
        catalog_root = run_dir.parent
    else:
        catalog_root = source
        entries = _rollout_entries(catalog_root)
        if not entries:
            raise FileNotFoundError(f"No rollout episodes found under {source}")
        run_dir, args.episode, _, _ = entries[0]
    entries = _rollout_entries(catalog_root)
    if not any(item[0] == run_dir and item[1] == args.episode for item in entries):
        entries.append((run_dir, args.episode, *_episode_context(run_dir, args.episode)[2:]))
    catalog = []
    for entry_run, episode, entry_log, result in entries:
        report_name = f"rollout_inspector_ep{episode:02d}.html"
        protocol_path = entry_run / "protocol.json"
        protocol = _load_json(protocol_path) if protocol_path.is_file() else {}
        try:
            folder_label = entry_run.relative_to(catalog_root).as_posix()
        except ValueError:
            folder_label = entry_run.name
        model_label = protocol.get("model") or (entry_log.get("eval", {}).get("policy_config", {}).get("model", ""))
        scene_label = protocol.get("suite") or protocol.get("scene") or entry_log.get("eval", {}).get("task", "")
        if not scene_label:
            path_parts = [part.lower() for part in entry_run.relative_to(catalog_root).parts]
            scene_label = next((part for part in path_parts if part.startswith("libero_")), "")
        if not scene_label and any("ur20" in part.lower() for part in entry_run.parts):
            scene_label = "ur20_screw_station"
        if not scene_label:
            scene_label = "unknown"
        date_match = re.search(r"(?<!\d)(20\d{6})(?!\d)", entry_run.as_posix())
        if date_match:
            run_date = f"{date_match.group(1)[:4]}-{date_match.group(1)[4:6]}-{date_match.group(1)[6:8]}"
        else:
            run_date = datetime.datetime.fromtimestamp(entry_run.stat().st_mtime).strftime("%Y-%m-%d")
        outcome_label = str(result.get("status") or entry_log.get("status") or "recorded").replace("_", " ")
        label = (f"{folder_label}/episode_{episode:02d} · {model_label or 'unknown model'} · "
                 f"{scene_label} · {outcome_label} · {result.get('steps', '—')} steps")
        catalog.append({"run": entry_run, "episode": episode, "report": report_name,
                        "label": label, "model": model_label or "unknown",
                        "date": run_date, "scene": scene_label, "outcome": outcome_label})
    selected_report = run_dir / f"rollout_inspector_ep{args.episode:02d}.html"
    for entry_run, episode, log, result in entries:
        candidate_episode = entry_run / f"episode_{episode:02d}"
        episode_dir = candidate_episode if candidate_episode.is_dir() else entry_run
        data = _dashboard_data(entry_run, episode_dir, log, result)
        data["rollouts"] = [
            {"url": quote(os.path.relpath(item['run'] / item['report'], start=entry_run), safe="/"),
             "label": item["label"],
             "selected": item["run"] == entry_run and item["episode"] == episode,
             "model": item["model"], "date": item["date"],
             "scene": item["scene"], "outcome": item["outcome"]}
            for item in catalog
        ]
        report = entry_run / f"rollout_inspector_ep{episode:02d}.html"
        report.write_text(render_dashboard(data), encoding="utf-8")
    report = selected_report
    print(f"Viewer report: {report}")
    if args.serve:
        os.chdir(catalog_root)
        handler = lambda *handler_args, **handler_kwargs: SimpleHTTPRequestHandler(
            *handler_args, directory=str(catalog_root), **handler_kwargs
        )
        server = ThreadingHTTPServer((args.host, args.port), handler)
        report_url = quote(os.path.relpath(report, start=catalog_root).replace(os.sep, "/"), safe="/")
        print(f"Open http://{args.host}:{server.server_port}/{report_url}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
