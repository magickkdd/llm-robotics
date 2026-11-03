"""Optional live dashboard for UR20 rollout progress."""

from __future__ import annotations

import json
import re
import shutil
import textwrap


class UR20Dashboard:
    COLORS = {
        "reset": "\033[0m",
        "cyan": "\033[36m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "magenta": "\033[35m",
        "red": "\033[31m",
        "dim": "\033[2m",
        "bold": "\033[1m",
        "blue": "\033[34m",
    }

    def __init__(self, stream):
        self.stream = stream
        self.tty = stream.isatty()
        self.pending = ""
        self.events: list[str] = []
        self.model = "—"
        self.state_id = "—"
        self.max_calls = 0
        self.calls = 0
        self.max_steps = 0
        self.steps = 0
        self.physics_steps = 0
        self.estimated_cost = "$0.00"
        self.api_status = "Waiting to start"
        self.task_status = "IN PROGRESS"
        self.socket_engaged = False
        self.engaged_seen = False
        self.screw_lift = 0.0
        self.rotation = 0.0
        self.current_action = "Waiting for policy action"
        self.targets = "—"
        self.note = "The live plan and tool call will appear here."

    @property
    def encoding(self):
        return self.stream.encoding

    def isatty(self):
        return self.tty

    def fileno(self):
        return self.stream.fileno()

    def _paint(self, text: str, color: str) -> str:
        if not self.tty:
            return text
        return f"{self.COLORS[color]}{text}{self.COLORS['reset']}"

    def _remember(self, message: str) -> None:
        self.events.append(message)
        self.events = self.events[-7:]

    def _handle(self, line: str) -> None:
        if line.startswith("[UR20] "):
            model = re.search(r"model=([^ ]+)", line)
            state = re.search(r"state=([^ ]+)", line)
            limits = re.search(r"limits=(\d+) calls/(\d+) sim steps", line)
            if model:
                self.model = model.group(1)
            if state:
                self.state_id = state.group(1)
            if limits:
                self.max_calls, self.max_steps = map(int, limits.groups())
            self._remember("Rollout started")
        elif line.startswith("[episode "):
            call_match = re.search(r"call (\d+)/(\d+)", line)
            if call_match:
                self.calls, self.max_calls = map(int, call_match.groups())
            message = line.split("] ", 1)[-1]
            cost_match = re.search(r"run ~\$(\d+\.\d+)", message)
            if cost_match:
                self.estimated_cost = f"${float(cost_match.group(1)):.2f}"
            self.api_status = message
            self._remember(f"API · {message}")
        elif line.startswith("[agent] << tool_call "):
            payload = line[len("[agent] << tool_call "):]
            name, separator, raw_arguments = payload.partition("(")
            try:
                arguments = json.loads(raw_arguments[:-1]) if separator and raw_arguments.endswith(")") else {}
            except json.JSONDecodeError:
                arguments = {}
            self.current_action = name or "Tool call"
            self.targets = ", ".join(
                f"{key}={value}" for key, value in arguments.get("targets", {}).items()
            ) or "No explicit targets"
            self.note = arguments.get("note", "No policy note provided.")
            self._remember(f"ACTION · {self.current_action} · {self.targets}")
        elif line.startswith("[agent] -- "):
            self._remember(f"EXEC · {line[len('[agent] -- '):]}")
        elif line.startswith("[UR20 step "):
            step_match = re.search(r"step (\d+)/(\d+) \| physics (\d+)", line)
            if step_match:
                self.steps, self.max_steps, self.physics_steps = map(int, step_match.groups())
            self.socket_engaged = "socket_engaged=True" in line
            self.engaged_seen = "ever_engaged=True" in line
            lift = re.search(r"screw_lift=(-?[\d.]+)m", line)
            rotation = re.search(r"rotation=(-?[\d.]+)deg", line)
            if lift:
                self.screw_lift = float(lift.group(1))
            if rotation:
                self.rotation = float(rotation.group(1))
            self._remember(f"SIM · {line.split('] ', 1)[-1]}")
        elif line.startswith("[agent] << "):
            self._remember(f"POLICY · {line[len('[agent] << '):]}")
        elif any(word in line.lower() for word in ("error", "failed", "exhausted", "stopping")):
            self._remember(f"ERROR · {line}")
        else:
            self._remember(line)

        if self.tty:
            self._render()
        else:
            self.stream.write(line + "\n")
            self.stream.flush()

    def _panel(self, title: str, body: list[str], width: int, color: str = "cyan") -> list[str]:
        inner = max(12, width - 4)
        colored_title = self._paint(title, color)
        output = [f"┌─ {colored_title} " + "─" * max(0, width - len(title) - 5) + "┐"]
        for line in body:
            wrapped = textwrap.wrap(line, width=inner) or [""]
            output.extend(f"│ {item:<{inner}} │" for item in wrapped)
        output.append("└" + "─" * (width - 2) + "┘")
        return output

    def _render(self) -> None:
        size = shutil.get_terminal_size((100, 32))
        width = max(60, size.columns)
        gap = 2
        left_width = max(25, (width - gap - 2) // 3)
        right_width = width - left_width - gap
        head = self._paint("UR20 SCREW REMOVAL  ·  LIVE ROLLOUT", "bold")
        subhead = f"{self.model}  ·  state {self.state_id}  ·  estimated spend {self.estimated_cost}"
        calls = f"LLM CALLS   {self.calls:>3}/{self.max_calls or '—'}"
        sim_steps = f"SIM STEPS   {self.steps:>3}/{self.max_steps or '—'}  ·  physics {self.physics_steps}"
        physical = [
            f"Task       {self.task_status}",
            f"Socket     {'ENGAGED' if self.socket_engaged else 'not engaged'}",
            f"Engaged    {'seen' if self.engaged_seen else 'not yet'}",
            f"Screw lift {self.screw_lift:.4f} m",
            f"Rotation   {self.rotation:.1f}°",
            f"API        {self.api_status}",
        ]
        action = [self.current_action, f"Targets: {self.targets}", "", *textwrap.wrap(self.note, right_width - 5)]
        left = self._panel("SESSION / PHYSICAL STATE", [calls, sim_steps, "", *physical], left_width, "blue")
        right = self._panel("CURRENT ACTION / POLICY NOTE", action, right_width, "yellow")
        rows = [head, subhead, ""]
        panel_rows = max(len(left), len(right))
        rows.extend(
            (left[index] if index < len(left) else " " * left_width)
            + " " * gap
            + (right[index] if index < len(right) else " " * right_width)
            for index in range(panel_rows)
        )
        event_width = width - 2
        rows.extend(["", *self._panel("RECENT EVENTS", self.events[-5:], event_width, "magenta")])
        output = "\033[?25l\033[2J\033[H" + "\n".join(rows)
        self.stream.write(output)
        self.stream.flush()

    def write(self, text: str) -> int:
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            self._handle(line)
        return len(text)

    def flush(self) -> None:
        if self.pending:
            self._handle(self.pending)
            self.pending = ""
        self.stream.flush()

    def close(self) -> None:
        if self.tty:
            self.stream.write("\033[?25h\033[0m\n")
            self.stream.flush()

    def finish(self, succeeded: bool) -> None:
        self.task_status = "SUCCESS" if succeeded else "TASK FAILURE"
        self._remember(f"RESULT · {self.task_status}")
        if self.tty:
            self._render()
