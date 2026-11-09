"""Optional live dashboard for UR20 rollout progress."""

from __future__ import annotations

import json
import re
import shutil
import threading
import textwrap
import unicodedata


class UR20Dashboard:
    ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    COLORS = {
        "reset": "\033[0m",
        "cyan": "\033[38;5;45m",
        "green": "\033[38;5;82m",
        "yellow": "\033[38;5;220m",
        "magenta": "\033[38;5;207m",
        "red": "\033[38;5;196m",
        "dim": "\033[2m",
        "bold": "\033[1m",
        "blue": "\033[38;5;75m",
        "orange": "\033[38;5;208m",
        "purple": "\033[38;5;141m",
        "gray": "\033[38;5;250m",
        "teal": "\033[38;5;43m",
        "white": "\033[38;5;255m",
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
        self.observation = "Waiting for the first observation"
        self.input_summary = "Waiting for observation input"
        self.policy_output = "Waiting for model response"
        self.approval = "No action processed yet"
        self.state_delta = "No simulation update yet"
        self.force_status = "Force feedback disabled"
        self.frame = 0
        self.lock = threading.RLock()
        self.stop_animation = threading.Event()
        self.animation = None
        if self.tty:
            self.stream.write("\033[?1049h\033[?25l")
            self.stream.flush()
            self.animation = threading.Thread(target=self._animate, daemon=True)
            self.animation.start()
            self._render()

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
        self.events.append(" ".join(message.split()))
        self.events = self.events[-20:]

    def _handle(self, line: str) -> None:
        with self.lock:
            self._process_line(line)

    def _process_line(self, line: str) -> None:
        if line.startswith("[UR20 phase] "):
            phase = line[len("[UR20 phase] "):].strip()
            self.approval = phase
            self._remember(f"PHASE · {phase}")
        elif line.startswith("[UR20 force] "):
            payload = line[len("[UR20 force] "):].strip()
            self.force_status = payload
            self._remember(f"FORCE · {payload}")
        elif line.startswith("[UR20] "):
            model = re.search(r"model=([^ ]+)", line)
            state = re.search(r"state=([^ ]+)", line)
            limits = re.search(r"limits=(\d+) calls/(\d+) sim steps", line)
            if model:
                self.model = model.group(1)
            if state:
                self.state_id = state.group(1)
            if limits:
                self.max_calls, self.max_steps = map(int, limits.groups())
            self._remember("SESSION · rollout started")
        elif line.startswith("[episode "):
            call_match = re.search(r"call (\d+)/(\d+)", line)
            if call_match:
                self.calls, self.max_calls = map(int, call_match.groups())
            message = line.split("] ", 1)[-1]
            cost_match = re.search(r"run ~\$(\d+\.\d+)", message)
            if cost_match:
                self.estimated_cost = f"${float(cost_match.group(1)):.2f}"
            self.api_status = message
            brief = re.sub(r"episode ~\$\d+\.\d+,?\s*", "", message)
            brief = re.sub(r"run ~\$\d+\.\d+.*", "", brief).strip(" ;")
            self._remember(f"API · {brief}")
        elif line.startswith("[agent] << tool_call "):
            payload = line[len("[agent] << tool_call "):]
            name, separator, raw_arguments = payload.partition("(")
            try:
                arguments = json.loads(raw_arguments[:-1]) if separator and raw_arguments.endswith(")") else {}
            except json.JSONDecodeError:
                arguments = {}
            self.current_action = name or "Tool call"
            self.targets = ", ".join(
                f"{key}={value}" for key, value in arguments.get(
                    "targets", arguments.get("deltas", {})
                ).items()
            ) or "No explicit targets"
            self.note = arguments.get("note", "No policy note provided.")
            self.policy_output = f"{self.current_action} · {self.targets}"
            self._remember(f"ACTION · {self.current_action} · {self.targets}")
        elif line.startswith("[agent] -- "):
            message = line[len("[agent] -- "):]
            self.approval = message
            self._remember(f"EXEC · {message}")
        elif line.startswith("[agent] goal"):
            return
        elif line.startswith("[agent] >> step "):
            step = re.search(r"step (\d+)", line)
            cameras = re.search(r"(\d+) camera", line)
            sites = re.search(r"state\[sites\]: \[([^]]+)\]", line)
            task_state = re.search(r"state\[task_state\]: \[([^]]+)\]", line)
            tip = "tip unavailable"
            if sites:
                values = [float(value.strip()) for value in sites.group(1).split(",")]
                if len(values) >= 6:
                    tip = "tip=(%.3f, %.3f, %.3f)" % tuple(values[3:6])
            state = ""
            if task_state:
                values = [value.strip() for value in task_state.group(1).split(",")]
                if len(values) >= 3:
                    self.screw_lift = float(values[0])
                    self.socket_engaged = float(values[1]) > 0
                    self.engaged_seen = float(values[2]) > 0
                    state = f" · socket={'engaged' if self.socket_engaged else 'clear'}"
            self._remember(
                f"OBS · step {step.group(1) if step else '?'} · "
                f"{cameras.group(1) if cameras else '?'} views · {tip}{state}"
            )
            self.observation = (f"{cameras.group(1) if cameras else '?'} camera views · "
                                f"{tip}{state or ' · awaiting state labels'}")
        elif line.startswith("[agent] >> input: "):
            payload = line[len("[agent] >> input: "):]
            observation_text = payload.split("Instruction:", 1)[0].strip(" |.")
            observation_text = textwrap.shorten(observation_text, width=100, placeholder="…")
            self.input_summary = (f"{observation_text or 'Observation refreshed'} · "
                                  "task instruction held constant")
            self.observation = self.input_summary
            self._remember("INPUT · refreshed camera/state observation; fixed instruction omitted")
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
            self._remember(
                f"SIM · step {self.steps}/{self.max_steps} · "
                f"socket={'engaged' if self.socket_engaged else 'clear'} · "
                f"lift={self.screw_lift:.3f}m · turn={self.rotation:.1f}°"
            )
            self.state_delta = (f"step {self.steps}/{self.max_steps} · "
                                f"socket={'engaged' if self.socket_engaged else 'clear'} · "
                                f"lift={self.screw_lift:.3f}m · turn={self.rotation:.1f}°")
        elif line.startswith("[agent] << "):
            self.policy_output = "Text response received · awaiting structured tool call"
            self._remember("POLICY · text response received; awaiting tool call")
        elif any(word in line.lower() for word in ("error", "failed", "exhausted", "stopping")):
            self.api_status = line.strip()
            self._remember(f"ERROR · {line}")
        else:
            return

        if self.tty:
            self._render()
        else:
            self.stream.write(line + "\n")
            self.stream.flush()

    def _panel(self, title: str, body: list[str], width: int, color: str = "cyan",
               max_body_lines: int = 7) -> list[str]:
        width = max(12, width)
        inner = width - 4
        title = textwrap.shorten(title, width=max(1, width - 5), placeholder="…")
        colored_title = self._paint(title, color)
        title_fill = max(0, width - self._display_width(title) - 5)
        output = [f"┌─ {colored_title} " + "─" * title_fill + "┐"]
        for line in body:
            for item in self._wrap_display(line, inner):
                padding = max(0, inner - self._display_width(item))
                output.append(f"│ {item}{' ' * padding} │")
        output = output[:max_body_lines + 1]
        output.append("└" + "─" * (width - 2) + "┘")
        return output

    @classmethod
    def _display_width(cls, text: str) -> int:
        visible = cls.ANSI_RE.sub("", text)
        return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in visible)

    @classmethod
    def _wrap_display(cls, text: str, width: int) -> list[str]:
        tokens = re.findall(r"\x1b\[[0-?]*[ -/]*[@-~]|\s+|[^\s\x1b]+|\x1b.", text)
        lines: list[str] = []
        current = ""
        used = 0
        for token in tokens:
            if cls.ANSI_RE.fullmatch(token):
                current += token
                continue
            token_width = cls._display_width(token)
            if token.isspace() and not current:
                continue
            if used and used + token_width > width:
                lines.append(current)
                current, used = "", 0
                if token.isspace():
                    continue
            if token_width > width:
                chars = list(token)
                for char in chars:
                    char_width = cls._display_width(char)
                    if used and used + char_width > width:
                        lines.append(current)
                        current, used = "", 0
                    current += char
                    used += char_width
            else:
                current += token
                used += token_width
        if current or not lines:
            lines.append(current)
        return lines

    def _phase(self) -> int:
        if self.task_status == "SUCCESS":
            return 5
        if self.socket_engaged or self.engaged_seen:
            return 4 if self.rotation < 120 else 5
        note = self.note.lower()
        if any(word in note for word in ("seat", "contact", "descend", "lower", "approach")):
            return 2
        if self.calls == 0:
            return 0
        return 1

    @classmethod
    def _fit(cls, value: str, width: int) -> str:
        if cls._display_width(value) <= width:
            return value
        if width <= 1:
            return "…"[:width]
        result = ""
        used = 0
        for char in value:
            char_width = cls._display_width(char)
            if used + char_width > width - 1:
                break
            result += char
            used += char_width
        return result + "…"

    def _field(self, label: str, value: str, width: int, label_color: str,
               value_color: str) -> str:
        prefix = f"{self._paint(label, label_color)}  "
        available = max(1, width - self._display_width(prefix))
        return prefix + self._paint(self._fit(value, available), value_color)

    @staticmethod
    def _bar(value: float, maximum: float, width: int = 14) -> str:
        fraction = min(1.0, max(0.0, value / maximum)) if maximum > 0 else 0.0
        filled = int(round(fraction * width))
        return "█" * filled + "░" * (width - filled)

    def _render(self) -> None:
        size = shutil.get_terminal_size((100, 32))
        width = max(20, size.columns)
        gap = 2
        self.frame += 1
        spinner = "◢◣◤◥"[self.frame % 4]
        phases = ["LOCATE", "ALIGN", "APPROACH", "SEAT", "VERIFY"]
        active_phase = min(self._phase(), len(phases) - 1)
        flow_parts = []
        for index, name in enumerate(phases):
            color = "green" if index < active_phase else "orange" if index == active_phase else "gray"
            flow_parts.append(self._paint(f"◆ {name}" if index == active_phase else name, color))
        flow = f"  {self._paint('━━━' if self.frame % 2 else '┄┄┄', 'teal')}➤  ".join(flow_parts)
        head = self._paint("UR20", "cyan") + self._paint("  /  HEX SOCKET SEATING", "bold")
        subhead = (self._paint(self.model, "blue") + self._paint("  ·  state ", "gray")
                   + self._paint(self.state_id, "orange") + self._paint("  ·  est. spend ", "gray")
                   + self._paint(self.estimated_cost, "green"))
        calls = (self._paint("CALLS  ", "cyan")
                 + self._paint(self._bar(self.calls, self.max_calls), "cyan")
                 + self._paint(f"  {self.calls}/{self.max_calls or '—'}", "white"))
        step_line = (self._paint("SIM    ", "green")
                     + self._paint(self._bar(self.steps, self.max_steps), "green")
                     + self._paint(f"  {self.steps}/{self.max_steps or '—'}", "white")
                     + self._paint(f"  · physics {self.physics_steps}", "teal"))
        api_spinner = f"{spinner} " if "wait" in self.api_status.lower() else "• "
        task_color = ("green" if self.task_status == "SUCCESS" else
                      "red" if self.task_status == "TASK FAILURE" else "orange")
        api_color = "red" if any(word in self.api_status.lower() for word in
                                 ("error", "failed", "timeout", "stopping")) else "blue"
        left_width = max(12, (width - gap) // 2)
        right_width = max(12, width - gap - left_width)
        physical = [
            self._field("TASK", self.task_status, left_width - 4, "gray", task_color),
            self._field("SOCKET", f"{'● ENGAGED' if self.socket_engaged else '○ NOT ENGAGED'} · ever {'yes' if self.engaged_seen else 'no'}",
                        left_width - 4, "gray", "green" if self.socket_engaged else "yellow"),
            self._field("COST", self.estimated_cost, left_width - 4, "gray", "green"),
            self._field("API", f"{api_spinner}{self.api_status}", left_width - 4, "gray", api_color),
            self._field("FORCE", self.force_status, left_width - 4, "gray", "magenta"),
        ]
        left = self._panel("RUN / SIM STATE", [calls, step_line, *physical,
                                               self._field("STATE", self.state_delta, left_width - 4, "gray", "teal")],
                           left_width, "teal", 8)
        approval_color = ("green" if "approv" in self.approval.lower() or "pass" in self.approval.lower()
                          else "red" if any(term in self.approval.lower() for term in ("reject", "denied", "abort"))
                          else "yellow")
        right = self._panel("LIVE DATA FLOW", [
            self._field("01 OBS IN", self.input_summary, right_width - 4, "cyan", "blue"),
            self._field("   WRENCH", self.force_status, right_width - 4, "gray", "magenta"),
            self._field("02 POLICY", self.current_action, right_width - 4, "purple", "magenta"),
            self._field("   TARGET", self.targets, right_width - 4, "gray", "yellow"),
            self._field("   INTENT", self.note, right_width - 4, "gray", "blue"),
            self._field("03 GATE", self.approval, right_width - 4, "orange", approval_color),
            self._field("04 SIM OUT", self.state_delta, right_width - 4, "green", "teal"),
        ], right_width, "purple", 8)
        rows = [*self._wrap_display(head, width), *self._wrap_display(subhead, width),
                *self._wrap_display(flow, width)]
        if width >= 80 and size.lines >= 18:
            panel_rows = max(len(left), len(right))
            for index in range(panel_rows):
                left_row = left[index] if index < len(left) else " " * left_width
                right_row = right[index] if index < len(right) else " " * right_width
                rows.append(left_row + " " * gap + right_row)
        else:
            rows.extend(left)
            rows.extend(right)
        rows.append(self._paint("RECENT EVENTS", "magenta"))
        max_rows = max(1, size.lines - 1)
        event_capacity = max(0, max_rows - len(rows))
        recent = self.events[-event_capacity:] if event_capacity else []
        event_width = max(20, width - 4)
        for event in recent:
            prefix, separator, detail = event.partition(" · ")
            color = {"API": "blue", "ACTION": "yellow", "EXEC": "green",
                     "SIM": "teal", "OBS": "cyan", "INPUT": "purple",
                     "POLICY": "magenta", "RESULT": "green", "ERROR": "red"}.get(prefix)
            compact = textwrap.shorten(event, width=event_width, placeholder="…")
            if color and separator:
                compact_detail = textwrap.shorten(detail, width=max(1, event_width - len(prefix) - 3), placeholder="…")
                detail_color = {"ERROR": "red", "RESULT": "green", "SIM": "teal",
                                "ACTION": "yellow", "EXEC": "green", "INPUT": "blue",
                                "API": "blue", "OBS": "cyan", "POLICY": "magenta"}.get(prefix, "gray")
                compact = f"{self._paint(prefix, color)} · {self._paint(compact_detail, detail_color)}"
            rows.append(compact)
        rows = rows[:max_rows]
        output = "\033[?25l\033[2J\033[H" + "\n".join(rows) + "\033[J"
        self.stream.write(output)
        self.stream.flush()

    def _animate(self) -> None:
        while not self.stop_animation.wait(0.14):
            with self.lock:
                self._render()

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
            self.stop_animation.set()
            if self.animation is not None:
                self.animation.join(timeout=1)
            self.stream.write("\033[?25h\033[?1049l\033[0m")
            self.stream.flush()

    def finish(self, succeeded: bool) -> None:
        with self.lock:
            self.task_status = "SUCCESS" if succeeded else "TASK FAILURE"
            self._remember(f"RESULT · {self.task_status}")
            if self.tty:
                self._render()
