"""Safe command execution for the voice assistant.

Only a small, whitelisted set of commands can be executed by voice. Anything
that does not match a known intent is handed back to the LLM for a normal
conversational response.
"""
from __future__ import annotations

import logging
import platform
import re
import subprocess
import webbrowser
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class CommandRouter:
    """Parse transcribed text and execute whitelisted local commands."""

    TIME_PATTERNS = [
        r"\bwhat(?:'s| is)? the time\b",
        r"\bcurrent time\b",
        r"\btell me the time\b",
    ]
    DATE_PATTERNS = [
        r"\bwhat(?:'s| is)? (?:the )?date\b",
        r"\bwhat day is it\b",
        r"\btoday'?s date\b",
    ]
    OPEN_PATTERNS = [
        r"\bopen (.+?)\b",
        r"\blaunch (.+?)\b",
        r"\bstart (?:the )?(.+?)\b",
    ]
    SEARCH_PATTERNS = [
        r"\bsearch (?:for )?(.+?)\b",
        r"\bgoogle (.+?)\b",
        r"\blook up (.+?)\b",
    ]

    def __init__(self, enabled: bool):
        self.enabled = bool(enabled)

    def try_handle(self, text: str) -> Optional[str]:
        """Return a spoken response if a command was handled, else None."""
        if not self.enabled or not text:
            return None
        lowered = text.strip().lower().rstrip(".!?")

        for pattern in self.TIME_PATTERNS:
            if re.search(pattern, lowered):
                now = datetime.now().strftime("%I:%M %p")
                logger.info("Command: time -> %s", now)
                return f"It is {now}."

        for pattern in self.DATE_PATTERNS:
            if re.search(pattern, lowered):
                today = datetime.now().strftime("%A, %B %d, %Y")
                logger.info("Command: date -> %s", today)
                return f"Today is {today}."

        for pattern in self.OPEN_PATTERNS:
            match = re.search(pattern, lowered)
            if match:
                target = match.group(1).strip()
                return self._open_application(target)

        for pattern in self.SEARCH_PATTERNS:
            match = re.search(pattern, lowered)
            if match:
                query = match.group(1).strip()
                url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
                webbrowser.open(url)
                logger.info("Command: search -> %s", url)
                return f"Searching the web for {query}."

        return None

    def _open_application(self, target: str) -> str:
        """Open an application or URL by name using platform-appropriate tools."""
        target = target.strip().strip(".")
        # If it looks like a URL, open it directly.
        if "." in target and " " not in target:
            url = target if target.startswith("http") else f"https://{target}"
            webbrowser.open(url)
            return f"Opening {target}."

        system = platform.system()
        try:
            if system == "Linux":
                subprocess.Popen(["xdg-open", target]) if "." in target else subprocess.Popen([target])
            elif system == "Darwin":
                subprocess.Popen(["open", "-a", target])
            elif system == "Windows":
                subprocess.Popen(["start", "", target], shell=True)
            else:
                return f"Sorry, I cannot open {target} on this platform."
            logger.info("Command: open -> %s", target)
            return f"Opening {target}."
        except FileNotFoundError:
            return f"I could not find an application called {target}."
        except Exception as exc:
            logger.warning("Open command failed: %s", exc)
            return f"I was unable to open {target}."


_router: Optional[CommandRouter] = None


def get_router(config: dict) -> CommandRouter:
    global _router
    if _router is None:
        _router = CommandRouter(enabled=config["enable_command_execution"])
    return _router
