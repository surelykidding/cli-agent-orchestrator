"""Tests for Kimi CLI provider.

Covers initialization, status detection, message extraction, command building,
pattern matching, and cleanup — targeting >90% code coverage.
"""

import json
import os
import re
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.providers.kimi_cli import (
    ANSI_CODE_PATTERN,
    ERROR_PATTERN,
    IDLE_PROMPT_PATTERN,
    IDLE_PROMPT_TAIL_LINES,
    RESPONSE_BULLET_PATTERN,
    STATUS_BAR_PATTERN,
    THINKING_BULLET_RAW_PATTERN,
    USER_INPUT_BOX_END_PATTERN,
    USER_INPUT_BOX_START_PATTERN,
    WELCOME_BANNER_PATTERN,
    KimiCliProvider,
    ProviderError,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _isolate_kimi_home(monkeypatch, tmp_path):
    """Every test in this file must be hermetic: no real KIMI_CODE_HOME env,
    and Path.home() redirected to the test's tmp dir. The provider now always
    materializes a session-local Kimi home (config/credentials symlinks), so
    without this the unit tests would create directories under the real
    ~/.kimi-code."""
    monkeypatch.delenv("KIMI_CODE_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))


def _read_fixture(name: str) -> str:
    """Read a test fixture file."""
    return (FIXTURES_DIR / name).read_text()


# =============================================================================
# Initialization tests
# =============================================================================


class TestKimiCliProviderInitialization:
    """Tests for KimiCliProvider initialization flow."""

    @pytest.fixture(autouse=True)
    def _skip_startup_dialog(self):
        # initialize() polls the pane to dismiss kimi's upgrade-reminder dialog;
        # that path has its own tests. Stub it so command-send/timeout tests stay
        # fast and independent of the (mocked) get_history return type.
        with patch.object(KimiCliProvider, "_handle_startup_dialog", return_value=None):
            yield

    @pytest.mark.asyncio
    @patch("cli_agent_orchestrator.providers.kimi_cli.wait_until_status")
    @patch("cli_agent_orchestrator.providers.kimi_cli.wait_for_shell")
    @patch("cli_agent_orchestrator.providers.kimi_cli.get_backend")
    async def test_initialize_success(self, mock_tmux, mock_wait_shell, mock_wait_status):
        """Test successful initialization sends kimi command and reaches IDLE."""
        mock_wait_shell.return_value = True
        mock_wait_status.return_value = True

        provider = KimiCliProvider("term-1", "session-1", "window-1")
        result = await provider.initialize()

        assert result is True
        assert provider._initialized is True
        mock_tmux.return_value.send_keys.assert_called_once()
        mock_wait_shell.assert_called_once()
        mock_wait_status.assert_called_once()

    @pytest.mark.asyncio
    @patch("cli_agent_orchestrator.providers.kimi_cli.wait_for_shell")
    @patch("cli_agent_orchestrator.providers.kimi_cli.get_backend")
    async def test_initialize_shell_timeout(self, mock_tmux, mock_wait_shell):
        """Test shell init timeout raises TimeoutError."""
        mock_wait_shell.return_value = False

        provider = KimiCliProvider("term-1", "session-1", "window-1")
        with pytest.raises(TimeoutError, match="Shell initialization"):
            await provider.initialize()

    @pytest.mark.asyncio
    @patch("cli_agent_orchestrator.providers.kimi_cli.wait_until_status")
    @patch("cli_agent_orchestrator.providers.kimi_cli.wait_for_shell")
    @patch("cli_agent_orchestrator.providers.kimi_cli.get_backend")
    async def test_initialize_kimi_timeout(self, mock_tmux, mock_wait_shell, mock_wait_status):
        """Test Kimi CLI init timeout raises TimeoutError."""
        mock_wait_shell.return_value = True
        mock_wait_status.return_value = False

        provider = KimiCliProvider("term-1", "session-1", "window-1")
        with pytest.raises(TimeoutError, match="Kimi CLI initialization"):
            await provider.initialize()

    @pytest.mark.asyncio
    @patch("cli_agent_orchestrator.providers.kimi_cli.wait_until_status")
    @patch("cli_agent_orchestrator.providers.kimi_cli.wait_for_shell")
    @patch("cli_agent_orchestrator.providers.kimi_cli.get_backend")
    @patch("cli_agent_orchestrator.providers.kimi_cli.load_agent_profile")
    async def test_initialize_with_agent_profile(
        self, mock_load, mock_tmux, mock_wait_shell, mock_wait_status
    ):
        """Test initialization with agent profile creates temp files."""
        mock_wait_shell.return_value = True
        mock_wait_status.return_value = True
        mock_profile = MagicMock()
        mock_profile.model = None
        mock_profile.system_prompt = "You are a helpful assistant"
        mock_profile.mcpServers = None
        mock_profile.provider_init_timeout = None
        mock_load.return_value = mock_profile

        provider = KimiCliProvider("term-1", "session-1", "window-1", agent_profile="developer")
        result = await provider.initialize()
        assert result is True

        # Verify kimi command includes --agent-file
        call_args = mock_tmux.return_value.send_keys.call_args
        command = call_args[0][2]
        assert "--agent-file" in command
        assert "--yolo" in command

        # Cleanup temp files
        provider.cleanup()

    @patch("cli_agent_orchestrator.providers.kimi_cli.load_agent_profile")
    def test_initialize_with_invalid_profile(self, mock_load):
        """Test initialization with invalid agent profile raises ProviderError."""
        mock_load.side_effect = FileNotFoundError("Profile not found")

        provider = KimiCliProvider("term-1", "session-1", "window-1", agent_profile="nonexistent")
        with pytest.raises(ProviderError, match="Failed to load agent profile"):
            provider._build_kimi_command()

    @pytest.mark.asyncio
    @patch("cli_agent_orchestrator.providers.kimi_cli.wait_until_status")
    @patch("cli_agent_orchestrator.providers.kimi_cli.wait_for_shell")
    @patch("cli_agent_orchestrator.providers.kimi_cli.get_backend")
    @patch("cli_agent_orchestrator.providers.kimi_cli.load_agent_profile")
    async def test_initialize_with_mcp_servers(
        self, mock_load, mock_tmux, mock_wait_shell, mock_wait_status, monkeypatch, tmp_path
    ):
        """Test initialization with MCP servers wires a session-local KIMI_CODE_HOME."""
        mock_wait_shell.return_value = True
        mock_wait_status.return_value = True
        mock_profile = MagicMock()
        mock_profile.model = None
        mock_profile.system_prompt = None
        mock_profile.mcpServers = {
            "cao-mcp-server": {
                "command": "npx",
                "args": ["-y", "cao-mcp-server"],
            }
        }
        mock_profile.provider_init_timeout = None
        mock_load.return_value = mock_profile
        monkeypatch.delenv("KIMI_CODE_HOME", raising=False)
        user_home = tmp_path / "user-kimi-home"
        user_home.mkdir()

        provider = KimiCliProvider("term-1", "session-1", "window-1", agent_profile="developer")

        with patch(
            "cli_agent_orchestrator.providers.kimi_cli.Path.home",
            return_value=tmp_path,
        ):
            result = await provider.initialize()
        assert result is True

        call_args = mock_tmux.return_value.send_keys.call_args
        command = call_args[0][2]
        # New Kimi Code: MCP overlay via KIMI_CODE_HOME, never --mcp-config
        assert "--mcp-config" not in command
        assert "KIMI_CODE_HOME=" in command
        assert "KIMI_MCP_TOOL_TIMEOUT_MS=600000" in command
        # No --config flag either (legacy workaround, now removed)
        assert "--config" not in command
        provider.cleanup()

    @pytest.mark.asyncio
    @patch("cli_agent_orchestrator.providers.kimi_cli.wait_until_status")
    @patch("cli_agent_orchestrator.providers.kimi_cli.wait_for_shell")
    @patch("cli_agent_orchestrator.providers.kimi_cli.get_backend")
    async def test_initialize_sends_kimi_command(
        self, mock_tmux, mock_wait_shell, mock_wait_status
    ):
        """Test that initialize sends the kimi --yolo command without cd and with TERM override."""
        mock_wait_shell.return_value = True
        mock_wait_status.return_value = True

        provider = KimiCliProvider("term-1", "session-1", "window-1")
        await provider.initialize()

        call_args = mock_tmux.return_value.send_keys.call_args
        command = call_args[0][2]
        # CAO owns cwd: the command must never cd into the provider temp dir
        assert "cd " not in command
        assert "TERM=xterm-256color" in command
        assert "kimi --yolo" in command
        provider.cleanup()


class TestKimiCliFolderTrustDialog:
    """Kimi Code 0.34's "Trust this folder?" boot dialog must be dismissed
    headlessly, or init can never reach IDLE (observed in the real E2E: CAO
    worktrees are never in the user's trust list, and the dialog blocks the
    whole boot). Verified on 0.34: "Don't trust" makes kimi refuse to run in
    the folder (clean exit), so CAO accepts the default "Trust this folder"
    with a plain Enter — governed by the kimi_auto_trust_workspaces SERVER
    SETTING (default true), deliberately decoupled from the session's
    allowed_tools: CAO already treats Kimi workers as unrestricted
    (SOFT_ENFORCEMENT_PROVIDERS logs "treat this worker as unrestricted"),
    so a tool-restricted worker must still be able to boot. With the setting
    disabled, the dialog is left unanswered and init fails safe."""

    def _settings():
        return {
            "startup_prompt_handler_timeout": 20,
            "provider_init_timeout": 60,
            "kimi_auto_trust_workspaces": True,
        }

    @pytest.mark.asyncio
    @patch("cli_agent_orchestrator.providers.kimi_cli.get_server_settings", _settings)
    @patch("cli_agent_orchestrator.providers.kimi_cli.asyncio.sleep")
    @patch("cli_agent_orchestrator.providers.kimi_cli.time")
    @patch("cli_agent_orchestrator.providers.kimi_cli.get_backend")
    async def test_accepts_folder_trust(self, mock_get_backend, mock_time, mock_sleep):
        mock_backend = MagicMock()
        mock_get_backend.return_value = mock_backend
        mock_time.monotonic.side_effect = [
            0.0,  # outer_deadline = 60
            0.0,  # last_prompt_time = 0
            5.0,  # iter1: trust dialog detected
            5.0,  # last_prompt_time reset
            6.0,  # iter2: ready prompt → return
        ]
        mock_backend.get_history.side_effect = [
            "Trust this folder?\n"
            "↑↓ navigate · Enter select · Esc exit\n"
            "❯ Trust this folder\n"
            "  Don't trust\n"
            "  Exit Kimi Code. Asked again next launch.",
            "yolo  agent (kimi ●)  /tmp/x\ncontext: 4.0%",
        ]

        # allowed_tools is IRRELEVANT here: the gate is the server setting,
        # not the session's tool policy (a restricted kimi worker still boots).
        p = KimiCliProvider("t1", "sess", "win", allowed_tools=["execute_bash", "fs_read"])
        with (
            patch.object(p, "get_status", return_value=TerminalStatus.IDLE),
            patch("cli_agent_orchestrator.services.status_monitor.status_monitor") as mock_monitor,
        ):
            await p._handle_startup_dialog()

        # Accept the default "Trust this folder": a single Enter, nothing else.
        assert mock_backend.send_special_key.call_args_list == [call("sess", "win", "Enter")]
        assert mock_backend.send_keys.call_count == 0  # no other key sent
        mock_monitor.notify_input_sent.assert_called_once_with("t1")

    @pytest.mark.asyncio
    @patch("cli_agent_orchestrator.providers.kimi_cli.get_server_settings", _settings)
    @patch("cli_agent_orchestrator.providers.kimi_cli.asyncio.sleep")
    @patch("cli_agent_orchestrator.providers.kimi_cli.time")
    @patch("cli_agent_orchestrator.providers.kimi_cli.get_backend")
    async def test_auto_trust_disabled_setting_blocks(
        self, mock_get_backend, mock_time, mock_sleep
    ):
        """kimi_auto_trust_workspaces=false: no key is sent — init fails safe
        instead of auto-escalating workspace trust."""
        mock_backend = MagicMock()
        mock_get_backend.return_value = mock_backend
        mock_time.monotonic.side_effect = [
            0.0,  # outer_deadline = 60
            0.0,  # last_prompt_time = 0
            5.0,  # iter1: trust dialog detected → setting off → return
        ]
        mock_backend.get_history.return_value = (
            "Trust this folder?\n"
            "↑↓ navigate · Enter select · Esc exit\n"
            "❯ Trust this folder\n"
            "  Don't trust\n"
            "  Exit Kimi Code. Asked again next launch."
        )

        def _settings_disabled():
            return {
                "startup_prompt_handler_timeout": 20,
                "provider_init_timeout": 60,
                "kimi_auto_trust_workspaces": False,
            }

        p = KimiCliProvider("t1", "sess", "win")
        with (
            patch(
                "cli_agent_orchestrator.providers.kimi_cli.get_server_settings",
                _settings_disabled,
            ),
            patch("cli_agent_orchestrator.services.status_monitor.status_monitor") as mock_monitor,
        ):
            await p._handle_startup_dialog()

        assert mock_backend.send_special_key.call_count == 0
        assert mock_backend.send_keys.call_count == 0
        mock_monitor.notify_input_sent.assert_not_called()


# =============================================================================
# Status detection tests
# =============================================================================


class TestKimiCliProviderStatusDetection:
    """Tests for KimiCliProvider.get_status()."""

    def test_get_status_idle(self):
        """Test IDLE detection from fresh startup output."""
        provider = KimiCliProvider("term-1", "session-1", "window-1")
        assert provider.get_status(_read_fixture("kimi_cli_idle_output.txt")) == TerminalStatus.IDLE

    def test_get_status_idle_no_thinking(self):
        """Test IDLE detection with ✨ prompt (no-thinking mode)."""
        output = (
            "Welcome to Kimi Code CLI!\n"
            "user@my-app✨\n"
            "\n\n"
            "23:14  yolo  agent (kimi-for-coding)  ctrl-x: toggle mode  context: 0.0%"
        )
        provider = KimiCliProvider("term-1", "session-1", "window-1")
        assert provider.get_status(output) == TerminalStatus.IDLE

    def test_get_status_completed(self):
        """Test COMPLETED detection when response is present with prompt."""
        provider = KimiCliProvider("term-1", "session-1", "window-1")
        assert (
            provider.get_status(_read_fixture("kimi_cli_completed_output.txt"))
            == TerminalStatus.COMPLETED
        )

    def test_get_status_completed_complex(self):
        """Test COMPLETED detection with multi-line code response."""
        provider = KimiCliProvider("term-1", "session-1", "window-1")
        assert (
            provider.get_status(_read_fixture("kimi_cli_complex_response.txt"))
            == TerminalStatus.COMPLETED
        )

    def test_get_status_processing(self):
        """Test PROCESSING detection when no prompt at bottom."""
        provider = KimiCliProvider("term-1", "session-1", "window-1")
        assert (
            provider.get_status(_read_fixture("kimi_cli_processing_output.txt"))
            == TerminalStatus.PROCESSING
        )

    def test_get_status_unknown_empty(self):
        """Empty output -> UNKNOWN (native=None always falls through, no guess)."""
        provider = KimiCliProvider("term-1", "session-1", "window-1")
        assert provider.get_status("") == TerminalStatus.UNKNOWN

    def test_get_status_unknown_none(self):
        """Test UNKNOWN on None output."""
        provider = KimiCliProvider("term-1", "session-1", "window-1")
        assert provider.get_status(None) == TerminalStatus.UNKNOWN

    def test_get_status_error_pattern(self):
        """Test ERROR detection from error output fixture."""
        provider = KimiCliProvider("term-1", "session-1", "window-1")
        assert (
            provider.get_status(_read_fixture("kimi_cli_error_output.txt")) == TerminalStatus.ERROR
        )

    def test_get_status_idle_with_ansi_codes(self):
        """Test IDLE detection with ANSI escape codes in output."""
        # Simulate raw ANSI output: bold prompt with color codes
        output = (
            "\x1b[38;5;33mWelcome to Kimi Code CLI!\x1b[0m\n"
            "\x1b[1muser@my-app💫\x1b[0m\n"
            "\n\n"
            "23:14  yolo  agent (kimi-for-coding, thinking)  ctrl-x: toggle mode  context: 0.0%"
        )
        provider = KimiCliProvider("term-1", "session-1", "window-1")
        assert provider.get_status(output) == TerminalStatus.IDLE

    def test_get_status_idle_tall_terminal(self):
        """Test IDLE detection in tall terminals (46+ rows) where prompt is far from bottom.

        In a 46-row terminal, the welcome banner takes ~12 lines, the prompt is at
        line ~14, and there are ~32 empty padding lines before the status bar. The
        IDLE_PROMPT_TAIL_LINES must be large enough to reach the prompt.
        """
        # Simulate a 46-row terminal: welcome banner + prompt + 32 empty lines + status bar
        output = (
            "╭───────────────────────────────────╮\n"
            "│ Welcome to Kimi Code CLI!          │\n"
            "│ Send /help for help information.   │\n"
            "╰───────────────────────────────────╯\n"
            "user@project💫\n"
            + "\n" * 32  # 32 empty padding lines (typical for 46-row terminal)
            + "00:05  yolo  agent (kimi-for-coding, thinking)  ctrl-x: toggle mode  context: 0.0%\n"
        )
        provider = KimiCliProvider("term-1", "session-1", "window-1")
        assert provider.get_status(output) == TerminalStatus.IDLE

    def test_get_status_processing_streaming(self):
        """Test PROCESSING when response is mid-stream (no prompt, no error)."""
        output = (
            "╭──────────────────╮\n"
            "│ write a function  │\n"
            "╰──────────────────╯\n"
            "• Here's the function:\n"
            "\n"
            "def foo():\n"
            "    pass\n"
        )
        provider = KimiCliProvider("term-1", "session-1", "window-1")
        assert provider.get_status(output) == TerminalStatus.PROCESSING

    def test_get_status_completed_long_response_no_bullets(self):
        """Test COMPLETED for long structured responses without • bullet markers.

        Kimi doesn't always use • bullets — report templates, tables, numbered lists
        produce structured output with no bullets at all. The latching flag must detect
        the user input box during PROCESSING and remember it for COMPLETED detection.
        """
        provider = KimiCliProvider("term-1", "session-1", "window-1")

        # Step 1: During PROCESSING, the user input box is visible
        processing_output = (
            "╭──────────────────╮\n"
            "│ create a report    │\n"
            "╰──────────────────╯\n"
            "  Data Analysis Report Template\n"
            "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "  1. Summary section...\n"
        )
        assert provider.get_status(processing_output) == TerminalStatus.PROCESSING
        # Flag should now be latched
        assert provider._has_received_input is True

        # Step 2: After completion, the user input box has scrolled out.
        # Output now shows only the tail end of the response + idle prompt.
        completed_output = (
            "  Appendix C: Code Reference\n"
            "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "  [Reference to analysis code]\n"
            "user@project💫\n"
            "\n\n"
            "19:12  yolo  agent (kimi-for-coding, thinking)  ctrl-x: toggle mode  context: 2.9%"
        )
        assert provider.get_status(completed_output) == TerminalStatus.COMPLETED

    def test_get_status_latching_persists_after_scrollout(self):
        """Test that _has_received_input flag persists after user input box scrolls out."""
        provider = KimiCliProvider("term-1", "session-1", "window-1")

        # Simulate user input box detected during PROCESSING
        provider._has_received_input = True

        # Now output has idle prompt but NO user input box (scrolled out)
        output = (
            "  some response content\n"
            "user@project💫\n"
            "\n\n"
            "23:14  yolo  agent (kimi-for-coding, thinking)  ctrl-x: toggle mode  context: 1.0%"
        )
        assert provider.get_status(output) == TerminalStatus.COMPLETED

    def test_get_status_idle_before_any_input(self):
        """Test IDLE when no user input has been received yet (fresh startup)."""
        provider = KimiCliProvider("term-1", "session-1", "window-1")
        assert provider._has_received_input is False

        output = (
            "Welcome to Kimi Code CLI!\n"
            "user@project💫\n"
            "\n\n"
            "23:14  yolo  agent (kimi-for-coding, thinking)  ctrl-x: toggle mode  context: 0.0%"
        )
        assert provider.get_status(output) == TerminalStatus.IDLE
        assert provider._has_received_input is False

    def test_get_status_processing_latches_flag(self):
        """Test that user input box detected during PROCESSING latches the flag."""
        provider = KimiCliProvider("term-1", "session-1", "window-1")
        assert provider._has_received_input is False

        # PROCESSING output with user input box visible
        output = (
            "╭──────────────────╮\n"
            "│ hello               │\n"
            "╰──────────────────╯\n"
            "Response content streaming...\n"
        )
        status = provider.get_status(output)
        assert status == TerminalStatus.PROCESSING
        assert provider._has_received_input is True


# =============================================================================
# Message extraction tests
# =============================================================================


class TestKimiCliProviderMessageExtraction:
    """Tests for KimiCliProvider.extract_last_message_from_script()."""

    def test_extract_message_success(self):
        """Test successful message extraction from completed output."""
        provider = KimiCliProvider("term-1", "session-1", "window-1")
        output = _read_fixture("kimi_cli_completed_output.txt")
        result = provider.extract_last_message_from_script(output)

        assert len(result) > 0
        assert "greet" in result.lower() or "function" in result.lower()

    def test_extract_message_complex_response(self):
        """Test extraction of multi-line response with code block."""
        provider = KimiCliProvider("term-1", "session-1", "window-1")
        output = _read_fixture("kimi_cli_complex_response.txt")
        result = provider.extract_last_message_from_script(output)

        assert len(result) > 0
        assert "Calculator" in result or "calculator" in result

    def test_extract_message_no_input(self):
        """Test ValueError when no content at all (not even response text)."""
        provider = KimiCliProvider("term-1", "session-1", "window-1")
        # Only idle prompt, no response content
        output = "user@my-app💫\n\n\n23:14  yolo  agent (kimi-for-coding)  context: 0.0%"
        with pytest.raises(ValueError, match="No extractable content"):
            provider.extract_last_message_from_script(output)

    def test_extract_message_long_response_fallback(self):
        """Test fallback extraction when user input box scrolled out of capture.

        For long responses (>200 lines), the user input box is not visible in the
        tmux capture. The fallback extracts everything before the idle prompt.
        """
        provider = KimiCliProvider("term-1", "session-1", "window-1")
        # Simulate long response where user input box has scrolled out
        output = (
            "  3. Statistical Analysis Results\n"
            "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "  Mean: 3.0, Median: 3.0, StdDev: 1.414\n"
            "  4. Conclusions\n"
            "  The data shows a normal distribution.\n"
            "user@my-app💫\n"
            "\n\n"
            "23:14  yolo  agent (kimi-for-coding, thinking)  ctrl-x: toggle mode  context: 2.9%"
        )
        result = provider.extract_last_message_from_script(output)
        assert "Statistical Analysis" in result
        assert "Conclusions" in result
        assert "normal distribution" in result
        # Status bar should be filtered
        assert "yolo" not in result

    def test_extract_message_empty_response(self):
        """Test ValueError on empty response after input box."""
        provider = KimiCliProvider("term-1", "session-1", "window-1")
        output = (
            "╭──────────────────╮\n"
            "│ test message      │\n"
            "╰──────────────────╯\n"
            "user@my-app💫\n"
        )
        with pytest.raises(ValueError, match="Empty Kimi CLI response"):
            provider.extract_last_message_from_script(output)

    def test_extract_message_filters_thinking(self):
        """Test that thinking bullets (gray ANSI) are filtered from output."""
        provider = KimiCliProvider("term-1", "session-1", "window-1")
        # Simulate raw output with thinking and response bullets
        output = (
            "╭──────────────────╮\n"
            "│ say hello          │\n"
            "╰──────────────────╯\n"
            "\x1b[38;5;244m•\x1b[39m \x1b[3m\x1b[38;5;244mThe user wants a greeting.\x1b[0m\n"
            "• Hello! \U0001f44b\n"
            "user@my-app💫\n"
        )
        result = provider.extract_last_message_from_script(output)

        assert "Hello!" in result
        # Thinking text should be filtered out
        assert "The user wants" not in result

    def test_extract_message_multiple_responses(self):
        """Test extraction picks content from last user input box."""
        provider = KimiCliProvider("term-1", "session-1", "window-1")
        output = (
            "╭──────────────────╮\n"
            "│ first question     │\n"
            "╰──────────────────╯\n"
            "• First answer\n"
            "user@my-app💫\n"
            "╭──────────────────╮\n"
            "│ second question    │\n"
            "╰──────────────────╯\n"
            "• Second answer\n"
            "user@my-app💫\n"
        )
        result = provider.extract_last_message_from_script(output)
        assert "Second answer" in result

    def test_extract_message_no_trailing_prompt(self):
        """Test extraction works when there's no trailing prompt."""
        provider = KimiCliProvider("term-1", "session-1", "window-1")
        output = (
            "╭──────────────────╮\n"
            "│ what is python?    │\n"
            "╰──────────────────╯\n"
            "• Python is a programming language.\n"
            "• It supports multiple paradigms.\n"
        )
        result = provider.extract_last_message_from_script(output)
        assert "Python" in result
        assert "paradigm" in result.lower()

    def test_extract_message_all_thinking_falls_back(self):
        """Test fallback when all lines are filtered as thinking."""
        provider = KimiCliProvider("term-1", "session-1", "window-1")
        # All bullets are thinking (gray ANSI) — should fall back to returning all content
        output = (
            "╭──────────────────╮\n"
            "│ analyze this       │\n"
            "╰──────────────────╯\n"
            "\x1b[38;5;244m• \x1b[39m\x1b[3m\x1b[38;5;244mLet me analyze the code.\x1b[0m\n"
            "\x1b[38;5;244m• \x1b[39m\x1b[3m\x1b[38;5;244mI see several patterns.\x1b[0m\n"
            "user@my-app💫\n"
        )
        result = provider.extract_last_message_from_script(output)
        # Should return the thinking content as fallback
        assert "analyze" in result.lower() or "pattern" in result.lower()

    def test_extract_message_with_status_bar_filtered(self):
        """Test that status bar lines are filtered from extracted content."""
        provider = KimiCliProvider("term-1", "session-1", "window-1")
        output = (
            "╭──────────────────╮\n"
            "│ hello               │\n"
            "╰──────────────────╯\n"
            "• Hi there!\n"
            "23:14  yolo  agent (kimi-for-coding, thinking)  ctrl-x: toggle mode\n"
            "user@my-app💫\n"
        )
        result = provider.extract_last_message_from_script(output)
        assert "Hi there!" in result
        assert "yolo" not in result
        assert "ctrl-x" not in result


# =============================================================================
# Command building tests
# =============================================================================


class TestKimiCliProviderBuildCommand:
    """Tests for KimiCliProvider._build_kimi_command()."""

    def test_build_command_no_profile(self):
        """Test command without agent profile: no cd, TERM override, kimi --yolo."""
        provider = KimiCliProvider("term-1", "session-1", "window-1")
        command = provider._build_kimi_command()
        # CAO owns cwd: the command must never cd into the provider temp dir
        assert "cd " not in command
        assert "TERM=xterm-256color" in command
        assert "kimi --yolo" in command
        assert "KIMI_MCP_TOOL_TIMEOUT_MS=600000" in command
        # The session-local Kimi home is ALWAYS enabled (no profile either),
        # so kimi never writes runtime state into the user's real home.
        assert "KIMI_CODE_HOME=" in command
        assert provider._temp_dir is not None
        provider.cleanup()

    @patch("cli_agent_orchestrator.providers.kimi_cli.load_agent_profile")
    def test_build_command_with_system_prompt(self, mock_load):
        """Test command with agent profile containing system prompt."""
        mock_profile = MagicMock()
        mock_profile.model = None
        mock_profile.system_prompt = "You are a developer"
        mock_profile.mcpServers = None
        mock_load.return_value = mock_profile

        provider = KimiCliProvider("term-1", "session-1", "window-1", agent_profile="dev")
        command = provider._build_kimi_command()

        assert "kimi" in command
        assert "--yolo" in command
        assert "--agent-file" in command
        # Temp directory should be created
        assert provider._temp_dir is not None

        # Cleanup
        provider.cleanup()

    @patch("cli_agent_orchestrator.providers.kimi_cli.load_agent_profile")
    def test_build_command_writes_agent_md(self, mock_load):
        """New-style Markdown agent file: agent.md with ${base_prompt}, no legacy YAML."""
        mock_profile = MagicMock()
        mock_profile.model = None
        mock_profile.system_prompt = "Custom system prompt"
        mock_profile.mcpServers = None
        mock_load.return_value = mock_profile

        provider = KimiCliProvider("term-1", "session-1", "window-1", agent_profile="dev")
        command = provider._build_kimi_command()

        # agent.md exists; legacy indirection files are gone
        assert provider._temp_dir is not None
        agent_md = os.path.join(provider._temp_dir, "agent.md")
        assert os.path.exists(agent_md)
        assert not os.path.exists(os.path.join(provider._temp_dir, "agent.yaml"))
        assert not os.path.exists(os.path.join(provider._temp_dir, "system.md"))

        with open(agent_md, encoding="utf-8") as f:
            content = f.read()
        assert content.startswith("---\n")
        assert "name: cao-agent" in content
        assert "description: CAO-managed Kimi Code agent" in content
        # ${base_prompt} kept as a literal for Kimi's template expansion
        assert "${base_prompt}" in content
        assert "Custom system prompt" in content

        assert f"--agent-file {agent_md}" in command

        # Cleanup
        provider.cleanup()

    @patch("cli_agent_orchestrator.providers.kimi_cli.load_agent_profile")
    def test_build_command_with_mcp_config(self, mock_load, monkeypatch, tmp_path):
        """MCP servers land in the session-local kimi-home/mcp.json, not --mcp-config."""
        user_home = tmp_path / "user-kimi-home"
        user_home.mkdir()
        monkeypatch.setenv("KIMI_CODE_HOME", str(user_home))

        mock_profile = MagicMock()
        mock_profile.model = None
        mock_profile.system_prompt = None
        mock_profile.mcpServers = {"test-server": {"command": "npx", "args": ["test"]}}
        mock_load.return_value = mock_profile

        provider = KimiCliProvider("term-1", "session-1", "window-1", agent_profile="dev")
        command = provider._build_kimi_command()

        assert "--mcp-config" not in command
        assert "KIMI_CODE_HOME=" in command
        assert "KIMI_MCP_TOOL_TIMEOUT_MS=600000" in command

        session_home = os.path.join(provider._temp_dir, "kimi-home")
        with open(os.path.join(session_home, "mcp.json"), encoding="utf-8") as f:
            config = json.load(f)
        # CAO_TERMINAL_ID is injected into the MCP server env
        env = config["mcpServers"]["test-server"]["env"]
        assert env["CAO_TERMINAL_ID"] == "term-1"
        # Source home untouched
        assert not (user_home / "mcp.json").exists()
        provider.cleanup()

    @patch("cli_agent_orchestrator.providers.kimi_cli.load_agent_profile")
    def test_build_command_resolves_bundled_mcp_command(self, mock_load, monkeypatch, tmp_path):
        """The bare cao-mcp-server command is resolved to a PATH-independent
        invocation in the session mcp.json (wiring guard: a refactor that
        drops the resolve_mcp_server_config call must fail this test)."""
        user_home = tmp_path / "user-kimi-home"
        user_home.mkdir()
        monkeypatch.setenv("KIMI_CODE_HOME", str(user_home))

        mock_profile = MagicMock()
        mock_profile.model = None
        mock_profile.system_prompt = None
        mock_profile.mcpServers = {
            "cao-mcp-server": {"type": "stdio", "command": "cao-mcp-server", "args": []}
        }
        mock_load.return_value = mock_profile

        provider = KimiCliProvider("term-1", "session-1", "window-1", agent_profile="dev")
        MOD = "cli_agent_orchestrator.utils.mcp_resolution"
        with (
            patch(f"{MOD}._sibling_script", return_value="/venv/bin/cao-mcp-server"),
            patch(f"{MOD}.shutil.which", return_value=None),
        ):
            provider._build_kimi_command()

        session_home = os.path.join(provider._temp_dir, "kimi-home")
        with open(os.path.join(session_home, "mcp.json"), encoding="utf-8") as f:
            config = json.load(f)
        assert config["mcpServers"]["cao-mcp-server"]["command"] == "/venv/bin/cao-mcp-server"
        provider.cleanup()

    @patch("cli_agent_orchestrator.providers.kimi_cli.load_agent_profile")
    def test_build_command_with_pydantic_mcp_config(self, mock_load, monkeypatch, tmp_path):
        """Test MCP servers as Pydantic model objects land in session mcp.json."""
        user_home = tmp_path / "user-kimi-home"
        user_home.mkdir()
        monkeypatch.setenv("KIMI_CODE_HOME", str(user_home))

        mock_server = MagicMock()
        mock_server.model_dump.return_value = {"command": "node", "args": ["server.js"]}
        # Not a dict, triggers model_dump branch
        type(mock_server).__instancecheck__ = lambda cls, inst: False

        mock_profile = MagicMock()
        mock_profile.model = None
        mock_profile.system_prompt = None
        mock_profile.mcpServers = {"my-server": mock_server}
        mock_load.return_value = mock_profile

        provider = KimiCliProvider("term-1", "session-1", "window-1", agent_profile="dev")
        command = provider._build_kimi_command()

        assert "--mcp-config" not in command
        session_home = os.path.join(provider._temp_dir, "kimi-home")
        with open(os.path.join(session_home, "mcp.json"), encoding="utf-8") as f:
            config = json.load(f)
        assert config["mcpServers"]["my-server"]["command"] == "node"
        assert config["mcpServers"]["my-server"]["env"]["CAO_TERMINAL_ID"] == "term-1"
        provider.cleanup()

    @patch("cli_agent_orchestrator.providers.kimi_cli.load_agent_profile")
    def test_build_command_mcp_preserves_existing_env(self, mock_load, monkeypatch, tmp_path):
        """Test that CAO_TERMINAL_ID injection preserves existing env vars."""
        user_home = tmp_path / "user-kimi-home"
        user_home.mkdir()
        monkeypatch.setenv("KIMI_CODE_HOME", str(user_home))

        mock_profile = MagicMock()
        mock_profile.model = None
        mock_profile.system_prompt = None
        mock_profile.mcpServers = {
            "test-server": {
                "command": "npx",
                "args": ["test"],
                "env": {"MY_VAR": "my_value"},
            }
        }
        mock_load.return_value = mock_profile

        provider = KimiCliProvider("abc123", "session-1", "window-1", agent_profile="dev")
        provider._build_kimi_command()

        session_home = os.path.join(provider._temp_dir, "kimi-home")
        with open(os.path.join(session_home, "mcp.json"), encoding="utf-8") as f:
            config = json.load(f)
        env = config["mcpServers"]["test-server"]["env"]
        assert env["MY_VAR"] == "my_value"
        assert env["CAO_TERMINAL_ID"] == "abc123"
        provider.cleanup()

    @patch("cli_agent_orchestrator.providers.kimi_cli.load_agent_profile")
    def test_build_command_mcp_does_not_override_existing_terminal_id(
        self, mock_load, monkeypatch, tmp_path
    ):
        """Test that existing CAO_TERMINAL_ID in env is not overwritten."""
        user_home = tmp_path / "user-kimi-home"
        user_home.mkdir()
        monkeypatch.setenv("KIMI_CODE_HOME", str(user_home))

        mock_profile = MagicMock()
        mock_profile.model = None
        mock_profile.system_prompt = None
        mock_profile.mcpServers = {
            "test-server": {
                "command": "npx",
                "args": ["test"],
                "env": {"CAO_TERMINAL_ID": "existing-id"},
            }
        }
        mock_load.return_value = mock_profile

        provider = KimiCliProvider("new-id", "session-1", "window-1", agent_profile="dev")
        provider._build_kimi_command()

        session_home = os.path.join(provider._temp_dir, "kimi-home")
        with open(os.path.join(session_home, "mcp.json"), encoding="utf-8") as f:
            config = json.load(f)
        # Should keep the existing value, not override
        assert config["mcpServers"]["test-server"]["env"]["CAO_TERMINAL_ID"] == "existing-id"
        provider.cleanup()

    @patch("cli_agent_orchestrator.providers.kimi_cli.load_agent_profile")
    def test_build_command_merges_existing_user_mcp_json(self, mock_load, monkeypatch, tmp_path):
        """The user's own mcp.json is preserved and merged, source untouched."""
        user_home = tmp_path / "user-kimi-home"
        user_home.mkdir()
        (user_home / "mcp.json").write_text(
            json.dumps(
                {
                    "mcpServers": {"existing-server": {"command": "npx", "args": ["-y", "srv"]}},
                    "customTopLevel": True,
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("KIMI_CODE_HOME", str(user_home))

        mock_profile = MagicMock()
        mock_profile.model = None
        mock_profile.system_prompt = None
        mock_profile.mcpServers = {
            "cao-mcp-server": {"type": "stdio", "command": "cao-mcp-server", "args": []}
        }
        mock_load.return_value = mock_profile

        provider = KimiCliProvider("term-1", "session-1", "window-1", agent_profile="dev")
        MOD = "cli_agent_orchestrator.utils.mcp_resolution"
        with (
            patch(f"{MOD}._sibling_script", return_value="/venv/bin/cao-mcp-server"),
            patch(f"{MOD}.shutil.which", return_value=None),
        ):
            provider._build_kimi_command()

        session_home = os.path.join(provider._temp_dir, "kimi-home")
        with open(os.path.join(session_home, "mcp.json"), encoding="utf-8") as f:
            mcp = json.load(f)
        # Both the user's server and the CAO profile's server are present
        assert set(mcp["mcpServers"]) == {"existing-server", "cao-mcp-server"}
        # Non-mcpServers top-level keys are preserved
        assert mcp["customTopLevel"] is True
        # Source user mcp.json is completely unchanged
        source = json.loads((user_home / "mcp.json").read_text(encoding="utf-8"))
        assert source == {
            "mcpServers": {"existing-server": {"command": "npx", "args": ["-y", "srv"]}},
            "customTopLevel": True,
        }
        provider.cleanup()

    @patch("cli_agent_orchestrator.providers.kimi_cli.load_agent_profile")
    def test_build_command_profile_mcp_overrides_user_same_name(
        self, mock_load, monkeypatch, tmp_path
    ):
        """On a name collision the CAO session profile wins over the user default."""
        user_home = tmp_path / "user-kimi-home"
        user_home.mkdir()
        (user_home / "mcp.json").write_text(
            json.dumps({"mcpServers": {"cao-mcp-server": {"command": "user-copy"}}}),
            encoding="utf-8",
        )
        monkeypatch.setenv("KIMI_CODE_HOME", str(user_home))

        mock_profile = MagicMock()
        mock_profile.model = None
        mock_profile.system_prompt = None
        mock_profile.mcpServers = {
            "cao-mcp-server": {"type": "stdio", "command": "cao-mcp-server", "args": []}
        }
        mock_load.return_value = mock_profile

        provider = KimiCliProvider("term-1", "session-1", "window-1", agent_profile="dev")
        MOD = "cli_agent_orchestrator.utils.mcp_resolution"
        with (
            patch(f"{MOD}._sibling_script", return_value="/venv/bin/cao-mcp-server"),
            patch(f"{MOD}.shutil.which", return_value=None),
        ):
            provider._build_kimi_command()

        session_home = os.path.join(provider._temp_dir, "kimi-home")
        with open(os.path.join(session_home, "mcp.json"), encoding="utf-8") as f:
            mcp = json.load(f)
        assert mcp["mcpServers"]["cao-mcp-server"]["command"] == "/venv/bin/cao-mcp-server"
        # Source user file unchanged
        assert "user-copy" in (user_home / "mcp.json").read_text(encoding="utf-8")
        provider.cleanup()

    @patch("cli_agent_orchestrator.providers.kimi_cli.load_agent_profile")
    def test_build_command_invalid_user_mcp_json_raises(self, mock_load, monkeypatch, tmp_path):
        """An unparseable user mcp.json must raise ProviderError, not be silently replaced."""
        user_home = tmp_path / "user-kimi-home"
        user_home.mkdir()
        (user_home / "mcp.json").write_text("{not valid json", encoding="utf-8")
        monkeypatch.setenv("KIMI_CODE_HOME", str(user_home))

        mock_profile = MagicMock()
        mock_profile.model = None
        mock_profile.system_prompt = None
        mock_profile.mcpServers = {
            "cao-mcp-server": {"type": "stdio", "command": "cao-mcp-server", "args": []}
        }
        mock_load.return_value = mock_profile

        provider = KimiCliProvider("term-1", "session-1", "window-1", agent_profile="dev")
        with pytest.raises(ProviderError, match="could not be parsed"):
            provider._build_kimi_command()
        provider.cleanup()

    @patch("cli_agent_orchestrator.providers.kimi_cli.load_agent_profile")
    def test_build_command_no_mcp_still_has_session_home(self, mock_load):
        """A profile without MCP servers still gets the session-local Kimi
        home (so kimi never writes workspace-trust/sessions/workspaces.json
        into the user's real home); no mcp.json is written when neither the
        profile nor the user has one."""
        mock_profile = MagicMock()
        mock_profile.model = None
        mock_profile.system_prompt = "You are helpful"
        mock_profile.mcpServers = None
        mock_load.return_value = mock_profile

        provider = KimiCliProvider("term-1", "session-1", "window-1", agent_profile="dev")
        command = provider._build_kimi_command()

        assert "--agent-file" in command
        assert "KIMI_CODE_HOME=" in command
        assert "KIMI_MCP_TOOL_TIMEOUT_MS=600000" in command
        session_home = os.path.join(provider._temp_dir, "kimi-home")
        assert not os.path.exists(os.path.join(session_home, "mcp.json"))
        provider.cleanup()

    @patch("cli_agent_orchestrator.providers.kimi_cli.load_agent_profile")
    def test_build_command_no_profile_mcp_carries_user_mcp_json(
        self, mock_load, monkeypatch, tmp_path
    ):
        """No profile MCP servers + a user mcp.json: the user's MCP config is
        carried into the session home UNMODIFIED (no CAO overlay, no
        CAO_TERMINAL_ID injection) and the source file is untouched."""
        user_home = tmp_path / "user-kimi-home"
        user_home.mkdir()
        (user_home / "mcp.json").write_text(
            json.dumps(
                {"mcpServers": {"existing-server": {"command": "npx", "args": ["-y", "srv"]}}}
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("KIMI_CODE_HOME", str(user_home))

        mock_profile = MagicMock()
        mock_profile.model = None
        mock_profile.system_prompt = None
        mock_profile.mcpServers = None
        mock_load.return_value = mock_profile

        provider = KimiCliProvider("term-1", "session-1", "window-1", agent_profile="dev")
        provider._build_kimi_command()

        session_home = os.path.join(provider._temp_dir, "kimi-home")
        with open(os.path.join(session_home, "mcp.json"), encoding="utf-8") as f:
            mcp = json.load(f)
        assert mcp["mcpServers"]["existing-server"]["command"] == "npx"
        assert "CAO_TERMINAL_ID" not in mcp["mcpServers"]["existing-server"].get("env", {})
        assert "CAO_TERMINAL_ID" not in json.dumps(mcp)
        # Source unchanged.
        assert "npx" in (user_home / "mcp.json").read_text(encoding="utf-8")
        provider.cleanup()

    def test_build_command_sets_timeout_env_var(self):
        """Timeout is process-local: KIMI_MCP_TOOL_TIMEOUT_MS is always set."""
        provider = KimiCliProvider("term-1", "session-1", "window-1")
        command = provider._build_kimi_command()
        assert "KIMI_MCP_TOOL_TIMEOUT_MS=600000" in command
        provider.cleanup()

    def test_build_command_does_not_touch_user_configs(self, monkeypatch, tmp_path):
        """No user config file is modified: neither legacy ~/.kimi nor ~/.kimi-code."""
        user_home = tmp_path / ".kimi-code"
        user_home.mkdir()
        legacy_kimi = tmp_path / ".kimi"
        legacy_kimi.mkdir()
        user_config = user_home / "config.toml"
        user_config.write_text("[mcp.client]\ntool_call_timeout_ms = 60000\n", encoding="utf-8")
        legacy_config = legacy_kimi / "config.toml"
        legacy_config.write_text("[mcp.client]\ntool_call_timeout_ms = 60000\n", encoding="utf-8")

        provider = KimiCliProvider("term-1", "session-1", "window-1")
        command = provider._build_kimi_command()

        assert "KIMI_MCP_TOOL_TIMEOUT_MS=600000" in command
        assert "tool_call_timeout_ms = 60000" in user_config.read_text(encoding="utf-8")
        assert "tool_call_timeout_ms = 60000" in legacy_config.read_text(encoding="utf-8")
        provider.cleanup()

    @patch("cli_agent_orchestrator.providers.kimi_cli.load_agent_profile")
    def test_build_command_profile_no_system_prompt(self, mock_load):
        """Test command with profile that has no system prompt (no agent file, but temp dir exists)."""
        mock_profile = MagicMock()
        mock_profile.model = None
        mock_profile.system_prompt = None
        mock_profile.mcpServers = None
        mock_load.return_value = mock_profile

        provider = KimiCliProvider("term-1", "session-1", "window-1", agent_profile="dev")
        command = provider._build_kimi_command()

        assert "kimi --yolo" in command
        assert "--agent-file" not in command
        assert provider._temp_dir is not None
        provider.cleanup()

    @patch("cli_agent_orchestrator.providers.kimi_cli.load_agent_profile")
    def test_build_command_profile_empty_system_prompt(self, mock_load):
        """Test command with profile that has empty string system prompt."""
        mock_profile = MagicMock()
        mock_profile.model = None
        mock_profile.system_prompt = ""
        mock_profile.mcpServers = None
        mock_load.return_value = mock_profile

        provider = KimiCliProvider("term-1", "session-1", "window-1", agent_profile="dev")
        command = provider._build_kimi_command()

        assert "kimi --yolo" in command
        assert "--agent-file" not in command
        assert provider._temp_dir is not None
        provider.cleanup()


class TestKimiSessionHomeIsolation:
    """Secret/state lifecycle of the session-local Kimi home:

    - credentials/ is SYMLINKED to the user's real store (token refresh must
      write back to the user's own home; a /tmp copy would linger whenever
      session teardown skips provider.cleanup()).
    - plugins/ and skills/ are REAL (isolated) directories: mutable metadata
      (installed.json) is copied, only read-only payload entries are
      symlinked — a session can never write back to the user's store.
    - bin/ (bundled binaries, read-only) is symlinked; AGENTS.md (small,
      read-only guidance) is copied.
    """

    def _provider_with_home(self, user_home):
        with patch("cli_agent_orchestrator.providers.kimi_cli.load_agent_profile") as mock_load:
            mock_profile = MagicMock()
            mock_profile.model = None
            mock_profile.system_prompt = None
            mock_profile.mcpServers = {
                "cao-mcp-server": {"type": "stdio", "command": "cao-mcp-server", "args": []}
            }
            mock_load.return_value = mock_profile
            provider = KimiCliProvider("term-1", "session-1", "window-1", agent_profile="dev")
            MOD = "cli_agent_orchestrator.utils.mcp_resolution"
            with (
                patch(f"{MOD}._sibling_script", return_value="/venv/bin/cao-mcp-server"),
                patch(f"{MOD}.shutil.which", return_value=None),
            ):
                provider._build_kimi_command()
            return provider

    def test_credentials_symlinked_not_copied(self, monkeypatch, tmp_path):
        """The /tmp session home must never hold a credential FILE copy."""
        user_home = tmp_path / "user-kimi-home"
        creds = user_home / "credentials"
        creds.mkdir(parents=True)
        (creds / "user.json").write_text('{"token": "sensitive"}', encoding="utf-8")
        monkeypatch.setenv("KIMI_CODE_HOME", str(user_home))

        provider = self._provider_with_home(user_home)
        link = os.path.join(provider._temp_dir, "kimi-home", "credentials")
        # Symlink to the user's real store — not a copy.
        assert os.path.islink(link)
        assert os.readlink(link) == str(creds)
        assert os.path.realpath(link) == str(creds)
        # Read-through works (kimi token refresh writes in place in source).
        assert (Path(link) / "user.json").read_text(encoding="utf-8").startswith('{"token"')
        provider.cleanup()

    def test_credentials_symlink_failure_raises_provider_error(self, monkeypatch, tmp_path):
        """Fail CLOSED: if the credentials symlink cannot be created, raise
        ProviderError — never fall back to a copy that would leave a /tmp
        secret residue."""
        user_home = tmp_path / "user-kimi-home"
        creds = user_home / "credentials"
        creds.mkdir(parents=True)
        (creds / "user.json").write_text('{"token": "sensitive"}', encoding="utf-8")
        monkeypatch.setenv("KIMI_CODE_HOME", str(user_home))

        provider = KimiCliProvider("term-1", "session-1", "window-1", agent_profile="dev")
        real_symlink = os.symlink

        def _fail_credentials_symlink(src, dst):
            if str(dst).endswith("credentials"):
                raise OSError("no")
            return real_symlink(src, dst)

        with (
            patch("cli_agent_orchestrator.providers.kimi_cli.load_agent_profile") as mock_load,
            patch(
                "cli_agent_orchestrator.providers.kimi_cli.os.symlink",
                side_effect=_fail_credentials_symlink,
            ),
        ):
            mock_profile = MagicMock()
            mock_profile.model = None
            mock_profile.system_prompt = None
            mock_profile.mcpServers = {"cao-mcp-server": {"command": "cao-mcp-server", "args": []}}
            mock_load.return_value = mock_profile
            with pytest.raises(ProviderError, match="Cannot symlink credentials store"):
                provider._build_kimi_command()

        # No credentials copy was created in the session home.
        session_home = os.path.join(provider._temp_dir, "kimi-home")
        assert not os.path.lexists(os.path.join(session_home, "credentials"))
        provider.cleanup()

    def test_config_toml_symlinked_not_copied(self, monkeypatch, tmp_path):
        """config.toml may carry an inlined api_key, so it is SYMLINKED to the
        user's real file — no /tmp copy, no secret residue."""
        user_home = tmp_path / "user-kimi-home"
        user_home.mkdir()
        config = user_home / "config.toml"
        config.write_text(
            '[providers.fake]\napi_key = "SUPER_SECRET_TEST_VALUE"\n', encoding="utf-8"
        )
        monkeypatch.setenv("KIMI_CODE_HOME", str(user_home))

        provider = self._provider_with_home(user_home)
        session_config = os.path.join(provider._temp_dir, "kimi-home", "config.toml")
        assert os.path.islink(session_config)
        assert os.path.realpath(session_config) == str(config)
        provider.cleanup()

    def test_no_secret_in_session_regular_files(self, monkeypatch, tmp_path):
        """The session temp dir must not contain the api_key inside any
        REGULAR file (symlinks point at the user's home and are excluded).
        This is the merge gate for secret isolation."""
        secret = "SUPER_SECRET_TEST_VALUE"
        user_home = tmp_path / "user-kimi-home"
        user_home.mkdir()
        (user_home / "config.toml").write_text(
            f'[providers.fake]\napi_key = "{secret}"\n', encoding="utf-8"
        )
        creds = user_home / "credentials"
        creds.mkdir()
        (creds / "user.json").write_text(f'{{"token": "{secret}"}}', encoding="utf-8")
        (user_home / "tui.toml").write_text("theme = 'light'\n", encoding="utf-8")
        monkeypatch.setenv("KIMI_CODE_HOME", str(user_home))

        provider = self._provider_with_home(user_home)

        temp_dir = provider._temp_dir
        hits = []
        for root, _dirs, files in os.walk(temp_dir):
            for fname in files:
                path = os.path.join(root, fname)
                if os.path.islink(path):
                    continue  # symlink → user's own home, not a copy
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        if secret in f.read():
                            hits.append(path)
                except OSError:
                    pass
        assert hits == [], f"secret found in session regular files: {hits}"
        provider.cleanup()

    def test_plugins_metadata_copied_payload_symlinked(self, monkeypatch, tmp_path):
        """plugins/ is isolated: installed.json is a real copy; plugin payload
        directories are symlinks — session writes can never reach source."""
        user_home = tmp_path / "user-kimi-home"
        plugins = user_home / "plugins"
        (plugins / "my-plugin").mkdir(parents=True)
        (plugins / "installed.json").write_text('{"installed": ["my-plugin"]}', encoding="utf-8")
        monkeypatch.setenv("KIMI_CODE_HOME", str(user_home))

        provider = self._provider_with_home(user_home)
        session_plugins = os.path.join(provider._temp_dir, "kimi-home", "plugins")
        assert os.path.isdir(session_plugins)
        assert not os.path.islink(session_plugins)  # isolated real dir
        # Mutable registry: a real copy in the session home.
        registry = os.path.join(session_plugins, "installed.json")
        assert os.path.isfile(registry)
        assert not os.path.islink(registry)
        assert '{"installed": ["my-plugin"]}' in open(registry, encoding="utf-8").read()
        # Read-only payload: symlink.
        assert os.path.islink(os.path.join(session_plugins, "my-plugin"))
        # Source registry unchanged.
        assert '{"installed": ["my-plugin"]}' in (plugins / "installed.json").read_text(
            encoding="utf-8"
        )
        provider.cleanup()

    def test_skills_isolated_like_plugins(self, monkeypatch, tmp_path):
        user_home = tmp_path / "user-kimi-home"
        skills = user_home / "skills"
        (skills / "reviewer").mkdir(parents=True)
        (skills / "reviewer" / "SKILL.md").write_text("# reviewer", encoding="utf-8")
        monkeypatch.setenv("KIMI_CODE_HOME", str(user_home))

        provider = self._provider_with_home(user_home)
        session_skills = os.path.join(provider._temp_dir, "kimi-home", "skills")
        assert os.path.isdir(session_skills)
        assert not os.path.islink(session_skills)
        assert os.path.islink(os.path.join(session_skills, "reviewer"))
        assert os.path.exists(os.path.join(session_skills, "reviewer", "SKILL.md"))
        provider.cleanup()

    def test_bin_symlinked_agents_md_copied(self, monkeypatch, tmp_path):
        user_home = tmp_path / "user-kimi-home"
        (user_home / "bin").mkdir(parents=True)
        (user_home / "bin" / "kimi").write_text("#!/bin/sh\n", encoding="utf-8")
        (user_home / "AGENTS.md").write_text("# global guidance", encoding="utf-8")
        monkeypatch.setenv("KIMI_CODE_HOME", str(user_home))

        provider = self._provider_with_home(user_home)
        session_home = os.path.join(provider._temp_dir, "kimi-home")
        # bin: read-only bundled binaries → symlink.
        assert os.path.islink(os.path.join(session_home, "bin"))
        # AGENTS.md: small read-only guidance → copy (no write-back surface).
        agents_md = os.path.join(session_home, "AGENTS.md")
        assert os.path.isfile(agents_md)
        assert not os.path.islink(agents_md)
        assert (user_home / "AGENTS.md").read_text(encoding="utf-8") == "# global guidance"
        provider.cleanup()

    def test_credentials_dir_created_when_missing(self, monkeypatch, tmp_path):
        """B3: source credentials/ absent (e.g. this machine: auth lives in
        config.toml) — CAO creates it (0700) and symlinks unconditionally, so
        any credential kimi produces during the session (MCP OAuth, token
        flush) lands in the USER's home, never as fresh files under /tmp."""
        user_home = tmp_path / "user-kimi-home"
        user_home.mkdir()
        monkeypatch.setenv("KIMI_CODE_HOME", str(user_home))
        assert not (user_home / "credentials").exists()

        provider = self._provider_with_home(user_home)

        # Source store created with private perms; session link points at it.
        assert (user_home / "credentials").is_dir()
        assert oct(os.stat(user_home / "credentials").st_mode & 0o777) == "0o700"
        link = os.path.join(provider._temp_dir, "kimi-home", "credentials")
        assert os.path.islink(link)
        assert os.path.realpath(link) == str(user_home / "credentials")
        # A credential written through the session link lands in the user's
        # home (kimi refresh behavior), not in /tmp.
        (Path(link) / "mcp").mkdir()
        (Path(link) / "mcp" / "srv-1.json").write_text('{"token": "x"}', encoding="utf-8")
        assert (user_home / "credentials" / "mcp" / "srv-1.json").exists()
        assert not os.path.exists(os.path.join(provider._temp_dir, "srv-1.json"))
        provider.cleanup()

    def test_config_rename_replacement_residue_cleaned_up(self, monkeypatch, tmp_path):
        """B2: kimi persists config via atomicWrite (tmp + rename), and rename
        REPLACES the session symlink instead of writing through — verified
        against 0.34. When that happens the api_key exists as a plain /tmp
        file; the guarantee is that provider.cleanup() removes the whole temp
        dir (residue included), and the user's real config is untouched."""
        secret = "SUPER_SECRET_TEST_VALUE"
        user_home = tmp_path / "user-kimi-home"
        user_home.mkdir()
        (user_home / "config.toml").write_text(
            f'[providers.fake]\napi_key = "{secret}"\n', encoding="utf-8"
        )
        monkeypatch.setenv("KIMI_CODE_HOME", str(user_home))

        provider = self._provider_with_home(user_home)
        session_config = os.path.join(provider._temp_dir, "kimi-home", "config.toml")
        assert os.path.islink(session_config)

        # Simulate kimi's atomicWrite: rename a fresh regular file over the
        # session config path (the symlink is replaced, source untouched).
        replacement = tmp_path / "config.toml.new"
        replacement.write_text(f'[providers.fake]\napi_key = "{secret}"\n', encoding="utf-8")
        os.replace(replacement, session_config)
        assert not os.path.islink(session_config)
        assert secret in open(session_config, encoding="utf-8").read()
        # User's real config is untouched.
        assert (
            (user_home / "config.toml").read_text(encoding="utf-8").startswith("[providers.fake]")
        )

        # Cleanup removes the residue with the whole temp dir.
        temp_dir = provider._temp_dir
        provider.cleanup()
        assert not os.path.exists(temp_dir)

    def test_mcp_json_is_the_only_allowed_regular_secret_carrier(self, monkeypatch, tmp_path):
        """H1: profile MCP server env (expandable via resolve_env_vars from
        .env) can embed secrets, and mcp.json is a regular file in the
        session home. The secret scan must FIND it there (proving coverage),
        and cleanup must remove it with the temp dir."""
        secret = "SUPER_SECRET_TEST_VALUE"
        user_home = tmp_path / "user-kimi-home"
        user_home.mkdir()
        monkeypatch.setenv("KIMI_CODE_HOME", str(user_home))

        with patch("cli_agent_orchestrator.providers.kimi_cli.load_agent_profile") as mock_load:
            mock_profile = MagicMock()
            mock_profile.model = None
            mock_profile.system_prompt = None
            mock_profile.mcpServers = {
                "secret-server": {
                    "command": "npx",
                    "args": ["-y", "srv"],
                    "env": {"API_TOKEN": secret},
                }
            }
            mock_load.return_value = mock_profile
            provider = KimiCliProvider("term-1", "session-1", "window-1", agent_profile="dev")
            provider._build_kimi_command()

        temp_dir = provider._temp_dir
        hits = []
        for root, _dirs, files in os.walk(temp_dir):
            for fname in files:
                path = os.path.join(root, fname)
                if os.path.islink(path):
                    continue
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        if secret in f.read():
                            hits.append(path)
                except OSError:
                    pass
        # The ONLY regular file carrying the secret is the session mcp.json —
        # session-local, removed with the temp dir.
        assert hits == [os.path.join(temp_dir, "kimi-home", "mcp.json")], hits

        provider.cleanup()
        assert not os.path.exists(temp_dir)


# =============================================================================
# Misc / lifecycle tests
# =============================================================================


class TestKimiCliProviderModelFlag:
    """Tests that profile.model is forwarded to Kimi CLI via --model."""

    @patch("cli_agent_orchestrator.providers.kimi_cli.load_agent_profile")
    def test_build_command_appends_model_when_set(self, mock_load):
        mock_profile = MagicMock()
        mock_profile.model = "kimi-k2-turbo"
        mock_profile.system_prompt = None
        mock_profile.mcpServers = None
        mock_load.return_value = mock_profile

        provider = KimiCliProvider("term-1", "sess", "win", "agent")
        command = provider._build_kimi_command()

        assert "--model kimi-k2-turbo" in command

    @patch("cli_agent_orchestrator.providers.kimi_cli.load_agent_profile")
    def test_build_command_omits_model_when_unset(self, mock_load):
        mock_profile = MagicMock()
        mock_profile.model = None
        mock_profile.system_prompt = None
        mock_profile.mcpServers = None
        mock_load.return_value = mock_profile

        provider = KimiCliProvider("term-1", "sess", "win", "agent")
        command = provider._build_kimi_command()

        assert "--model" not in command

    @patch("cli_agent_orchestrator.providers.kimi_cli.load_agent_profile")
    def test_explicit_model_override_wins_over_profile_model(self, mock_load):
        mock_profile = MagicMock()
        mock_profile.model = "kimi-k2-turbo"
        mock_profile.system_prompt = None
        mock_profile.mcpServers = None
        mock_load.return_value = mock_profile

        provider = KimiCliProvider("term-1", "sess", "win", "agent", model="fable-5")
        command = provider._build_kimi_command()

        assert "--model fable-5" in command
        assert "--model kimi-k2-turbo" not in command

    def test_explicit_model_override_applies_with_no_agent_profile(self):
        """Regression: PR #501 review -- model resolution used to live
        entirely inside `if self._agent_profile is not None:`, so an
        override passed with agent_profile=None (unreachable through
        handoff/assign today, but inconsistent with codex/hermes's own
        no-profile-still-applies shape) was silently dropped."""
        provider = KimiCliProvider("term-1", "sess", "win", None, model="fable-5")
        command = provider._build_kimi_command()

        assert "--model fable-5" in command
        provider.cleanup()


class TestKimiCliProviderMisc:
    """Tests for miscellaneous KimiCliProvider methods and lifecycle."""

    def test_exit_cli(self):
        """Test exit command returns /exit."""
        provider = KimiCliProvider("term-1", "session-1", "window-1")
        assert provider.exit_cli() == "/exit"

    def test_cleanup(self):
        """Test cleanup resets initialized state and latching flag."""
        provider = KimiCliProvider("term-1", "session-1", "window-1")
        provider._initialized = True
        provider._has_received_input = True
        provider.cleanup()
        assert provider._initialized is False
        assert provider._has_received_input is False

    def test_cleanup_removes_temp_dir(self):
        """Test cleanup removes temporary directory and its contents."""
        provider = KimiCliProvider("term-1", "session-1", "window-1")
        provider._temp_dir = tempfile.mkdtemp(prefix="cao_kimi_test_")
        temp_path = provider._temp_dir  # Save path before cleanup resets it

        # Create a file in temp dir to verify it's removed
        test_file = os.path.join(temp_path, "test.txt")
        with open(test_file, "w") as f:
            f.write("test")

        provider.cleanup()
        assert provider._temp_dir is None
        assert not os.path.exists(temp_path)

    @patch("cli_agent_orchestrator.providers.kimi_cli.load_agent_profile")
    def test_cleanup_removes_session_kimi_home(self, mock_load, monkeypatch, tmp_path):
        """agent.md + session kimi-home are removed together with the temp dir."""
        user_home = tmp_path / "user-kimi-home"
        user_home.mkdir()
        monkeypatch.setenv("KIMI_CODE_HOME", str(user_home))

        mock_profile = MagicMock()
        mock_profile.model = None
        mock_profile.system_prompt = "You are a developer"
        mock_profile.mcpServers = {
            "cao-mcp-server": {"type": "stdio", "command": "cao-mcp-server", "args": []}
        }
        mock_load.return_value = mock_profile

        provider = KimiCliProvider("term-1", "session-1", "window-1", agent_profile="dev")
        MOD = "cli_agent_orchestrator.utils.mcp_resolution"
        with (
            patch(f"{MOD}._sibling_script", return_value="/venv/bin/cao-mcp-server"),
            patch(f"{MOD}.shutil.which", return_value=None),
        ):
            provider._build_kimi_command()

        temp_dir = provider._temp_dir
        assert os.path.exists(os.path.join(temp_dir, "agent.md"))
        assert os.path.exists(os.path.join(temp_dir, "kimi-home", "mcp.json"))

        provider.cleanup()
        assert provider._temp_dir is None
        assert not os.path.exists(temp_dir)

    def test_cleanup_nonexistent_temp_dir(self):
        """Test cleanup handles already-removed temp directory gracefully."""
        provider = KimiCliProvider("term-1", "session-1", "window-1")
        provider._temp_dir = "/tmp/cao_kimi_nonexistent_12345"
        provider.cleanup()
        assert provider._temp_dir is None

    def test_provider_inherits_base(self):
        """Test provider inherits from BaseProvider."""
        from cli_agent_orchestrator.providers.base import BaseProvider

        provider = KimiCliProvider("term-1", "session-1", "window-1")
        assert isinstance(provider, BaseProvider)

    def test_provider_default_state(self):
        """Test provider default initialization state."""
        provider = KimiCliProvider("term-1", "session-1", "window-1")
        assert provider._initialized is False
        assert provider._agent_profile is None
        assert provider._temp_dir is None
        assert provider._has_received_input is False
        assert provider.terminal_id == "term-1"
        assert provider.session_name == "session-1"
        assert provider.window_name == "window-1"

    def test_provider_with_agent_profile(self):
        """Test provider stores agent profile."""
        provider = KimiCliProvider("term-1", "session-1", "window-1", agent_profile="dev")
        assert provider._agent_profile == "dev"


# =============================================================================
# Pattern tests
# =============================================================================


class TestKimiCliProviderPatterns:
    """Tests for Kimi CLI regex patterns — validates correctness of all patterns."""

    def test_idle_prompt_pattern_thinking(self):
        """Test idle prompt pattern matches thinking mode prompt (💫)."""
        assert re.search(IDLE_PROMPT_PATTERN, "user@my-app💫")
        assert re.search(IDLE_PROMPT_PATTERN, "haofeif@cli-agent-orchestrator💫")

    def test_idle_prompt_pattern_bare_emoji(self):
        """Test idle prompt pattern matches bare emoji (Kimi v1.20.0+ format)."""
        assert re.search(IDLE_PROMPT_PATTERN, "💫")
        assert re.search(IDLE_PROMPT_PATTERN, "✨")

    def test_idle_prompt_pattern_no_thinking(self):
        """Test idle prompt pattern matches no-thinking mode prompt (✨)."""
        assert re.search(IDLE_PROMPT_PATTERN, "user@my-app✨")
        assert re.search(IDLE_PROMPT_PATTERN, "haofeif@project✨")

    def test_idle_prompt_pattern_with_dots_in_hostname(self):
        """Test idle prompt pattern matches hostnames with dots."""
        assert re.search(IDLE_PROMPT_PATTERN, "user@host.domain.com💫")

    def test_idle_prompt_pattern_does_not_match_random_text(self):
        """Test idle prompt pattern doesn't match arbitrary text."""
        assert not re.search(IDLE_PROMPT_PATTERN, "Hello world")
        assert not re.search(IDLE_PROMPT_PATTERN, "some random text")
        # With EOL anchor (as used in get_status), emoji followed by text doesn't match
        idle_prompt_eol = IDLE_PROMPT_PATTERN + r"\s*$"
        assert not re.search(idle_prompt_eol, "💫 alone")

    def test_welcome_banner_pattern(self):
        """Test welcome banner detection."""
        assert re.search(WELCOME_BANNER_PATTERN, "Welcome to Kimi Code CLI!")
        assert not re.search(WELCOME_BANNER_PATTERN, "Welcome to Claude Code")

    def test_user_input_box_patterns(self):
        """Test user input box boundary detection."""
        assert re.search(USER_INPUT_BOX_START_PATTERN, "╭──────────────╮")
        assert re.search(USER_INPUT_BOX_END_PATTERN, "╰──────────────╯")
        assert not re.search(USER_INPUT_BOX_START_PATTERN, "│ text │")

    def test_response_bullet_pattern(self):
        """Test response bullet detection."""
        assert re.search(RESPONSE_BULLET_PATTERN, "• Hello world!")
        assert re.search(RESPONSE_BULLET_PATTERN, "• Here is the code")
        assert not re.search(RESPONSE_BULLET_PATTERN, "Hello world")
        assert not re.search(RESPONSE_BULLET_PATTERN, "  • indented bullet")

    def test_thinking_bullet_raw_pattern(self):
        """Test thinking bullet detection in raw ANSI output."""
        # Gray-colored bullet (thinking mode)
        raw = "\x1b[38;5;244m•\x1b[39m \x1b[3m\x1b[38;5;244mThinking...\x1b[0m"
        assert re.search(THINKING_BULLET_RAW_PATTERN, raw)
        # Gray bullet with space before •
        raw_space = "\x1b[38;5;244m •\x1b[39m"
        assert re.search(THINKING_BULLET_RAW_PATTERN, raw_space)
        # Regular bullet (response mode) — should NOT match
        assert not re.search(THINKING_BULLET_RAW_PATTERN, "• Hello world")

    def test_error_pattern(self):
        """Test error pattern detection."""
        assert re.search(ERROR_PATTERN, "Error: connection failed", re.MULTILINE)
        assert re.search(ERROR_PATTERN, "ERROR: something went wrong", re.MULTILINE)
        assert re.search(ERROR_PATTERN, "ConnectionError: timeout", re.MULTILINE)
        assert re.search(ERROR_PATTERN, "APIError: rate limited", re.MULTILINE)
        assert re.search(ERROR_PATTERN, "Traceback (most recent call last):", re.MULTILINE)
        assert not re.search(ERROR_PATTERN, "No errors found", re.MULTILINE)

    def test_status_bar_pattern(self):
        """Test status bar detection."""
        assert re.search(STATUS_BAR_PATTERN, "23:14  yolo  agent (kimi-for-coding, thinking)")
        assert re.search(STATUS_BAR_PATTERN, "10:30  agent (kimi-for-coding)")
        assert not re.search(STATUS_BAR_PATTERN, "Hello world")

    def test_ansi_code_stripping(self):
        """Test ANSI code pattern strips all escape sequences."""
        raw = "\x1b[1muser@app💫\x1b[0m"
        clean = re.sub(ANSI_CODE_PATTERN, "", raw)
        assert clean == "user@app💫"

        raw2 = "\x1b[38;5;244m•\x1b[39m \x1b[3m\x1b[38;5;244mThinking\x1b[0m"
        clean2 = re.sub(ANSI_CODE_PATTERN, "", raw2)
        assert clean2 == "• Thinking"

    def test_idle_prompt_tail_lines(self):
        """Test tail lines constant is reasonable for Kimi's TUI layout."""
        assert IDLE_PROMPT_TAIL_LINES >= 40  # Must cover tall terminals (46+ rows)
        assert IDLE_PROMPT_TAIL_LINES <= 100  # Not unreasonably large


# ---------------------------------------------------------------------------
# Newest "Kimi Code" TUI status detection (regression for the redesigned CLI)
#
# These fixtures are REAL pipe-pane captures (last ~8KB rolling-buffer window)
# of the redesigned Kimi Code TUI, which has no ✨/💫 prompt — readiness is the
# "agent (<model> ●)" status bar / "context: N%" footer with an empty
# "── input ──" box, and a turn-in-flight is a braille spinner ("⠧ Thinking…
# Ns · N tokens"). The legacy emoji-prompt detector timed out at init on this
# build; get_status() must classify these raw buffers correctly.
# ---------------------------------------------------------------------------
_KIMI_FIXTURES = Path(__file__).parent / "fixtures"


class TestKimiCodeNewTuiStatus:
    """get_status() against real captured raw buffers of the new Kimi Code TUI."""

    def _provider(self):
        return KimiCliProvider("term-x", "session-x", "window-x", agent_profile="developer")

    def test_new_tui_idle_raw_capture(self):
        """A freshly-initialized terminal (no task yet) reads IDLE — this is the
        init readiness signal the legacy emoji detector missed."""
        buf = (_KIMI_FIXTURES / "kimi_code_tui_idle_raw.txt").read_text(encoding="utf-8")
        assert self._provider().get_status(buf) == TerminalStatus.IDLE

    def test_new_tui_processing_raw_capture(self):
        """A turn in flight (braille spinner is the freshest output) reads PROCESSING."""
        buf = (_KIMI_FIXTURES / "kimi_code_tui_processing_raw.txt").read_text(encoding="utf-8")
        assert self._provider().get_status(buf) == TerminalStatus.PROCESSING

    def test_new_tui_completed_raw_capture(self):
        """A finished turn (response present, spinner cleared, prompt visible)
        reads COMPLETED even though stale spinner frames linger in the buffer."""
        buf = (_KIMI_FIXTURES / "kimi_code_tui_completed_raw.txt").read_text(encoding="utf-8")
        assert self._provider().get_status(buf) == TerminalStatus.COMPLETED

    def test_new_tui_stale_spinner_does_not_block_completed(self):
        """Position-based detection: a stale braille frame earlier in the buffer
        must not be mistaken for a live turn once the prompt has redrawn."""
        buf = (_KIMI_FIXTURES / "kimi_code_tui_completed_raw.txt").read_text(encoding="utf-8")
        # Sanity: the completed fixture really does contain stale braille frames.
        assert any("⠀" <= ch <= "⣿" for ch in buf)
        assert self._provider().get_status(buf) != TerminalStatus.PROCESSING


class TestKimiCodeNewTuiStaleStreamSpinner:
    """kimi 0.34 renders its last "working…" spinner frame BELOW the response
    bullets (screen-bottom spinner), so a finished turn can end with
    spinner-after-bullet inside the rolling buffer while the RENDERED pane is
    already back at the ready chrome. The stream-only checks would pin such a
    terminal at PROCESSING forever (no further chunks → no quiescence
    re-detection); the rendered pane is the authoritative post-dispatch
    disambiguator (live E2E evidence, kimi 0.34 on a 220x50 CAO terminal)."""

    STREAM = (
        "✨ What is 17*23?\n"
        "\n"
        "• The user asks for multiplication.\n"
        "\n"
        "• 391.\n"
        "\n"
        " ⠴ working... · Tip: ctrl-s to add guidance without waiting for the turn to finish\n"
        "\n"
        "╭──────────────╮\n"
        "│ >            │\n"
        "╰──────────────╯\n"
        "yolo  agent (Kimi-k2.6 ●)  /tmp/x\n"
        "context: 4.0% (10.4k/262.1k)\n"
    )

    READY_PANE = (
        "• 391.\n"
        "╭──────────────╮\n"
        "│ >            │\n"
        "╰──────────────╯\n"
        "yolo  agent (Kimi-k2.6 ●)  /tmp/x\n"
        "context: 4.0% (10.4k/262.1k)\n"
    )

    def _dispatched_provider(self):
        import time as _time

        p = KimiCliProvider("test123", "test-session", "window-0")
        p.mark_input_received()
        p._last_dispatch_time = _time.time() - 30.0  # dispatch grace expired
        return p

    def test_stale_stream_spinner_with_clean_pane_is_completed(self):
        """Stream tail shows a stale spinner after the bullets, but the rendered
        pane is back at the ready chrome → COMPLETED (the live 0.34 shape)."""
        p = self._dispatched_provider()
        with patch("cli_agent_orchestrator.providers.kimi_cli.get_backend") as mock_backend:
            mock_backend.return_value.get_history.return_value = self.READY_PANE
            assert p.get_status(self.STREAM) == TerminalStatus.COMPLETED

    def test_live_pane_spinner_is_processing(self):
        """Same stream, but the rendered pane still shows a live spinner →
        genuinely PROCESSING."""
        p = self._dispatched_provider()
        pane = self.READY_PANE.replace("• 391.\n", "• 391.\n⠹ Thinking… 5s · 220 tokens\n")
        with patch("cli_agent_orchestrator.providers.kimi_cli.get_backend") as mock_backend:
            mock_backend.return_value.get_history.return_value = pane
            assert p.get_status(self.STREAM) == TerminalStatus.PROCESSING

    def test_no_dispatch_still_trusts_stream(self):
        """Pre-dispatch (no input sent yet, e.g. boot screens): the stream's
        busy signal wins without a pane read — unchanged behavior."""
        p = KimiCliProvider("test123", "test-session", "window-0")
        with patch("cli_agent_orchestrator.providers.kimi_cli.get_backend") as mock_backend:
            assert p.get_status(self.STREAM) == TerminalStatus.PROCESSING
            mock_backend.return_value.get_history.assert_not_called()


class TestKimiCodeCompletionInvariant:
    """COMPLETED is NOT "no spinner". It requires a conjunction of positive
    signals (see report section D):
      1. new-TUI ready chrome present (status bar "agent (…●)" OR
         "context: N%" footer),
      2. input latched (_has_received_input — a "•" bullet or a dispatch),
      3. no live spinner in the stream tail / after the last bullet
         (position-aware), confirmed against the RENDERED pane
         post-dispatch,
      4. no error pattern, and the dispatch grace window expired.
    A transient frame with neither spinner nor idle chrome must stay
    PROCESSING; long tool output mid-turn must not read COMPLETED."""

    READY_CHROME = "yolo  agent (Kimi-k2.6 ●)  /tmp/x\ncontext: 4.0% (10.4k/262.1k)\n"

    def _dispatched_provider(self):
        import time as _time

        p = KimiCliProvider("test123", "test-session", "window-0")
        p.mark_input_received()
        p._last_dispatch_time = _time.time() - 30.0  # dispatch grace expired
        return p

    def test_transient_frame_without_idle_chrome_not_completed(self):
        """Output exists but neither spinner NOR idle chrome: the ready-chrome
        gate fails → must NOT be COMPLETED (mid-stream / tool-output frame)."""
        p = self._dispatched_provider()
        transient = "• still working on it\n• chunk of tool output\n"
        with patch("cli_agent_orchestrator.providers.kimi_cli.get_backend") as mock_backend:
            assert p.get_status(transient) != TerminalStatus.COMPLETED

    def test_stale_spinner_above_latest_interaction_does_not_block(self):
        """A stale spinner ABOVE the latest interaction (bullet) + ready
        chrome: position-aware logic (spinner before last bullet, outside the
        tail window) must not block COMPLETED."""
        p = self._dispatched_provider()
        buf = (
            "⠹ Using handoff({...})\n"  # stale spinner above the interaction
            "\n"
            "• dispatched the task\n"
            "\n"
            "• done, all results returned\n"
            "\n" + self.READY_CHROME
        )
        with patch("cli_agent_orchestrator.providers.kimi_cli.get_backend") as mock_backend:
            mock_backend.return_value.get_history.return_value = self.READY_CHROME
            assert p.get_status(buf) == TerminalStatus.COMPLETED

    def test_long_tool_output_without_ready_chrome_not_completed(self):
        """Long tool output (no spinner frames yet, no idle chrome): must not
        read COMPLETED while the turn is still streaming output."""
        p = self._dispatched_provider()
        long_output = "".join(
            f"• tool output line {i:04d} with a very long wrapped tail\n" for i in range(60)
        )
        with patch("cli_agent_orchestrator.providers.kimi_cli.get_backend") as mock_backend:
            assert p.get_status(long_output) != TerminalStatus.COMPLETED

    def test_wrapped_long_response_with_ready_chrome_completed(self):
        """Final long/wrapped response + idle chrome, no spinner anywhere:
        COMPLETED (the normal happy end state)."""
        p = self._dispatched_provider()
        long_response = "".join(f"• wrapped bullet {i:03d} " + "x" * 200 + "\n" for i in range(40))
        buf = long_response + "\n" + self.READY_CHROME
        with patch("cli_agent_orchestrator.providers.kimi_cli.get_backend") as mock_backend:
            mock_backend.return_value.get_history.return_value = self.READY_CHROME
            assert p.get_status(buf) == TerminalStatus.COMPLETED

    def test_active_spinner_is_processing(self):
        """A live spinner (freshest content in stream AND rendered pane) is
        PROCESSING even with chrome present."""
        p = self._dispatched_provider()
        buf = "• partial answer\n" + "⠹ Thinking… 5s · 220 tokens\n" + self.READY_CHROME
        pane_with_spinner = "⠹ Thinking… 5s · 220 tokens\n" + self.READY_CHROME
        with patch("cli_agent_orchestrator.providers.kimi_cli.get_backend") as mock_backend:
            mock_backend.return_value.get_history.return_value = pane_with_spinner
            assert p.get_status(buf) == TerminalStatus.PROCESSING


class TestKimiCodeNewTuiExtraction:
    """Extraction for the newest "Kimi Code" TUI (✨ prompt, • bullets).

    Ground truth from a live Kimi Code session: user messages render as
    ✨-prefixed lines (no ╭─ input box), responses as • bullets, and the
    footer is a "── input ──" rule + status bar. Decorative ╰─ boxes from
    boot banners (Kimi welcome box, FastMCP server banner) appear ABOVE the
    conversation — anchoring on the last box-end used to slice the
    "response" out of the boot screen and run it to end-of-capture.
    """

    NEW_TUI_CAPTURE = (
        "╭──────────────────────────────╮\n"
        "│  ▐█▛█▛█▌  Welcome to Kimi Code CLI!  │\n"
        "│  Directory: /tmp/cao_kimi_x  │\n"
        "╰──────────────────────────────╯\n"
        "╭──────────────────────────────╮\n"
        "│  🖥  Server: cao-mcp-server, 3.4.2  │\n"
        "╰──────────────────────────────╯\n"
        "✨ What is 17*23? Reply with just the number and one short sentence.\n"
        "\n"
        "• The user is asking for a simple multiplication: 17 * 23.\n"
        "\n"
        "• 391. The product of 17 and 23 is 391.\n"
        "\n"
        "── input ─────────────────────────────────\n"
        "\n"
        "\n"
        "──────────────────────────────────────────\n"
        "yolo  agent (Kimi-k2.6 ●)  /tmp/cao_kimi_x  ctrl-o: editor\n"
        "context: 4.0% (10.4k/262.1k)\n"
    )

    def test_extracts_response_after_sparkle_prompt(self):
        provider = KimiCliProvider("test123", "test-session", "window-0")
        result = provider.extract_last_message_from_script(self.NEW_TUI_CAPTURE)
        assert "391" in result

    def test_extraction_excludes_footer_chrome_and_boot_banners(self):
        provider = KimiCliProvider("test123", "test-session", "window-0")
        result = provider.extract_last_message_from_script(self.NEW_TUI_CAPTURE)
        assert "input ─" not in result
        assert "context:" not in result
        assert "Welcome to Kimi Code CLI" not in result
        assert "cao-mcp-server, 3.4.2" not in result


class TestKimiCodeDispatchGrace:
    """Newest-TUI: a fresh dispatch must read PROCESSING even when the ready
    chrome (status bar) is the freshest content in the buffer.

    Real failure (supervisor-assign e2e): the paste repaints the status bar
    before the turn's first spinner frame, so the spinner-vs-ready position
    compare read COMPLETED ~130ms after send_input(); the StatusMonitor
    ready-latch then pinned the false COMPLETED for the entire turn and the
    test extracted mid-flight output.
    """

    NEW_TUI_READY_CHROME = (
        "✨ Analyze the datasets and report back.\n"
        "\n"
        "• Dispatching analysts now.\n"
        "── input ─────────────────────────────────\n"
        "\n"
        "──────────────────────────────────────────\n"
        "yolo  agent (Kimi-k2.6 ●)  /tmp/cao_kimi_x  ctrl-o: editor\n"
        "context: 4.0% (10.4k/262.1k)\n"
    )

    def test_fresh_dispatch_reads_processing(self):
        provider = KimiCliProvider("test123", "test-session", "window-0")
        provider.mark_input_received()
        assert provider.get_status(self.NEW_TUI_READY_CHROME) == TerminalStatus.PROCESSING

    def test_grace_expires_to_completed_when_pane_clear(self):
        import time as _time

        provider = KimiCliProvider("test123", "test-session", "window-0")
        provider.mark_input_received()
        provider._last_dispatch_time = _time.time() - 6.0
        with patch("cli_agent_orchestrator.providers.kimi_cli.get_backend") as mock_backend:
            # Rendered pane shows the same ready chrome — no live spinner.
            mock_backend.return_value.get_history.return_value = self.NEW_TUI_READY_CHROME
            assert provider.get_status(self.NEW_TUI_READY_CHROME) == TerminalStatus.COMPLETED

    def test_ready_stream_with_live_pane_spinner_is_processing(self):
        """A ready-looking chunk boundary mid-turn: the stream shows the ready
        chrome, but the RENDERED pane still shows a live spinner — must stay
        PROCESSING (this is the frame the StatusMonitor latch would otherwise
        pin as a false COMPLETED)."""
        import time as _time

        provider = KimiCliProvider("test123", "test-session", "window-0")
        provider.mark_input_received()
        provider._last_dispatch_time = _time.time() - 6.0
        pane = self.NEW_TUI_READY_CHROME.replace("── input ─", "⠹ Using handoff({...})\n── input ─")
        with patch("cli_agent_orchestrator.providers.kimi_cli.get_backend") as mock_backend:
            mock_backend.return_value.get_history.return_value = pane
            assert provider.get_status(self.NEW_TUI_READY_CHROME) == TerminalStatus.PROCESSING

    def test_pane_read_failure_falls_back_to_stream(self):
        import time as _time

        provider = KimiCliProvider("test123", "test-session", "window-0")
        provider.mark_input_received()
        provider._last_dispatch_time = _time.time() - 6.0
        with patch("cli_agent_orchestrator.providers.kimi_cli.get_backend") as mock_backend:
            mock_backend.return_value.get_history.side_effect = RuntimeError("pane gone")
            assert provider.get_status(self.NEW_TUI_READY_CHROME) == TerminalStatus.COMPLETED

    def test_no_dispatch_unaffected(self):
        provider = KimiCliProvider("test123", "test-session", "window-0")
        # bullet in buffer latches _has_received_input → COMPLETED, as before;
        # no dispatch yet → pane confirmation is skipped entirely.
        assert provider.get_status(self.NEW_TUI_READY_CHROME) == TerminalStatus.COMPLETED


class TestKimiScreenDetection:
    """Viewport detector (get_status_from_screen) — pyte-composited screens."""

    def _p(self):
        return KimiCliProvider("test123", "test-session", "window-0")

    def test_mcp_connecting_is_processing_not_idle(self):
        """Kimi draws its status bar BEFORE accepting input; during
        'connecting to mcp servers' a premature IDLE lets the inbox paste a
        message into the boot screen where it is absorbed (observed live)."""
        screen = [
            "⠧ MCP Servers: 0/1 connected, 0 tools",
            "⠦ cao-mcp-server (connecting)",
            "── input ──────────────",
            "yolo  agent (Kimi-k2.6 ●)  /tmp/x",
            "connecting to mcp servers...",
        ]
        assert self._p().get_status_from_screen(screen) == TerminalStatus.PROCESSING

    def test_ready_status_bar_no_bullet_is_idle(self):
        screen = [
            "▐█▛█▛█▌  Welcome to Kimi Code CLI!",
            "── input ──────────────",
            "yolo  agent (Kimi-k2.6 ●)  /tmp/x",
            "context: 1.0% (2.6k/262.1k)",
        ]
        assert self._p().get_status_from_screen(screen) == TerminalStatus.IDLE

    def test_status_bar_with_bullet_is_completed(self):
        screen = [
            "✨ What is 17*23?",
            "• 391. The product of 17 and 23 is 391.",
            "── input ──────────────",
            "yolo  agent (Kimi-k2.6 ●)  /tmp/x",
            "context: 4.0% (10.4k/262.1k)",
        ]
        assert self._p().get_status_from_screen(screen) == TerminalStatus.COMPLETED

    def test_response_mentioning_connecting_is_not_boot_gated(self):
        """A COMPLETED turn whose "•" response bullet merely MENTIONS the boot
        chrome ("connecting to mcp servers" / "(connecting)") must not be
        re-stranded as PROCESSING. The boot gate scans non-bullet boot chrome
        only; a whole-screen scan would pin this terminal at PROCESSING and
        InboxService (delivers on IDLE/COMPLETED) would never deliver to it."""
        screen = [
            "✨ How does kimi boot?",
            "• It logs 'connecting to mcp servers' and shows cao-mcp-server (connecting) until ready.",
            "── input ──────────────",
            "yolo  agent (Kimi-k2.6 ●)  /tmp/x",
            "context: 4.0% (10.4k/262.1k)",
        ]
        assert self._p().get_status_from_screen(screen) == TerminalStatus.COMPLETED

    def test_live_spinner_is_processing(self):
        screen = [
            "✨ Analyze the data",
            "• Working through it.",
            "⠹ Using handoff({...})",
            "── input ──────────────",
            "yolo  agent (Kimi-k2.6 ●)  /tmp/x",
        ]
        assert self._p().get_status_from_screen(screen) == TerminalStatus.PROCESSING

    def test_torn_down_shell_is_unknown(self):
        screen = ["Bye!", "rkram@host:/tmp/x$"]
        assert self._p().get_status_from_screen(screen) == TerminalStatus.UNKNOWN
