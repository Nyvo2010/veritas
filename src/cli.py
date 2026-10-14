"""Interactive TUI for the Veritas Deep Research Agent."""

import sys
import os
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from rich.console import Console
from rich.prompt import Prompt

from ui import theme, screens
from settings import (
    load_settings,
    update_settings,
    mask_key,
    reports_dir,
    settings_path,
    DEFAULTS,
)
from main import DeepResearchAgent


def clear_screen():
    """Clear the terminal screen."""
    os.system('clear' if os.name == 'posix' else 'cls')


def pause(console: Console, label: str = "Press Enter to continue"):
    Prompt.ask(f"[{theme.PRIMARY}]{label}[/{theme.PRIMARY}]", default="", show_default=False)


def apply_saved_theme(settings: dict) -> str:
    return theme.apply_theme(str(settings.get("theme") or DEFAULTS["theme"]))


# ---------------------------------------------------------------------------
# Settings screens
# ---------------------------------------------------------------------------

def theme_menu(console: Console, settings: dict) -> dict:
    while True:
        clear_screen()
        screens.print_header(console, "Settings - Color Theme")
        names = list(theme.THEMES.keys())
        current = str(settings.get("theme", DEFAULTS["theme"]))
        rows = [(f"Color {i + 1}", name + ("  (current)" if name == current else ""))
                for i, name in enumerate(names)]
        screens.kv_panel(console, "Available Themes", rows)
        choice = Prompt.ask(
            f"[{theme.PRIMARY}]Pick 1-{len(names)} or Enter to go back[/{theme.PRIMARY}]",
            default="", show_default=False,
        ).strip()
        if choice == "":
            return settings
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            picked = names[int(choice) - 1]
            settings = update_settings(theme=picked)
            apply_saved_theme(settings)
            screens.print_panel(console, f"Theme set to {picked}.", "Saved", style=theme.SUCCESS)
            pause(console)
            return settings


def reports_menu(console: Console, settings: dict) -> dict:
    clear_screen()
    screens.print_header(console, "Settings - Reports Folder")
    screens.kv_panel(console, "Current", [
        ("Folder", reports_dir(settings)),
    ])
    value = Prompt.ask(
        f"[{theme.PRIMARY}]New folder or Enter to keep[/{theme.PRIMARY}]",
        default="", show_default=False,
    ).strip()
    if value:
        expanded = os.path.expanduser(value)
        try:
            os.makedirs(expanded, exist_ok=True)
        except OSError as e:
            screens.print_panel(console, f"Cannot use that folder: {e}", "Error", style=theme.ERROR)
            pause(console)
            return settings
        settings = update_settings(reports_dir=value)
        screens.print_panel(console, f"Reports will be saved to {expanded}.", "Saved", style=theme.SUCCESS)
    else:
        screens.print_panel(console, "No changes made.", "Reports Folder")
    pause(console)
    return settings


def keys_menu(console: Console, settings: dict) -> dict:
    while True:
        clear_screen()
        screens.print_header(console, "Settings - API Keys")
        screens.kv_panel(console, "Keys (secrets stay in your settings file)", [
            ("[1] NVIDIA", mask_key(settings.get("nvidia_api_key", ""))),
            ("[2] Tavily", mask_key(settings.get("tavily_api_key", ""))),
            ("[3] Base URL", str(settings.get("nvidia_api_base_url", ""))),
        ])
        console.print("Keys typed here overwrite the settings file. Environment variables still take precedence.")
        console.print()
        choice = Prompt.ask(
            f"[{theme.PRIMARY}]Pick 1-3, 'clear' to wipe keys, or Enter to go back[/{theme.PRIMARY}]",
            default="", show_default=False,
        ).strip().lower()
        if choice == "":
            return settings
        if choice == "clear":
            settings = update_settings(nvidia_api_key="", tavily_api_key="")
            screens.print_panel(console, "Stored keys cleared.", "Saved", style=theme.SUCCESS)
            pause(console)
            return settings
        if choice in ("1", "2", "3"):
            label = {"1": "NVIDIA API key", "2": "Tavily API key", "3": "NVIDIA base URL"}[choice]
            value = Prompt.ask(
                f"[{theme.PRIMARY}]{label}[/{theme.PRIMARY}]",
                default="", show_default=False,
            ).strip()
            if value:
                key = {"1": "nvidia_api_key", "2": "tavily_api_key", "3": "nvidia_api_base_url"}[choice]
                settings = update_settings(**{key: value})
                screens.print_panel(console, f"{label} saved.", "Saved", style=theme.SUCCESS)
                pause(console)
                return settings
            return settings


def defaults_menu(console: Console, settings: dict) -> dict:
    clear_screen()
    screens.print_header(console, "Settings - Research Defaults")
    screens.kv_panel(console, "Current Defaults", [
        ("Max iterations", str(settings.get("max_iterations", 10))),
        ("Max search results", str(settings.get("max_search_results", 15))),
    ])
    raw_iter = Prompt.ask(
        f"[{theme.PRIMARY}]Max iterations 1-20 or Enter to keep[/{theme.PRIMARY}]",
        default="", show_default=False,
    ).strip()
    raw_results = Prompt.ask(
        f"[{theme.PRIMARY}]Max search results 1-20 or Enter to keep[/{theme.PRIMARY}]",
        default="", show_default=False,
    ).strip()
    updates = {}
    if raw_iter.isdigit() and 1 <= int(raw_iter) <= 20:
        updates["max_iterations"] = int(raw_iter)
    if raw_results.isdigit() and 1 <= int(raw_results) <= 20:
        updates["max_search_results"] = int(raw_results)
    if updates:
        settings = update_settings(**updates)
        screens.print_panel(console, "Defaults saved.", "Saved", style=theme.SUCCESS)
    else:
        screens.print_panel(console, "No changes made.", "Research Defaults")
    pause(console)
    return settings


def default_sources_menu(console: Console, settings: dict) -> dict:
    toggles = dict(settings.get("category_toggles", {}))
    while True:
        clear_screen()
        screens.print_header(console, "Settings - Default Sources")
        screens.filters_panel(console, toggles, title="Default Sources")
        choice = Prompt.ask(
            f"[{theme.PRIMARY}]Toggle 1-{len(toggles)}, 'all', 'none', or Enter to go back[/{theme.PRIMARY}]",
            default="", show_default=False,
        ).strip().lower()
        if choice == "":
            return settings
        names = list(toggles.keys())
        if choice == "all":
            toggles = {k: True for k in names}
        elif choice == "none":
            toggles = {k: False for k in names}
        elif choice.isdigit() and 1 <= int(choice) <= len(names):
            name = names[int(choice) - 1]
            toggles[name] = not toggles[name]
        else:
            continue
        settings = update_settings(category_toggles=toggles)


def settings_menu(console: Console, settings: dict) -> dict:
    options = [
        ("1", "Color theme"),
        ("2", "Reports folder"),
        ("3", "API keys"),
        ("4", "Research defaults"),
        ("5", "Default sources"),
        ("6", "Back to main menu"),
    ]
    while True:
        clear_screen()
        screens.print_header(console, "Settings")
        screens.kv_panel(console, "Stored In", [
            ("File", settings_path()),
            ("Theme", str(settings.get("theme", ""))),
            ("Reports", reports_dir(settings)),
            ("NVIDIA key", mask_key(settings.get("nvidia_api_key", ""))),
            ("Tavily key", mask_key(settings.get("tavily_api_key", ""))),
        ])
        screens.menu_panel(console, options)
        choice = Prompt.ask(f"[{theme.PRIMARY}]Select option[/{theme.PRIMARY}]").strip()
        if choice == "1":
            settings = theme_menu(console, settings)
        elif choice == "2":
            settings = reports_menu(console, settings)
        elif choice == "3":
            settings = keys_menu(console, settings)
        elif choice == "4":
            settings = defaults_menu(console, settings)
        elif choice == "5":
            settings = default_sources_menu(console, settings)
        elif choice == "6" or choice == "":
            return settings


def view_config(console: Console, agent: DeepResearchAgent, settings: dict):
    from config import NVIDIA_MODEL, MAX_PARALLEL_SECTIONS

    clear_screen()
    screens.print_header(console, "Configuration")
    screens.kv_panel(console, "System", [
        ("Model", NVIDIA_MODEL),
        ("Parallel sections", str(MAX_PARALLEL_SECTIONS)),
        ("Theme", str(settings.get("theme", ""))),
        ("Reports folder", reports_dir(settings)),
        ("Settings file", settings_path()),
        ("NVIDIA key", mask_key(settings.get("nvidia_api_key", ""))),
        ("Tavily key", mask_key(settings.get("tavily_api_key", ""))),
        ("Default iterations", str(settings.get("max_iterations", 10))),
    ])
    screens.filters_panel(console, agent.category_toggles, title="Active Sources")
    pause(console)


# ---------------------------------------------------------------------------
# Research flow
# ---------------------------------------------------------------------------

def start_research(console: Console, agent: DeepResearchAgent, settings: dict):
    default_iter = int(settings.get("max_iterations", 10))

    # Step 1: query
    while True:
        clear_screen()
        screens.print_header(console, "New Research")
        screens.kv_panel(console, "Session", [
            ("Model", agent.nvidia.model),
            ("Reports to", reports_dir(settings)),
        ])
        query = Prompt.ask(f"[{theme.PRIMARY}]Query (Enter to cancel)[/{theme.PRIMARY}]").strip()
        if query == "":
            return
        if len(query.split()) > 1:
            break
        screens.print_panel(console, "Please enter a full question, not a single word.", "Too Short", style=theme.WARNING)

    short_query = query if len(query) <= 60 else query[:57] + "..."

    # Step 2: iterations
    while True:
        clear_screen()
        screens.print_header(console, "New Research")
        screens.kv_panel(console, "Research", [
            ("Query", short_query),
            ("Default iterations", str(default_iter)),
        ])
        raw = Prompt.ask(
            f"[{theme.PRIMARY}]Max iterations 1-20 (Enter for {default_iter})[/{theme.PRIMARY}]",
            default="", show_default=False,
        ).strip()
        if raw == "":
            max_iter = default_iter
            break
        if raw.isdigit() and 1 <= int(raw) <= 20:
            max_iter = int(raw)
            break

    # Step 3: sources for this run
    while True:
        clear_screen()
        screens.print_header(console, "New Research")
        screens.kv_panel(console, "Research", [
            ("Query", short_query),
            ("Max iterations", str(max_iter)),
        ])
        screens.filters_panel(console, agent.category_toggles, title="Sources For This Run")
        names = list(agent.category_toggles.keys())
        choice = Prompt.ask(
            f"[{theme.PRIMARY}]Toggle 1-{len(names)}, 'all', 'none', or Enter to continue[/{theme.PRIMARY}]",
            default="", show_default=False,
        ).strip().lower()
        if choice == "":
            break
        if choice == "all":
            for name in names:
                agent.set_category_toggle(name, True, silent=True)
        elif choice == "none":
            for name in names:
                agent.set_category_toggle(name, False, silent=True)
        elif choice.isdigit() and 1 <= int(choice) <= len(names):
            name = names[int(choice) - 1]
            agent.set_category_toggle(name, not agent.category_toggles[name], silent=True)

    # Step 4: confirm
    clear_screen()
    screens.print_header(console, "New Research")
    active = sum(1 for v in agent.category_toggles.values() if v)
    screens.kv_panel(console, "Confirm", [
        ("Query", short_query),
        ("Max iterations", str(max_iter)),
        ("Sources", f"{active}/{len(agent.category_toggles)} enabled"),
        ("Reports to", reports_dir(settings)),
    ])
    confirm = Prompt.ask(f"[{theme.PRIMARY}]Start research? (y/n)[/{theme.PRIMARY}]").strip().lower()
    if confirm not in ("y", "yes"):
        return

    # Run (the live renderer owns the screen from here; no clearing mid-run)
    try:
        state = asyncio.run(agent.research(query, max_iterations=max_iter))
    except KeyboardInterrupt:
        console.print()
        screens.print_panel(console, "Research stopped by user.", "Interrupted", style=theme.WARNING)
        pause(console)
        return
    except Exception as e:
        console.print()
        screens.print_panel(console, f"Research failed: {e}", "Error", style=theme.ERROR)
        pause(console)
        return

    # Save + report location
    filename = agent.save_report(state)
    if filename:
        shown = filename if len(filename) <= 80 else "..." + filename[-77:]
        screens.print_panel(console, f"Saved to {shown}", "Report Saved", style=theme.SUCCESS)
    else:
        screens.print_panel(console, "Report could not be saved.", "Save Failed", style=theme.ERROR)
    pause(console, "Press Enter to return to main menu")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_agent(settings: dict):
    return DeepResearchAgent(
        category_toggles=dict(settings.get("category_toggles", {})),
    )


def classic_main():
    console = Console()
    settings = load_settings()
    apply_saved_theme(settings)

    clear_screen()
    screens.print_header(console)
    screens.print_panel(console, "Initializing agent...", "Startup")

    try:
        agent = build_agent(settings)
        # Fail fast if no keys resolve from env or settings.
        from settings import resolve_api_keys
        nvidia_key, tavily_key, _ = resolve_api_keys(settings)
        if not nvidia_key or not tavily_key:
            raise ValueError("Missing API keys")
    except Exception as e:
        clear_screen()
        screens.print_header(console)
        screens.print_panel(
            console,
            "Could not initialize the agent. Add your API keys under Settings (option 2), "
            "via environment variables, or in your .env file.\n"
            f"Details: {e}",
            "Initialization Error",
            style=theme.ERROR,
        )
        pause(console)
        # Still enter the menu so the user can fix keys via Settings.
        try:
            agent = build_agent(settings)
        except Exception:
            agent = None

    while True:
        # Reload settings each loop so edits on disk are picked up.
        settings = load_settings()
        apply_saved_theme(settings)
        if agent is None:
            try:
                agent = build_agent(settings)
            except Exception:
                agent = None

        clear_screen()
        screens.print_header(console)
        screens.menu_panel(console, [
            ("1", "Start new research"),
            ("2", "Settings"),
            ("3", "View configuration"),
            ("4", "Exit"),
        ])
        choice = Prompt.ask(f"[{theme.PRIMARY}]Select option[/{theme.PRIMARY}]").strip()

        if choice == "1":
            if agent is None:
                screens.print_panel(console, "Agent is not initialized. Add API keys in Settings first.", "Not Ready", style=theme.WARNING)
                pause(console)
                continue
            # Refresh per-run toggles from defaults each time.
            agent.category_toggles = dict(settings.get("category_toggles", agent.category_toggles))
            start_research(console, agent, settings)
        elif choice == "2":
            settings = settings_menu(console, settings)
            try:
                agent = build_agent(settings)
            except Exception:
                agent = None
        elif choice == "3":
            if agent is None:
                try:
                    agent = build_agent(settings)
                except Exception:
                    agent = None
            if agent is None:
                screens.print_panel(console, "Agent is not initialized.", "Configuration", style=theme.WARNING)
                pause(console)
            else:
                view_config(console, agent, settings)
        elif choice == "4" or choice.lower() in ("q", "quit", "exit"):
            clear_screen()
            screens.print_header(console)
            screens.print_panel(console, "Thank you for using Veritas. Goodbye.", "Exit")
            sys.exit(0)


def main():
    """Launch the OpenTUI app, falling back to the Rich CLI if needed."""
    try:
        from tui_app import run as run_tui
    except ImportError:
        return classic_main()
    try:
        run_tui()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console = Console()
        console.print()
        screens.print_panel(console, "Goodbye.", "Exit")
        sys.exit(0)
