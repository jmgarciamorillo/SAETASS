import logging
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    SpinnerColumn,
)
from rich.theme import Theme
from .palette import (
    SAETASS_YELLOW,
    SAETASS_ORANGE,
    SAETASS_RED,
    SAETASS_BLUE,
    SAETASS_CYAN,
    SAETASS_GREEN,
)

# SAETASS Theme
custom_theme = Theme(
    {
        "info": SAETASS_CYAN,
        "warning": SAETASS_YELLOW,
        "danger": SAETASS_RED,
        "saetass_yellow": SAETASS_YELLOW,
        "saetass_orange": SAETASS_ORANGE,
        "saetass_red": SAETASS_RED,
        "saetass_blue": SAETASS_BLUE,
        "saetass_cyan": SAETASS_CYAN,
        "saetass_green": SAETASS_GREEN,
    }
)

import sys

width = None if sys.stdout.isatty() else 120
console = Console(theme=custom_theme, width=width)


def setup_rich_logging(level=logging.INFO):
    """
    Set up rich logging handler for the root logger.

    Parameters
    ----------
    level : int, optional
        The logging level to set (e.g., logging.INFO, logging.DEBUG).
    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)],
    )


def create_progress_bar():
    """
    Create a customized Rich Progress bar for the SAETASS solver.
    This bar uses the built-in global `console` and specifies custom columns.

    Returns
    -------
    Progress
        A rich.progress.Progress instance.
    """
    return Progress(
        SpinnerColumn(spinner_name="dots", style=SAETASS_YELLOW),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(
            bar_width=None,
            complete_style=SAETASS_CYAN,
            finished_style=SAETASS_GREEN,
            pulse_style=SAETASS_YELLOW,
        ),
        MofNCompleteColumn(),
        has_metrics_column := TextColumn("[dim]metrics:[/] {task.fields[metrics]}"),
        TimeElapsedColumn(),
        TextColumn("<"),
        TimeRemainingColumn(),
        console=console,
        transient=False,  # We want it to stay at the bottom
    )
