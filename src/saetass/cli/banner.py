import logging

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .palette import SAETASS_ORANGE, SAETASS_YELLOW
from .progress import console as default_console

logger = logging.getLogger(__name__)

DEFAULT_BANNER = r"""
          :::::::::::::::           
      :::::::::::::::::::::::       
    :::::::::++++++++++::::::::     
   ::::::::++++++++++++++::::::::   
  :::::::++++++++++++++++++:::::::  
 ::::::::++++++++++++++++++:::::::: 
 :::::::++++++++++++++++++++::::::: 
 :::::::+++++++++++++++++++:::::::: 
 ::::::::++++++++++++++++++:::::::: 
  ::::::::::++++++++++++++          
   :::::::::::::::++++++            
     :::::::::::::::::::::          
        ::::::::::::::::::::::      
            ++:::::::::::::::::::   
          ++++++++++++::::::::::::  
        ++++++++++++++++++::::::::: 
::::::::++++++++++++++++++++::::::::
:::::::++++++++++++++++++++++:::::::
:::::::++++++++++++++++++++++:::::::
 :::::::++++++++++++++++++++::::::::
 ::::::::++++++++++++++++++:::::::: 
   :::::::++++++++++++++++::::::::  
    :::::::::++++++++++:::::::::    
       :::::::::::::::::::::::      
           :::::::::::::::          
"""


class BannerManager:
    """
    Singleton manager for printing the SAETASS ASCII banner.
    Ensures the banner is printed at most once per Python process lifecycle,
    avoiding clutter when multiple Solver instances are created sequentially.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(BannerManager, cls).__new__(cls)
            cls._instance._printed = False
        return cls._instance

    def print_once(
        self,
        console: Console = None,
        text: str = DEFAULT_BANNER,
        title: str = "SAETASS: Solver for Astroparticle Equation of Transport Analysis in Spherical Symmetry",
    ):
        """
        Prints the banner if it hasn't been printed yet by this singleton.
        """
        if self._printed:
            return

        if console is None:
            console = default_console

        rich_text = Text(text)
        rich_text.highlight_regex(r":", SAETASS_ORANGE)
        rich_text.highlight_regex(r"\+", SAETASS_YELLOW)

        panel = Panel(
            Align.center(rich_text),
            title=f"[bold {SAETASS_YELLOW}]{title}[/]",
            expand=False,
            border_style=SAETASS_ORANGE,
        )
        console.print(Align.center(panel))
        self._printed = True

    def reset(self):
        """Allow resetting the singleton state, useful for testing."""
        self._printed = False


def print_banner(
    console: Console = None,
    text: str = DEFAULT_BANNER,
    title: str = "SAETASS: Solver for Astroparticle Equation of Transport Analysis in Spherical Symmetry",
):
    """
    Print the SAETASS ASCII banner.
    It delegates to a Singleton to ensure it's only printed once per runtime.

    Parameters
    ----------
    console : rich.console.Console, optional
        A rich Console instance to use for printing.
    text : str, optional
        The ASCII art/text to print.
    title : str, optional
        The string to place at the top of the panel border.
    """
    BannerManager().print_once(console=console, text=text, title=title)
