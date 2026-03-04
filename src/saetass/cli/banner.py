import logging
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.align import Align
from .progress import console as default_console
from .palette import SAETASS_CYAN, SAETASS_YELLOW, SAETASS_ORANGE

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


def print_banner(
    console: Console = None,
    text: str = DEFAULT_BANNER,
    title: str = "SAETASS: Solver for Astroparticle Equation of Transport Analysis in Spherical Symmetry",
):
    """
    Print the SAETASS ASCII banner.

    Parameters
    ----------
    console : rich.console.Console, optional
        A rich Console instance to use for printing.
    text : str, optional
        The ASCII art/text to print.
    title : str, optional
        The string to place at the top of the panel border.
    """
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
