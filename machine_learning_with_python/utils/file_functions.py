from __future__ import annotations

import errno
import logging
import pathlib
import select
import string
import sys
import unicodedata

from typing import Tuple

import pandas as pd
from rich.console import Console
from rich.table import Table

import machine_learning_with_python
from machine_learning_with_python.constants import (
    FIFTY_THOUSAND,
    FIVE_HUNDRED_THOUSAND,
    FIVE_THOUSAND,
    ONE_HUNDRED_THOUSAND,
    ONE_MILLION,
    TEN_THOUSAND,
    THIRTY_THOUSAND,
    TWENTY_THOUSAND,
)
from machine_learning_with_python.ml_logger import get_logger  # noqa: E402
# # from machine_learning_with_python.fileobject import FileInfo
# # from machine_learning_with_python.utils.file_functions_mapping import FILE_FUNCTIONS_MAPPING
# from machine_learning_with_python.shell import ShellConsole, _popen, _popen_stdout
from machine_learning_with_python.utils.env import environ_get

LOGGER = get_logger(__name__, provider="File Functions", level=logging.DEBUG)

PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3

# https://gist.github.com/wassname/1393c4a57cfcbf03641dbc31886123b8
# python convert string to safe filename
VALID_FILENAME_CHARS = "-_.() %s%s" % (string.ascii_letters, string.digits)
CHAR_LIMIT = 255


def get_dataframe_from_csv(
    filename: str, return_parent_folder_name: bool = False
) -> pd.core.frame.DataFrame:
    """Open csv files and return a dataframe from pandas

    Args:
        filename (str): path to file
    """
    src = pathlib.Path(f"{filename}").resolve()
    df = pd.read_csv(f"{src}")

    # import bpdb
    # bpdb.set_trace()

    if return_parent_folder_name:
        return df, f"{src.parent.stem}"
    else:
        return df


def sort_dataframe(
    df: pd.core.frame.DataFrame, columns: list = [], ascending: Tuple = ()
) -> pd.core.frame.DataFrame:
    """Return dataframe sorted via columns

    Args:
        df (pd.core.frame.DataFrame): existing dataframe
        columns (list, optional): [description]. Defaults to []. Eg. ["Total Followers", "Total Likes", "Total Comments", "ERDay", "ERpost"]
        ascending (Tuple, optional): [description]. Defaults to (). Eg. (False, False, False, False, False)

    Returns:
        pd.core.frame.DataFrame: [description]
    """
    df = df.sort_values(columns, ascending=ascending)
    return df


def rich_format_followers(val: int) -> str:
    """Given a arbritary int, return a 'rich' string formatting

    Args:
        val (int): eg. followers = 4366347347457

    Returns:
        str: [description] eg. "[bold bright_yellow]4366347347457[/bold bright_yellow]"
    """
    ret = ""

    if val > ONE_MILLION:
        ret = f"[bold bright_yellow]{val}[/bold bright_yellow]"
    elif FIVE_HUNDRED_THOUSAND < val < ONE_MILLION:
        ret = f"[bold dark_orange]{val}[/bold dark_orange]"
    elif FIVE_HUNDRED_THOUSAND < val < ONE_MILLION:
        ret = f"[bold deep_pink2]{val}[/bold deep_pink2]"

    elif ONE_HUNDRED_THOUSAND < val < FIVE_HUNDRED_THOUSAND:
        ret = f"[bold orange_red1]{val}[/bold orange_red1]"

    elif FIFTY_THOUSAND < val < ONE_HUNDRED_THOUSAND:
        ret = f"[bold dodger_blue2]{val}[/bold dodger_blue2]"
    elif THIRTY_THOUSAND < val < FIFTY_THOUSAND:
        ret = f"[bold purple3]{val}[/bold purple3]"
    elif TWENTY_THOUSAND < val < THIRTY_THOUSAND:
        ret = f"[bold rosy_brown]{val}[/bold rosy_brown]"
    elif TEN_THOUSAND < val < TWENTY_THOUSAND:
        ret = f"[bold green]{val}[/bold green]"
    else:
        ret = f"[bold bright_white]{val}[/bold bright_white]"

    return ret


def rich_likes_or_comments(val: int) -> str:
    """Given a arbritary int, return a 'rich' string formatting

    Args:
        val (int): eg. followers = 4366347347457

    Returns:
        str: [description] eg. "[bold bright_yellow]4366347347457[/bold bright_yellow]"
    """
    ret = ""

    if TEN_THOUSAND >= val:
        ret = f"[bold bright_yellow]{val}[/bold bright_yellow]"
    elif FIFTY_THOUSAND < val < ONE_HUNDRED_THOUSAND:
        ret = f"[bold dodger_blue2]{val}[/bold dodger_blue2]"
    elif THIRTY_THOUSAND < val < FIFTY_THOUSAND:
        ret = f"[bold purple3]{val}[/bold purple3]"
    elif TWENTY_THOUSAND < val < THIRTY_THOUSAND:
        ret = f"[bold rosy_brown]{val}[/bold rosy_brown]"
    elif TEN_THOUSAND < val < TWENTY_THOUSAND:
        ret = f"[bold green]{val}[/bold green]"
    else:
        ret = f"[bold bright_white]{val}[/bold bright_white]"

    return ret


def rich_display_meme_pull_list(df: pd.core.frame.DataFrame):  # noqa
    console = Console()

    table = Table(show_header=True, header_style="bold magenta")

    table.add_column("Account")
    table.add_column("Social")
    table.add_column("Total Followers")
    table.add_column("Total Likes")
    table.add_column("Total Comments")
    table.add_column("Total Posts")
    table.add_column("Start Date")
    table.add_column("End Date")
    table.add_column("ERDay")
    table.add_column("ERpost")
    table.add_column("Average Likes")
    table.add_column("Average Comments")
    table.add_column("Links")

    for index, row in df.iterrows():
        account = f"[bold blue]{row['Account']}[/bold blue]"
        social = f"[bold]{row['Social']}[/bold]"
        total_followers = rich_format_followers(row["Total Followers"])
        total_likes = f"[bold]{row['Total Likes']}[/bold]"
        total_comments = f"[bold]{row['Total Comments']}[/bold]"
        total_posts = f"[bold]{row['Total Posts']}[/bold]"
        start_date = f"[bold]{row['Start Date']}[/bold]"
        end_date = f"[bold]{row['End Date']}[/bold]"
        erday = f"[bold]{row['ERDay']}[/bold]"
        erpost = f"[bold]{row['ERpost']}[/bold]"
        average_likes = f"[bold]{row['Average Likes']}[/bold]"
        average_comments = f"[bold]{row['Average Comments']}[/bold]"
        links = f"[bold]{row['Links']}[/bold]"

        table.add_row(
            account,
            social,
            total_followers,
            total_likes,
            total_comments,
            total_posts,
            start_date,
            end_date,
            erday,
            erpost,
            average_likes,
            average_comments,
            links,
        )

    console.print(table)


def rich_display_popstars_analytics(df: pd.core.frame.DataFrame):  # noqa
    console = Console()

    table = Table(show_header=True, header_style="bold magenta")

    table.add_column("Social")
    table.add_column("Author")
    table.add_column("Url")
    table.add_column("Likes")
    table.add_column("Comments")
    table.add_column("ER")
    table.add_column("Text")
    table.add_column("Date")
    table.add_column("Media 1")

    for index, row in df.iterrows():
        social = f"[bold]{row['Social']}[/bold]"
        author = f"[bold]{row['Author']}[/bold]"
        url = f"[bold]{row['Url']}[/bold]"
        likes = f"[bold]{rich_likes_or_comments(row['Likes'])}[/bold]"
        comments = f"[bold]{rich_likes_or_comments(row['Comments'])}[/bold]"
        er = f"[bold]{row['ER']}[/bold]"
        text = f"[bold]{row['Text']}[/bold]"
        date = f"[bold]{row['Date']}[/bold]"
        media = f"[bold]{row['Media 1']}[/bold]"

        table.add_row(social, author, url, likes, comments, er, text, date, media)

    console.print(table)


# smoke tests
if __name__ == "__main__":
    pass
