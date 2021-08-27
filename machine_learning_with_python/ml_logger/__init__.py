#!/usr/bin/env python3

import collections

try:  # python 3
    from collections import abc
except ImportError:  # python 2
    import collections as abc

import concurrent.futures
from datetime import datetime
import gc
import inspect
import logging
from logging import Logger, LogRecord
import os
# import slack
import sys
from types import FrameType

from typing import Deque, Optional, cast

from loguru import logger

from machine_learning_with_python.models.loggers import LoggerModel, LoggerPatch

LOGGERS = __name__


class InterceptHandler(logging.Handler):
    """
    Intercept all logging calls (with standard logging) into our Loguru Sink
    See: https://github.com/Delgan/loguru#entirely-compatible-with-standard-logging
    """

    loglevel_mapping = {
        50: "CRITICAL",
        40: "ERROR",
        30: "WARNING",
        20: "INFO",
        10: "DEBUG",
        0: "NOTSET",
    }

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:  # noqa: WPS609
            frame = cast(FrameType, frame.f_back)
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level,
            record.getMessage(),
        )


# """ Logging handler intercepting existing handlers to redirect them to loguru """
class LoopDetector(logging.Filter):
    """
    Log filter which looks for repeating WARNING and ERROR log lines, which can
    often indicate that a module is spinning on a error or stuck waiting for a
    condition.

    When a repeating line is found, a summary message is printed and a message
    optionally sent to Slack.
    """

    LINE_HISTORY_SIZE = 50
    LINE_REPETITION_THRESHOLD = 5

    def __init__(self) -> None:
        self._recent_lines: Deque[str] = collections.deque(
            maxlen=self.LINE_HISTORY_SIZE
        )
        self._supressed_lines: collections.Counter = collections.Counter()

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < logging.WARNING:
            return True

        self._recent_lines.append(record.getMessage())

        counter = collections.Counter(list(self._recent_lines))
        repeated_lines = [
            line
            for line, count in counter.most_common()
            if count > self.LINE_REPETITION_THRESHOLD
            and line not in self._supressed_lines
        ]

        if repeated_lines:
            for line in repeated_lines:
                self._supressed_lines[line] = self.LINE_HISTORY_SIZE

        for line, count in self._supressed_lines.items():
            self._supressed_lines[line] = count - 1
            # mypy doesn't understand how to deal with collection.Counter's
            # unary addition operator
            self._supressed_lines = +self._supressed_lines  # type: ignore

        # https://docs.python.org/3/library/logging.html#logging.Filter.filter
        # The docs lie when they say that this returns an int, it's really a bool.
        # https://bugs.python.org/issue42011
        # H6yQOs93Cgg
        return True


def get_logger(
    name: str, provider: Optional[str] = None, level: int = logging.INFO
) -> logging.Logger:
    if provider is not None:
        name = "{}#{}".format(name, provider)
        fmt = "[%(levelname)s] {}: %(message)s".format(provider.upper())
    else:
        fmt = "[%(levelname)s] %(message)s"

    logger = logging.getLogger(name)

    if not logger.handlers:
        # Add stdout handler
        stdout_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(fmt)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)
        intercept_handler = InterceptHandler()
        logger.addHandler(intercept_handler)

        # # For now, run log all the time
        # # if enable_file_logger:
        # file_handler = logging.FileHandler("tui.log")
        # file_formatter = logging.Formatter(
        #     "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        # )
        # file_handler.setFormatter(file_formatter)
        # logger.addHandler(file_handler)

    # Set logging level
    logger.setLevel(level)
    logging.getLogger("requests.packages.urllib3.connectionpool").setLevel(
        logging.ERROR
    )

    for logger_name in LOGGERS:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler(level=level)]

    # Disable propagation to avoid conflict with Artifactory
    logger.propagate = False

    # set to true for async or multiprocessing logging
    logger.enqueue = True

    # Caution, may leak sensitive data in prod
    logger.diagnose = True

    logger.addFilter(LoopDetector())

    return logger


# SOURCE: https://github.com/jupiterbjy/CUIAudioPlayer/blob/dev_master/CUIAudioPlayer/LoggingConfigurator.py
def get_caller_stack_name(depth=1):
    """
    Gets the name of caller.
    :param depth: determine which scope to inspect, for nested usage.
    """
    return inspect.stack()[depth][3]


# SOURCE: https://github.com/jupiterbjy/CUIAudioPlayer/blob/dev_master/CUIAudioPlayer/LoggingConfigurator.py
def get_caller_stack_and_association(depth=1):
    stack_frame = inspect.stack()[depth][0]
    f_code_ref = stack_frame.f_code

    def get_reference_filter():
        for obj in gc.get_referrers(f_code_ref):
            try:
                if obj.__code__ is f_code_ref:  # checking identity
                    return obj
            except AttributeError:
                continue

    actual_function_ref = get_reference_filter()
    try:
        return actual_function_ref.__qualname__
    except AttributeError:
        return "<Module>"


# https://stackoverflow.com/questions/52715425


def log_caller():
    return f"<{get_caller_stack_name()}>"


def get_lm_from_tree(loggertree: LoggerModel, find_me: str) -> LoggerModel:
    if find_me == loggertree.name:
        LOGGER.debug("Found")
        return loggertree
    else:
        for ch in loggertree.children:
            LOGGER.debug(f"Looking in: {ch.name}")
            i = get_lm_from_tree(ch, find_me)
            if i:
                return i


def generate_tree() -> LoggerModel:
    # adapted from logging_tree package https://github.com/brandon-rhodes/logging_tree
    rootm = LoggerModel(
        name="root", level=logging.getLogger().getEffectiveLevel(), children=[]
    )
    nodesm = {}
    items = list(logging.root.manager.loggerDict.items())  # type: ignore
    items.sort()
    for name, loggeritem in items:
        if isinstance(loggeritem, logging.PlaceHolder):
            nodesm[name] = nodem = LoggerModel(name=name, children=[])
        else:
            nodesm[name] = nodem = LoggerModel(
                name=name, level=loggeritem.getEffectiveLevel(), children=[]
            )
        i = name.rfind(".", 0, len(name) - 1)  # same formula used in `logging`
        if i == -1:
            parentm = rootm
        else:
            parentm = nodesm[name[:i]]
        parentm.children.append(nodem)
    return rootm


# SMOKE-TESTS
if __name__ == "__main__":
    from logging_tree import printout

    LOGGER = get_logger(__name__, provider="Logger")

    def dump_logger_tree():
        rootm = generate_tree()
        LOGGER.debug(rootm)

    def dump_logger(logger_name: str):
        LOGGER.debug(f"getting logger {logger_name}")
        rootm = generate_tree()
        lm = get_lm_from_tree(rootm, logger_name)
        return lm

    logger.info("TESTING TESTING 1-2-3")
    printout()
