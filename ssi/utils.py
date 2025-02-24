import os
import pdb
import sys
import traceback


def info_excepthook(type, value, tb):
    if hasattr(sys, "ps1") or not sys.stderr.isatty():
        # interactive mode or we don't have a tty-like device: call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        # NOT in interactive mode: print the exception then start the debugger in post-mortem mode
        traceback.print_exception(type, value, tb)
        pdb.post_mortem(tb)


def get_terminal_width(default_width: int = 120) -> int:
    try:
        TERMINAL_WIDTH = os.get_terminal_size().columns
    except OSError:
        TERMINAL_WIDTH = default_width
    return TERMINAL_WIDTH
