# -*- coding: utf-8 -*-
"""TODO"""

import os
from datetime import datetime
from enum import Enum
from functools import total_ordering


@total_ordering
class LogMessageType(Enum):
    VERBOSE = 1
    INFO = 2
    NOTIFICATION = 3
    SETTINGS = 4
    PROGRESS = 5
    WARNING = 6
    ERROR = 7

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value


class Logger:
    """TODO"""

    def __init__(self, filename, rank, console_out=LogMessageType.PROGRESS,
                 file_out=LogMessageType.NOTIFICATION):
        """TODO"""
        self.rank = rank
        self.file_out = file_out
        self.console_out = console_out

        self._filename = filename

        self._last_message = ""

        try:
            os.makedirs(os.path.dirname(self._filename), exist_ok=True)
            self._outfile = open(self._filename, 'w', 1)
        except OSError:
            print("Can't open logger output file.")
            raise

        self.notification("Logger started on Rank %d" % self.rank)

    def msg(self, message, msg_type=LogMessageType.NOTIFICATION,
            global_msg=False):
        """TODO"""
        text = ""
        if msg_type == LogMessageType.WARNING:
            text += "!! WARNING: "
        elif msg_type == LogMessageType.ERROR:
            text += "!! ERROR: "

        text += message
        if (msg_type >= self.console_out
                and not global_msg
                or (global_msg and self.rank == 0)):
            if self._last_message != message:
                if global_msg:
                    print(text)
                else:
                    print("On rank %d: %s" % (self.rank, text))

        text = datetime.now().strftime("%Y-%m-%d %H:%M:%S: ") + text
        if msg_type >= self.file_out:
            self._outfile.write(text + '\n')

        self._last_message = message

    def notification(self, message):
        """TODO"""
        self.msg(message, LogMessageType.NOTIFICATION)

    def progress(self, message):
        """TODO"""
        self.msg(message, LogMessageType.PROGRESS, True)

    def warning(self, message):
        """TODO"""
        self.msg(message, LogMessageType.WARNING)

    def error(self, message):
        """TODO"""
        self.msg(message, LogMessageType.ERROR)

    def __del__(self):
        """TODO"""
        try:
            self._outfile.close()
        except AttributeError:
            pass
