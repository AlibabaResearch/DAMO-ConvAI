from __future__ import absolute_import, division, unicode_literals


import logging
from pathlib import Path


import logging
import multiprocessing
import threading

try:
    from queue import Empty
except ImportError:  # Python 2.
    from Queue import Empty  # type: ignore[no-redef]


__version__ = "0.3.4"


def setup_logger(folder_path, log_file_name="logger.log", console_output=False, logger_name="task"):
    dir_root = Path(folder_path)
    full_path = dir_root.joinpath(log_file_name)
    # print("File: ", full_path)

    already_exist = Path(full_path).exists()

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s| %(message)s", "%m-%d|%H:%M:%S")

    file_hdl = logging.FileHandler(full_path)
    file_hdl.setFormatter(formatter)

    logger.addHandler(file_hdl)

    if console_output:
        console_hdl = logging.StreamHandler()
        console_hdl.setFormatter(formatter)
        logger.addHandler(console_hdl)

    logger.info("")
    logger.info("-*" * 30)
    logger.info("Logger ready")
    if already_exist:
        logger.info("")
        logger.info("")
        logger.info(f">>>>> Logger file {full_path} already exist, append to it. <<<<<")
        logger.info("")
        logger.info("")


def setup_simple_logger():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s| %(message)s", "%m-%d|%H:%M:%S")

    console_hdl = logging.StreamHandler()
    console_hdl.setFormatter(formatter)
    root_logger.addHandler(console_hdl)


def tabular_pretty_print(grid):
    lens = [max(map(len, col)) for col in zip(*grid)]

    fmt = " | ".join("{{:{}}}".format(x) for x in lens)
    table = [fmt.format(*row) for row in grid]

    sep = ["~" * len(table[0])]
    table = sep + table + sep

    res = []
    for idx, line in enumerate(table):
        if idx == 0 or idx == len(table) - 1:
            ps = "* {} *".format(line)
        else:
            ps = "| {} |".format(line)
        res.append(ps)
    return res


def fmt_float(num, d=4):
    fmt_string = "{{:.{}f}}".format(d)
    return fmt_string.format(num)


def install_mp_handler(logger=None):
    """Wraps the handlers in the given Logger with an MultiProcessingHandler.
    :param logger: whose handlers to wrap. By default, the root logger.
    """
    if logger is None:
        logger = logging.getLogger()

    for i, orig_handler in enumerate(list(logger.handlers)):
        handler = MultiProcessingHandler("mp-handler-{0}".format(i), sub_handler=orig_handler)

        logger.removeHandler(orig_handler)
        logger.addHandler(handler)


def uninstall_mp_handler(logger=None):
    """Unwraps the handlers in the given Logger from a MultiProcessingHandler wrapper
    :param logger: whose handlers to unwrap. By default, the root logger.
    """
    if logger is None:
        logger = logging.getLogger()

    for handler in logger.handlers:
        if isinstance(handler, MultiProcessingHandler):
            orig_handler = handler.sub_handler
            logger.removeHandler(handler)
            logger.addHandler(orig_handler)


class MultiProcessingHandler(logging.Handler):
    def __init__(self, name, sub_handler=None):
        super(MultiProcessingHandler, self).__init__()

        if sub_handler is None:
            sub_handler = logging.StreamHandler()
        self.sub_handler = sub_handler

        self.setLevel(self.sub_handler.level)
        self.setFormatter(self.sub_handler.formatter)
        self.filters = self.sub_handler.filters

        self.queue = multiprocessing.Queue(-1)
        self._is_closed = False
        # The thread handles receiving records asynchronously.
        self._receive_thread = threading.Thread(target=self._receive, name=name)
        self._receive_thread.daemon = True
        self._receive_thread.start()

    def setFormatter(self, fmt):
        super(MultiProcessingHandler, self).setFormatter(fmt)
        self.sub_handler.setFormatter(fmt)

    def _receive(self):
        while True:
            try:
                if self._is_closed and self.queue.empty():
                    break

                record = self.queue.get(timeout=0.2)
                self.sub_handler.emit(record)
            except (KeyboardInterrupt, SystemExit):
                raise
            except (EOFError, OSError):
                break  # The queue was closed by child?
            except Empty:
                pass  # This periodically checks if the logger is closed.
            except:
                from sys import stderr
                from traceback import print_exc

                print_exc(file=stderr)
                raise

        self.queue.close()
        self.queue.join_thread()

    def _send(self, s):
        self.queue.put_nowait(s)

    def _format_record(self, record):
        # ensure that exc_info and args
        # have been stringified. Removes any chance of
        # unpickleable things inside and possibly reduces
        # message size sent over the pipe.
        if record.args:
            record.msg = record.msg % record.args
            record.args = None
        if record.exc_info:
            self.format(record)
            record.exc_info = None

        return record

    def emit(self, record):
        try:
            s = self._format_record(record)
            self._send(s)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

    def close(self):
        if not self._is_closed:
            self._is_closed = True
            self._receive_thread.join(5.0)  # Waits for receive queue to empty.

            self.sub_handler.close()
            super(MultiProcessingHandler, self).close()
