### bsf.c 2024.05.21 stable version 1.0

import functools
import os, os.path as osp
from termcolor import colored
import logging    
import atexit
import sys

class _ColorfulFormatter(logging.Formatter):
    ERROR_PREFIX = colored("ERROR", "red", attrs=["blink", "underline"])
    WARN_PREFIX  = colored("WARNING", "red", attrs=["blink"])
    def __init__(self, root_name="", brief_name=None, *args, **kwargs):
        self._root_name = root_name + "."
        if brief_name is not None:
            self._brief_name = brief_name + "."
        else:
            self._brief_name = None
        super().__init__(*args, **kwargs)

    def formatMessage(self, record):
        if self._brief_name is not None:
            record.name = record.name.replace(self._root_name, self._brief_name)
        log = super().formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = self.WARN_PREFIX
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = self.ERROR_PREFIX
        else:
            return log
        return prefix + " " + log


@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers
def setup_logger(
    output=None, color=True, name="bff", brief_name=None, stdout=True
):
    """ usage setup_logger("/path/to/log/file", name="bff.log")
    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger
        brief_name (str): an abbreviation of the module, to avoid long names in logs.
            Set to "" to not log the root module in logs.
            By default, will abbreviate "detectron2" to "d2" and leave other
            modules unchanged.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    if not hasattr(logger, "_setuped"):
        logger._setuped = True
    if stdout is True:
        # stdout logging: master only
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        if color:
            fmt = colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s"
            formatter = _ColorfulFormatter(
                root_name=name,
                brief_name=brief_name,
                fmt = fmt,
                datefmt="%m/%d %H:%M:%S",
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = osp.join(output, "log.txt")

        parent_dir = osp.dirname(filename)
        if len(parent_dir) > 0:
            os.makedirs(parent_dir, exist_ok=True)

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)


    return logger

_opened_stream = []
def clear_streams():
    global _opened_stream
    for s in _opened_stream:
        try:
            s.close()
        except:
            pass

@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    f = open(filename, "a", encoding="utf-8")
    
    global _opened_stream
    if len(_opened_stream) == 0:
        atexit.register(clear_streams)

    _opened_stream.append(f)
    return f

_LOG_DIR = osp.abspath(osp.expanduser("~/.bfp/.logs"))
os.makedirs(_LOG_DIR, exist_ok=True)

formaterStr = "%(asctime)s %(levelname)s:  %(message)s"

# _customLogger = {}

def createLogger(name: str, savefile = True, stream = False, 
    level = logging.INFO, basedir = None,  **kwargs):
    """
    @deprecated use setup_logger instead
    create logger
    Args: 
        name : suffix will be appended, for example, `test` will be `test.log`
        basedir: default is `~/.logs`

    kwargs:
        timeformat: default is "%Y-%m-%d %H:%M:%S" 
    :: logger_prefix deprecated ::

    """
    raise ValueError("Deperacted, use setup_logger instead")
    global _customLogger
    if name in _customLogger:
        return _customLogger[name]
    logger = logging.Logger(name)
    _customLogger[name] = logger
    tformat = kwargs.get("timeformat", "%Y-%m-%d %H:%M:%S")
    _formater = logging.Formatter(formaterStr, tformat)
    print("Warning, this function has been deprecated, use setup_logger instead")

    if savefile:
        if basedir is None:
            basedir = _LOG_DIR
        elif type(basedir) == str:
            basedir = osp.abspath(basedir)
        os.makedirs(basedir, exist_ok=True)
        if not name.endswith(".log"):
            name = name + ".log"
        log_file = osp.join(basedir, name)
        fh = logging.FileHandler(log_file, encoding="utf8")
        fh.setFormatter(_formater)
        fh.setLevel(level)
        logger.addHandler(fh)
        # print("add file handler", log_file)
    if stream:
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(_formater)
        logger.addHandler(sh)
    logger.setLevel(level)
    return logger

def changeLoggerLevel(logger, level):
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)