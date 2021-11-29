import sys
import logging
from logging.handlers import TimedRotatingFileHandler
from PyQt6 import QtCore, QtWidgets


class QTextEditLogger(logging.Handler):
    def __init__(self, parent):
        super().__init__()
        self.widget = QtWidgets.QPlainTextEdit(parent)
        self.widget.setReadOnly(True)

    def emit(self, record):
        msg = self.format(record)
        self.widget.appendPlainText(msg)


def get_logger(qtlog_handler: QTextEditLogger,
               file_handler_level: logging = logging.DEBUG,
               qtlog_handler_level: logging = logging.INFO,
               file_formatter: str = "%(asctime)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s",
               out_name: str = "log_file.log",
               qtlog_formatter: str = "%(levelname)s - %(message)s") -> logging.Logger:
    """

    Args:
        text:
        qtlog_handler:
        level:
        file_formatter:
        out_name:
        qtlog_formatter:

    Returns:

    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    qtlog_handler.setLevel(qtlog_handler_level)
    qtlog_handler.setFormatter(logging.Formatter(qtlog_formatter))
    logger.addHandler(qtlog_handler)

    file_logger = TimedRotatingFileHandler(out_name, when='D', interval=1, backupCount=5)
    file_logger.suffix = '%Y_%m_%d.log'
    file_logger.setLevel(file_handler_level)
    file_logger.setFormatter(logging.Formatter(file_formatter))
    logger.addHandler(file_logger)

    return logger
