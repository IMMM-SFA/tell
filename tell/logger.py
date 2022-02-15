import os
import logging
import sys
import time


class Logger:
    """Initialize project-wide logger. The logger outputs to both stdout and a file.

    :param output_directory:                    Full path to the output directory where the log file is to be written;
                                                If no directory is passed, the file will not be written.
    :type output_directory:                     str

    """

    # output format for log string
    LOG_FORMAT_STRING = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # format for datetime string
    DATETIME_FORMAT = '%Y-%m-%d_%Hh%Mm%Ss'

    def __init__(self, output_directory=None):

        self.output_directory = output_directory

        # construct logfile name
        if output_directory is not None:
            self.logfile = os.path.join(output_directory, f"tell_logfile_{time.strftime(Logger.DATETIME_FORMAT)}.log")

    @property
    def log_format(self):
        """Generate log formatter."""

        return logging.Formatter(self.LOG_FORMAT_STRING)

    @property
    def logger(self):
        """Initialize logger as level info."""

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        return logger

    def initialize_logger(self):
        """Initialize logger to stdout and file."""

        # logger console handler
        self.console_handler()

        # logger file handler
        if self.output_directory is not None:
            self.file_handler()

    def console_handler(self):
        """Construct console handler."""

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self.log_format)
        self.logger.addHandler(console_handler)

    def file_handler(self):
        """Construct file handler."""

        file_handler = logging.FileHandler(self.logfile)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(self.log_format)
        self.logger.addHandler(file_handler)

    @staticmethod
    def close_logger():
        """Shutdown logger."""

        # Remove logging handlers
        logger = logging.getLogger()

        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        logging.shutdown()
