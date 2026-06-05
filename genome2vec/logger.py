import os
import logging
from datetime import datetime


class Logger():
    def __init__(self, config):
        os.makedirs(config['logging_dir'], exist_ok=True)

        # Name log file with timestamp
        log_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(config['logging_dir'], f"{log_time}.log")

        # instantiate logger for class, unique since we use the time
        self._logger = logging.getLogger(f"{__name__}.{log_time}")
        self._logger.setLevel(logging.INFO)

        # set log format
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler = logging.FileHandler(self.log_path)
        handler.setFormatter(formatter)

        # Attach handler to this logger only
        self._logger.addHandler(handler)

        # Example log line
        self._logger.info("Logger initialized")

    def __getattr__(self, name):
        return getattr(self._logger, name)
