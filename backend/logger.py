import logging
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

from backend.config import project_root

now = datetime.now()
date = now.strftime("%Y-%m-%d")

# Ensure 'logs' folder exists
log_dir = f"{project_root}/logs"
os.makedirs(log_dir, exist_ok=True)

# Log file path
log_file_path = f"{log_dir}/hotel_review_workflow_{date}.log"

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Log format
formatter = logging.Formatter(
    fmt="%(asctime)s | %(filename)s | %(module)s | line:%(lineno)d | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

file_handler = TimedRotatingFileHandler(
    filename=log_file_path,
    when="midnight",
    backupCount=0,  # Keep only the current day's log
    encoding="utf-8",
    utc=False,
)
file_handler.setFormatter(formatter)

# Stream Handler (console)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

# Add handlers to the logger, ensuring no duplicates
if not logger.handlers:
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
