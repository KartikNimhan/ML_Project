import logging
import os
from datetime import datetime

# Define project root manually
PROJECT_ROOT = r"D:\MLOPs\ML_Project"

# Create log file name with timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Directory for logs (always outside src, inside project root)
logs_dir = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(logs_dir, exist_ok=True)

# Full path for the log file
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Test logging
if __name__ == "__main__":
    logging.info("Logging has started")
