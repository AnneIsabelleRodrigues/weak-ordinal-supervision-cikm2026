import os
from dotenv import load_dotenv

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

load_dotenv()

DIR_RESULTS = os.getenv("DIR_RESULTS")

