import re
import glob
import json

from pathlib import Path

# Path to config directory
CONFIG_DIR = Path(__file__).parent / "model_configs"

# Create path if it doesn't exist
CONFIG_DIR.mkdir(exist_ok=True)

# Default configuration as fallback (kept for reference and initial setup)
DEFAULT_CONFIG = {
    "ICON": "🤖",
    "TEMPERATURE": 0.7,
    "TOP_K": 40,
    "TOP_P": 0.9,
    "MIN_P": 0.02,
    "TYPICAL_P": 0.75,
    "NUM_CTX": 8192,
    "NUM_PREDICT": 256,
    "REPEAT_LAST_N": 64,
    "REPEAT_PENALTY": 1.21,
    "MIROSTAT": 0,
    "MIROSTAT_ETA": 0.10,
    "MIROSTAT_TAU": 4.0,
}

MODEL_CONFIGS = glob.glob(f"{CONFIG_DIR}/*.json")

def get_defaults(model: str)-> dict|None:
    if not model:
        return None
    filename = f"{CONFIG_DIR}/{model.split(":")[0]}.json"

    if filename in MODEL_CONFIGS:
        with open(filename, "r") as f:
            return json.load(f)
    else:
        with open(f"{CONFIG_DIR}/{filename}", "w") as f:
            json.dump(DEFAULT_CONFIG, f)
        return DEFAULT_CONFIG
