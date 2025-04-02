import glob
import json
from pathlib import Path


class Parameters:
    # Default configuration as fallback (kept for reference and initial setup)
    DEFAULT_CONFIG = {
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.9,
        "min_p": 0.02,
        "typical_p": 0.75,
        "num_ctx": 8192,
        "num_predict": 256,
        "repeat_last_n": 64,
        "repeat_penalty": 1.21,
        "mirostat": 0,
        "mirostat_eta": 0.10,
        "mirostat_tau": 4.0,
        "icon": "🤖",
    }

    def __init__(
        self,
        config_dir: Path = Path(__file__).parent / "model_configs",
    ) -> None:
        # Path to model parameters
        self.CONFIG_DIR = config_dir

        # Create path if it doesn't exist
        self.CONFIG_DIR.mkdir(exist_ok=True)

        # Get all Files
        self.MODEL_CONFIGS = glob.glob(f"{self.CONFIG_DIR}/*.json")

    def get_defaults(self, model: str) -> dict | None:
        if not model:
            return None
        filename = f"{self.CONFIG_DIR}/{model.split(':')[0]}.json"

        if filename in self.MODEL_CONFIGS:
            # Return model parameters
            with open(filename, "r") as f:
                return json.load(f)
        else:
            # Create a new Model Config File from default
            with open(filename, "w") as f:
                json.dump(self.DEFAULT_CONFIG, f)
            return self.DEFAULT_CONFIG

    def update_defaults(self, model: str, ollama_params: dict) -> bool:
        if not model:
            return False
        filename = f"{self.CONFIG_DIR}/{model.split(':')[0]}.json"
        with open(filename, "r") as fo:
            old_data = json.load(fo)

        ollama_params["icon"] = old_data["icon"]
        with open(filename, "w") as fn:
            json.dump(ollama_params, fn)
        return True
