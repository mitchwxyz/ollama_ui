import glob
import json
from pathlib import Path
from typing import Any, ClassVar

from ollama import Options


class Parameters:
    """Handles model parameter configuration for Ollama models.

    This class manages loading, saving, and updating model parameters,
    with fallback to sensible defaults when no configuration exists.
    """

    # Default configuration for initial setup
    DEFAULT_CONFIG: ClassVar[dict] = {
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.9,
        "typical_p": 0.4,
        "num_ctx": 8000,
        "num_predict": 256,
        "repeat_last_n": 128,
        "repeat_penalty": 1.00,
        "mirostat": 0,
        "mirostat_eta": 0.10,
        "mirostat_tau": 4.0,
        "icon": "🤖",
    }

    def __init__(
        self,
        config_dir: Path = Path(__file__).parent / "model_configs",
    ) -> None:
        """Initialize the Parameters manager.

        Args:
            config_dir: Directory path where model configurations are stored.

        """
        # Path to model parameters
        self.config_dir = config_dir

        # Create config directory if it doesn't exist
        self.config_dir.mkdir(exist_ok=True)

        # Get all existing configuration files
        self.model_configs = glob.glob(f"{self.config_dir}/*.json")

    def get_defaults(self, model: str) -> dict[str, Any]:
        """Retrieve default parameters for a specific model.

        If no configuration exists for the model, creates one with default values.

        Args:
            model: Name of the model to get parameters for

        Returns:
            Dictionary of model parameters or None if model name is empty

        """
        if not model:
            raise ValueError("No model defined.")

        model_name = model.split(":")[0]
        config_path = f"{self.config_dir}/{model_name}.json"

        if config_path in self.model_configs:
            # Return existing model parameters
            with open(config_path) as config_file:
                return json.load(config_file)
        else:
            # Create a new model config file from defaults
            with open(config_path, "w") as config_file:
                json.dump(self.DEFAULT_CONFIG, config_file)
            return self.DEFAULT_CONFIG

    def update_defaults(self, model: str, ollama_params: Options) -> bool:
        """Update the default parameters for a specific model.

        Args:
            model: Name of the model to update parameters for
            ollama_params: New parameters from Ollama to save

        Returns:
            True if update was successful, False otherwise

        """
        if not model:
            return False

        model_name = model.split(":")[0]
        config_path = f"{self.config_dir}/{model_name}.json"

        # Read existing config to preserve icon
        with open(config_path) as config_file:
            existing_config = json.load(config_file)

        # Get parameters from Ollama options
        params = ollama_params.dict(exclude_unset=True, exclude_none=True)

        # Preserve the icon from existing configuration
        params["icon"] = existing_config["icon"]

        # Write updated configuration
        with open(config_path, "w") as config_file:
            json.dump(params, config_file)

        return True
