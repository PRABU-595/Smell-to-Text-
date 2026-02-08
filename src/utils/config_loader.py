"""
Configuration loader utilities
"""
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import os


class ConfigLoader:
    """Load and manage configuration files."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.configs = {}
    
    def load(self, name: str) -> Dict[str, Any]:
        """Load configuration by name."""
        yaml_path = self.config_dir / f"{name}.yaml"
        json_path = self.config_dir / f"{name}.json"
        
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
        elif json_path.exists():
            with open(json_path, 'r') as f:
                config = json.load(f)
        else:
            raise FileNotFoundError(f"Config {name} not found")
        
        self.configs[name] = config
        return config
    
    def load_all(self) -> Dict[str, Dict]:
        """Load all configs in directory."""
        for path in self.config_dir.glob("*.yaml"):
            self.load(path.stem)
        for path in self.config_dir.glob("*.json"):
            if path.stem not in self.configs:
                self.load(path.stem)
        return self.configs
    
    def get(self, name: str, key: str, default: Any = None) -> Any:
        """Get specific config value."""
        if name not in self.configs:
            self.load(name)
        return self.configs.get(name, {}).get(key, default)
    
    def merge_with_args(self, config: Dict, args: Any) -> Dict:
        """Merge config with command line arguments."""
        merged = config.copy()
        for key, value in vars(args).items():
            if value is not None:
                merged[key] = value
        return merged
    
    def save(self, name: str, config: Dict):
        """Save configuration."""
        path = self.config_dir / f"{name}.yaml"
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)


def get_default_config() -> Dict:
    """Get default configuration."""
    return {
        'model': {'name': 'bert-base-uncased', 'max_length': 128, 'num_chemicals': 300},
        'training': {'batch_size': 16, 'epochs': 20, 'learning_rate': 2e-5},
        'evaluation': {'top_k': [1, 3, 5, 10]}
    }


def override_from_env(config: Dict, prefix: str = "STM_") -> Dict:
    """Override config from environment variables."""
    for key, value in os.environ.items():
        if key.startswith(prefix):
            config_key = key[len(prefix):].lower()
            try:
                config[config_key] = yaml.safe_load(value)
            except:
                config[config_key] = value
    return config
