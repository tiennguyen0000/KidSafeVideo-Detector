"""Configuration loader module."""
import os
import yaml
from typing import Dict, Any
from pathlib import Path


class Config:
    """Configuration loader for the application."""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            # Try multiple locations for config file
            possible_paths = [
                os.environ.get('CONFIG_PATH'),
                '/app/config/config.yaml',
                '/opt/airflow/config/config.yaml',
                'config/config.yaml',
            ]
            for path in possible_paths:
                if path and Path(path).exists():
                    config_path = path
                    break
            
            if config_path is None:
                config_path = '/app/config/config.yaml'
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports nested keys with dot notation)."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration."""
        return self._config
    
    @property
    def system(self) -> Dict[str, Any]:
        """Get system configuration."""
        return self._config.get('system', {})
    
    @property
    def data(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self._config.get('data', {})
    
    @property
    def models(self) -> Dict[str, Any]:
        """Get models configuration."""
        return self._config.get('models', {})
    
    @property
    def fusion(self) -> Dict[str, Any]:
        """Get fusion configuration."""
        return self._config.get('fusion', {})
    
    @property
    def training(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self._config.get('training', {})
    
    @property
    def preprocessing(self) -> Dict[str, Any]:
        """Get preprocessing configuration."""
        return self._config.get('preprocessing', {})
    
    @property
    def labels(self) -> Dict[str, Any]:
        """Get labels configuration."""
        return self._config.get('labels', {})


# Global config instance
config = Config()
