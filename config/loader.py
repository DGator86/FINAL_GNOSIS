"""Configuration loader."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger

from config.config_models import AppConfig


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load application configuration.
    
    Args:
        config_path: Path to YAML config file. If None, uses default or environment variable.
        
    Returns:
        AppConfig instance
    """
    if config_path is None:
        config_path = os.getenv("DHPE_CONFIG_PATH", "config/config.yaml")
    
    config_file = Path(config_path)
    
    if config_file.exists():
        logger.info(f"Loading configuration from {config_file}")
        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f) or {}
        return AppConfig(**config_data)
    else:
        logger.warning(f"Config file {config_file} not found, using defaults")
        return AppConfig()
