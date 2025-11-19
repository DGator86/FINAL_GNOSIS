"""Configuration module for Super Gnosis / DHPE v3."""

from config.config_models import AppConfig
from config.loader import load_config

__all__ = ["AppConfig", "load_config"]
