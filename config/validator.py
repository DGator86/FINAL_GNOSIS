"""Configuration validation module for GNOSIS trading system.

Validates environment variables, configuration files, and system settings
before system startup to prevent runtime errors.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import yaml
from loguru import logger


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""


class ConfigValidator:
    """Validates system configuration before startup."""
    
    def __init__(self, strict: bool = True):
        """
        Initialize configuration validator.
        
        Args:
            strict: If True, fail on warnings. If False, only fail on errors.
        """
        self.strict = strict
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_all(self) -> Tuple[bool, List[str], List[str]]:
        """
        Run all validation checks.
        
        Returns:
            Tuple of (success, errors, warnings)
        """
        logger.info("Starting configuration validation...")
        
        self.validate_environment_variables()
        self.validate_config_files()
        self.validate_trading_parameters()
        self.validate_directories()
        
        success = len(self.errors) == 0 and (not self.strict or len(self.warnings) == 0)
        
        if success:
            logger.info("✅ Configuration validation passed!")
        else:
            logger.error(f"❌ Configuration validation failed with {len(self.errors)} errors and {len(self.warnings)} warnings")
        
        return success, self.errors, self.warnings
    
    def validate_environment_variables(self) -> None:
        """Validate required environment variables."""
        logger.info("Validating environment variables...")
        
        # Required variables
        required_vars = {
            "ALPACA_API_KEY": "Alpaca API key for trading",
            "ALPACA_SECRET_KEY": "Alpaca secret key for trading",
        }
        
        for var, description in required_vars.items():
            value = os.getenv(var)
            if not value:
                self.errors.append(f"Missing required environment variable: {var} ({description})")
            elif value.startswith("your_") or value == "":
                self.errors.append(f"Environment variable {var} has placeholder value. Set a real API key.")
        
        # Optional but recommended
        optional_vars = {
            "UNUSUAL_WHALES_API_TOKEN": "Unusual Whales API for options data",
            "UNUSUAL_WHALES_API_KEY": "Unusual Whales API (legacy env var)",
            "LOG_LEVEL": "Logging level",
        }
        
        for var, description in optional_vars.items():
            value = os.getenv(var)
            if not value:
                self.warnings.append(f"Optional environment variable not set: {var} ({description})")
    
    def validate_config_files(self) -> None:
        """Validate configuration files exist and are valid YAML."""
        logger.info("Validating configuration files...")
        
        config_files = {
            "config/config.yaml": "Main system configuration",
            "config/watchlist.yaml": "Trading watchlist",
        }
        
        for file_path, description in config_files.items():
            full_path = Path(file_path)
            
            if not full_path.exists():
                self.errors.append(f"Missing configuration file: {file_path} ({description})")
                continue
            
            try:
                with open(full_path, 'r') as f:
                    config = yaml.safe_load(f)
                    if config is None:
                        self.errors.append(f"Configuration file is empty: {file_path}")
            except yaml.YAMLError as e:
                self.errors.append(f"Invalid YAML in {file_path}: {str(e)}")
            except Exception as e:
                self.errors.append(f"Error reading {file_path}: {str(e)}")
    
    def validate_trading_parameters(self) -> None:
        """Validate trading parameters are within safe ranges."""
        logger.info("Validating trading parameters...")
        
        # Load config
        try:
            with open("config/config.yaml", 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            # Already reported in validate_config_files
            return
        
        # Validate agent trade settings
        if "agents" in config and "trade" in config["agents"]:
            trade_config = config["agents"]["trade"]
            
            risk_per_trade = trade_config.get("risk_per_trade", 0.02)
            if risk_per_trade > 0.10:  # More than 10% risk per trade
                self.warnings.append(
                    f"High risk per trade: {risk_per_trade:.1%}. Consider reducing below 10%."
                )
            
            max_position = trade_config.get("max_position_size", 10000)
            if max_position > 50000:
                self.warnings.append(
                    f"Large max position size: ${max_position:,.0f}. Ensure this is intentional."
                )
    
    def validate_directories(self) -> None:
        """Validate required directories exist or can be created."""
        logger.info("Validating directories...")
        
        required_dirs = ["logs", "config"]
        
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created directory: {dir_name}")
                except Exception as e:
                    self.errors.append(f"Cannot create directory {dir_name}: {str(e)}")


def validate_configuration(strict: bool = True) -> None:
    """
    Validate system configuration and raise exception if validation fails.
    
    Args:
        strict: If True, fail on warnings. If False, only fail on errors.
        
    Raises:
        ConfigurationError: If validation fails
    """
    validator = ConfigValidator(strict=strict)
    success, errors, warnings = validator.validate_all()
    
    if not success:
        error_msg = "Configuration validation failed:\n"
        
        if errors:
            error_msg += "\nERRORS:\n"
            for error in errors:
                error_msg += f"  ❌ {error}\n"
        
        if warnings:
            error_msg += "\nWARNINGS:\n"
            for warning in warnings:
                error_msg += f"  ⚠️  {warning}\n"
        
        raise ConfigurationError(error_msg)
    
    if warnings:
        logger.warning(f"Configuration validation passed with {len(warnings)} warnings:")
        for warning in warnings:
            logger.warning(f"  ⚠️  {warning}")


if __name__ == "__main__":
    # Run validation
    try:
        validate_configuration(strict=False)
        print("✅ All configuration checks passed!")
    except ConfigurationError as e:
        print(str(e))
        exit(1)
