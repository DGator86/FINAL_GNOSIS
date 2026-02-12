#!/usr/bin/env python3
"""
GNOSIS Trading System - Full Model Training

Comprehensive training with real market data from Alpaca.
Trains RL Agent and Transformer with optimal hyperparameters.

Usage:
    python scripts/full_train.py
"""

import os
import sys
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gnosis.full_train')

# Training Configuration - Optimized for best results
TRAINING_CONFIG = {
    'symbols': ['SPY', 'QQQ', 'AAPL', 'NVDA', 'MSFT', 'AMD', 'TSLA', 'AMZN', 'META', 'GOOGL'],
    'history_days': 365 * 2,  # 2 years of data
    'rl': {
        'num_episodes': 100,  # Reduced for faster training
        'max_steps_per_episode': 200,
        'eval_frequency': 20,
        'initial_capital': 100000,
        'transaction_cost': 0.001,
        'early_stopping_patience': 30,
    },
    'transformer': {
        'epochs': 50,  # Reduced for faster training
        'd_model': 64,
        'n_heads': 4,
        'n_layers': 2,
        'sequence_length': 30,
        'batch_size': 32,
        'learning_rate': 0.001,
        'dropout': 0.1,
        'early_stopping_patience': 10,
    }
}


class AlpacaDataFetcher:
    """Fetch historical data from Alpaca."""
    
    def __init__(self):
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        self.base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets/v2')
        self.data_url = 'https://data.alpaca.markets/v2'
        
    async def fetch_bars(self, symbol: str, days: int = 365) -> Optional[np.ndarray]:
        """Fetch OHLCV bars for a symbol."""
        import aiohttp
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        url = f"{self.data_url}/stocks/{symbol}/bars"
        params = {
            'start': start_date.strftime('%Y-%m-%dT00:00:00Z'),
            'end': end_date.strftime('%Y-%m-%dT00:00:00Z'),
            'timeframe': '1Day',
            'limit': 10000
        }
        headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        bars = data.get('bars', [])
                        
                        if bars:
                            # Convert to numpy array [open, high, low, close, volume]
                            arr = np.array([
                                [b['o'], b['h'], b['l'], b['c'], b['v']]
                                for b in bars
                            ])
                            logger.info(f"  {symbol}: {len(arr)} bars fetched")
                            return arr
                        else:
                            logger.warning(f"  {symbol}: No bars returned")
                            return None
                    else:
                        error = await resp.text()
                        logger.error(f"  {symbol}: API error {resp.status} - {error[:100]}")
                        return None
        except Exception as e:
            logger.error(f"  {symbol}: Fetch failed - {e}")
            return None
    
    async def fetch_multiple(self, symbols: List[str], days: int = 365) -> Dict[str, np.ndarray]:
        """Fetch data for multiple symbols."""
        logger.info(f"Fetching {days} days of data for {len(symbols)} symbols...")
        
        results = {}
        for symbol in symbols:
            data = await self.fetch_bars(symbol, days)
            if data is not None and len(data) >= 100:
                results[symbol] = data
            await asyncio.sleep(0.2)  # Rate limit
        
        logger.info(f"Successfully fetched data for {len(results)} symbols")
        return results


class FullTrainer:
    """Comprehensive model trainer."""
    
    def __init__(self, config: dict = None):
        self.config = config or TRAINING_CONFIG
        self.models_dir = Path('models/trained')
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = Path('checkpoints')
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        self.fetcher = AlpacaDataFetcher()
        self.training_data = {}
        self.results = {}
        
    async def fetch_training_data(self):
        """Fetch all training data from Alpaca."""
        logger.info("=" * 60)
        logger.info("FETCHING TRAINING DATA FROM ALPACA")
        logger.info("=" * 60)
        
        self.training_data = await self.fetcher.fetch_multiple(
            self.config['symbols'],
            self.config['history_days']
        )
        
        if not self.training_data:
            logger.warning("No data fetched, generating synthetic data...")
            self.training_data = self._generate_synthetic_data()
        
        # Combine all data for training
        all_data = []
        for symbol, data in self.training_data.items():
            all_data.append(data)
        
        if all_data:
            self.combined_data = np.vstack(all_data)
            logger.info(f"Combined training data: {self.combined_data.shape}")
        else:
            self.combined_data = self._generate_synthetic_data()['SYNTHETIC']
        
        return self.training_data
    
    def _generate_synthetic_data(self, n_samples: int = 2000) -> Dict[str, np.ndarray]:
        """Generate synthetic market data as fallback."""
        logger.info("Generating synthetic market data...")
        
        data = {}
        for symbol in self.config['symbols'][:3]:
            price = 100.0 + np.random.uniform(-20, 50)
            prices = []
            
            for _ in range(n_samples):
                # Random walk with trend and mean reversion
                trend = 0.0002  # Slight upward bias
                volatility = 0.02
                mean_reversion = 0.01
                
                change = (
                    trend + 
                    volatility * np.random.normal() +
                    mean_reversion * (100 - price) / 100
                )
                
                open_p = price
                high_p = price * (1 + abs(np.random.normal(0, 0.01)))
                low_p = price * (1 - abs(np.random.normal(0, 0.01)))
                close_p = price * (1 + change)
                volume = np.random.uniform(1e6, 1e8)
                
                prices.append([open_p, high_p, low_p, close_p, volume])
                price = close_p
            
            data[symbol] = np.array(prices)
            logger.info(f"  Generated {n_samples} synthetic bars for {symbol}")
        
        return data
    
    def train_rl_agent(self) -> dict:
        """Train the RL trading agent."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("TRAINING RL TRADING AGENT")
        logger.info("=" * 60)
        
        from ml.training.rl_trainer import RLTrainer, RLTrainingConfig
        
        rl_config = self.config['rl']
        
        config = RLTrainingConfig(
            num_episodes=rl_config['num_episodes'],
            max_steps_per_episode=min(rl_config['max_steps_per_episode'], len(self.combined_data) - 1),
            eval_frequency=rl_config['eval_frequency'],
            log_frequency=rl_config['eval_frequency'],
            initial_capital=rl_config['initial_capital'],
            transaction_cost=rl_config['transaction_cost'],
            early_stopping_patience=rl_config['early_stopping_patience'],
            checkpoint_dir=str(self.checkpoints_dir / 'rl'),
        )
        
        logger.info(f"Configuration:")
        logger.info(f"  Episodes: {config.num_episodes}")
        logger.info(f"  Max Steps: {config.max_steps_per_episode}")
        logger.info(f"  Initial Capital: ${config.initial_capital:,}")
        logger.info(f"  Eval Frequency: {config.eval_frequency}")
        logger.info(f"  Early Stopping Patience: {config.early_stopping_patience}")
        logger.info("")
        
        trainer = RLTrainer(config)
        
        start_time = datetime.now()
        # Convert combined data to list of episodes
        # Each episode is a list of prices
        episode_length = config.max_steps_per_episode + 50
        price_data = []
        
        # Split data into episodes
        for i in range(0, len(self.combined_data) - episode_length, episode_length // 2):
            episode = self.combined_data[i:i+episode_length, 3].tolist()  # Close prices
            price_data.append(episode)
        
        if len(price_data) < 10:
            # Generate more episodes from the data by using different starting points
            for _ in range(20):
                start_idx = np.random.randint(0, len(self.combined_data) - episode_length)
                episode = self.combined_data[start_idx:start_idx+episode_length, 3].tolist()
                price_data.append(episode)
        
        logger.info(f"  Generated {len(price_data)} training episodes")
        
        result = trainer.train(price_data=price_data)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Save model
        model_path = self.models_dir / f"rl_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        trainer.save_model(str(model_path))
        
        # Also save as 'latest' for easy loading
        latest_path = self.models_dir / "rl_agent_latest.pkl"
        trainer.save_model(str(latest_path))
        
        result['model_path'] = str(model_path)
        result['training_time_seconds'] = training_time
        
        logger.info("")
        logger.info(f"RL Agent Training Complete!")
        logger.info(f"  Episodes: {result.get('episodes_completed', 0)}")
        logger.info(f"  Best Reward: {result.get('best_reward', 0):.2f}")
        logger.info(f"  Final Portfolio: ${result.get('final_portfolio_value', 0):,.2f}")
        logger.info(f"  Training Time: {training_time:.1f}s")
        logger.info(f"  Model Saved: {model_path}")
        
        self.results['rl_agent'] = result
        return result
    
    def train_transformer(self) -> dict:
        """Train the Transformer price predictor."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("TRAINING TRANSFORMER PRICE PREDICTOR")
        logger.info("=" * 60)
        
        from ml.training.transformer_trainer import (
            TransformerTrainer, 
            TransformerTrainingConfig,
            PredictionHorizon
        )
        
        tf_config = self.config['transformer']
        
        config = TransformerTrainingConfig(
            epochs=tf_config['epochs'],
            d_model=tf_config['d_model'],
            n_heads=tf_config['n_heads'],
            n_layers=tf_config['n_layers'],
            sequence_length=tf_config['sequence_length'],
            batch_size=tf_config['batch_size'],
            learning_rate=tf_config['learning_rate'],
            dropout=tf_config['dropout'],
            early_stopping_patience=tf_config['early_stopping_patience'],
        )
        
        logger.info(f"Configuration:")
        logger.info(f"  Epochs: {config.epochs}")
        logger.info(f"  Model Dimension: {config.d_model}")
        logger.info(f"  Attention Heads: {config.n_heads}")
        logger.info(f"  Layers: {config.n_layers}")
        logger.info(f"  Sequence Length: {config.sequence_length}")
        logger.info(f"  Learning Rate: {config.learning_rate}")
        logger.info("")
        
        trainer = TransformerTrainer(config)
        
        # Prepare data
        feature_set = trainer.prepare_data(
            prices=self.combined_data,
            horizon=PredictionHorizon.HOUR_1
        )
        
        start_time = datetime.now()
        result = trainer.train(feature_set)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Save model
        model_path = self.models_dir / f"transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        trainer.save_model(str(model_path))
        
        # Save as latest
        latest_path = self.models_dir / "transformer_latest.pkl"
        trainer.save_model(str(latest_path))
        
        result_dict = {
            'model_path': str(model_path),
            'training_completed': result.training_completed,
            'final_loss': result.final_loss,
            'best_loss': result.best_loss,
            'epochs_completed': len(result.training_history),
            'training_time_seconds': training_time,
            'direction_accuracy': result.final_metrics.get('direction_accuracy', 0) if result.final_metrics else 0,
        }
        
        logger.info("")
        logger.info(f"Transformer Training Complete!")
        logger.info(f"  Epochs: {result_dict['epochs_completed']}")
        logger.info(f"  Final Loss: {result_dict['final_loss']:.6f}")
        logger.info(f"  Best Loss: {result_dict['best_loss']:.6f}")
        logger.info(f"  Direction Accuracy: {result_dict['direction_accuracy']*100:.1f}%")
        logger.info(f"  Training Time: {training_time:.1f}s")
        logger.info(f"  Model Saved: {model_path}")
        
        self.results['transformer'] = result_dict
        return result_dict
    
    def train_for_multiple_horizons(self):
        """Train Transformer models for different prediction horizons."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("TRAINING MULTI-HORIZON PREDICTORS")
        logger.info("=" * 60)
        
        from ml.training.transformer_trainer import (
            TransformerTrainer, 
            TransformerTrainingConfig,
            PredictionHorizon
        )
        
        horizons = [
            (PredictionHorizon.MIN_15, 'Transformer-15min'),
            (PredictionHorizon.HOUR_1, 'Transformer-1hr'),
            (PredictionHorizon.HOUR_4, 'Transformer-4hr'),
        ]
        
        for horizon, name in horizons:
            logger.info(f"\nTraining {name}...")
            
            config = TransformerTrainingConfig(
                epochs=50,  # Fewer epochs for each horizon
                d_model=64,
                n_heads=4,
                n_layers=2,
                sequence_length=30,
                batch_size=32,
                learning_rate=0.001,
                early_stopping_patience=10,
            )
            
            trainer = TransformerTrainer(config)
            feature_set = trainer.prepare_data(
                prices=self.combined_data,
                horizon=horizon
            )
            
            result = trainer.train(feature_set)
            
            model_path = self.models_dir / f"{name.lower().replace('-', '_')}_{datetime.now().strftime('%Y%m%d')}.pkl"
            trainer.save_model(str(model_path))
            
            logger.info(f"  {name}: Loss={result.best_loss:.6f}, Saved to {model_path}")
    
    def save_training_report(self):
        """Save training report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'data_summary': {
                'symbols': list(self.training_data.keys()),
                'total_samples': len(self.combined_data) if hasattr(self, 'combined_data') else 0,
            },
            'results': self.results
        }
        
        report_path = self.models_dir / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Training report saved: {report_path}")
        return report
    
    async def run_full_training(self):
        """Run complete training pipeline."""
        logger.info("=" * 60)
        logger.info("  GNOSIS FULL MODEL TRAINING")
        logger.info(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
        
        total_start = datetime.now()
        
        # Fetch data
        await self.fetch_training_data()
        
        # Train models
        self.train_rl_agent()
        self.train_transformer()
        
        # Optional: Train multi-horizon models
        # self.train_for_multiple_horizons()
        
        # Save report
        report = self.save_training_report()
        
        total_time = (datetime.now() - total_start).total_seconds()
        
        # Final summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("  TRAINING COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"  Total Training Time: {total_time/60:.1f} minutes")
        logger.info("")
        logger.info("  Models Saved:")
        for model_type, result in self.results.items():
            if 'model_path' in result:
                logger.info(f"    - {model_type}: {result['model_path']}")
        logger.info("")
        logger.info("  Ready to trade! Start with:")
        logger.info("    ./start.sh local")
        logger.info("=" * 60)
        
        return report


async def main():
    """Main entry point."""
    trainer = FullTrainer(TRAINING_CONFIG)
    await trainer.run_full_training()


if __name__ == '__main__':
    asyncio.run(main())
