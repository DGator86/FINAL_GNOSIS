#!/usr/bin/env python3
"""
GNOSIS Trading System - Quick Model Training

Downloads historical data and trains ML models for trading.
Designed to run in ~30 minutes to get models ready for trading.

Usage:
    python scripts/quick_train.py [--symbols SPY,QQQ] [--days 365]
"""

import os
import sys
import asyncio
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('GNOSIS-Train')


class QuickTrainer:
    """Quick training pipeline for GNOSIS models."""
    
    def __init__(self, symbols: list = None, days: int = 365):
        self.symbols = symbols or ['SPY', 'QQQ', 'AAPL', 'NVDA', 'MSFT']
        self.days = days
        self.data_dir = Path('data/training')
        self.model_dir = Path('models/trained')
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    async def download_historical_data(self) -> dict:
        """Download historical data from Alpaca."""
        logger.info(f"📥 Downloading {self.days} days of data for {len(self.symbols)} symbols...")
        
        import aiohttp
        import pandas as pd
        
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        headers = {
            'APCA-API-KEY-ID': api_key,
            'APCA-API-SECRET-KEY': secret_key,
        }
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.days)
        
        all_data = {}
        
        async with aiohttp.ClientSession() as session:
            for symbol in self.symbols:
                logger.info(f"  Fetching {symbol}...")
                
                url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
                params = {
                    'start': start_date.strftime('%Y-%m-%dT00:00:00Z'),
                    'end': end_date.strftime('%Y-%m-%dT00:00:00Z'),
                    'timeframe': '1Hour',
                    'limit': 10000,
                }
                
                try:
                    async with session.get(url, headers=headers, params=params) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            bars = data.get('bars', [])
                            
                            if bars:
                                df = pd.DataFrame(bars)
                                df['t'] = pd.to_datetime(df['t'])
                                df = df.rename(columns={
                                    't': 'timestamp',
                                    'o': 'open',
                                    'h': 'high',
                                    'l': 'low',
                                    'c': 'close',
                                    'v': 'volume'
                                })
                                
                                # Save to file
                                filepath = self.data_dir / f"{symbol}_1h.csv"
                                df.to_csv(filepath, index=False)
                                
                                all_data[symbol] = df
                                logger.info(f"    ✅ {symbol}: {len(df)} bars")
                            else:
                                logger.warning(f"    ⚠️ {symbol}: No data returned")
                        else:
                            error = await resp.text()
                            logger.error(f"    ❌ {symbol}: {error}")
                            
                except Exception as e:
                    logger.error(f"    ❌ {symbol}: {e}")
        
        return all_data
    
    async def train_rl_agent(self, data: dict):
        """Train the RL trading agent."""
        logger.info("\n🤖 Training RL Agent...")
        
        try:
            from ml.training.rl_trainer import RLTrainer, RLTrainingConfig
            import numpy as np
            
            # Combine all data for training
            all_prices = []
            for symbol, df in data.items():
                prices = df['close'].values
                all_prices.append(prices)
            
            if not all_prices:
                logger.warning("No price data available, using synthetic data")
                # Generate synthetic data for demo
                np.random.seed(42)
                prices = 100 * np.cumprod(1 + np.random.randn(5000) * 0.02)
            else:
                prices = np.concatenate(all_prices)
            
            # Configure for quick training
            config = RLTrainingConfig(
                num_episodes=100,          # Quick training
                max_steps_per_episode=200,
                eval_frequency=20,
                log_frequency=10,
                early_stopping_patience=20,
                initial_capital=100000,
                transaction_cost=0.001,
                checkpoint_dir=str(self.model_dir / 'rl_checkpoints'),
            )
            
            trainer = RLTrainer(config)
            
            # Train with progress updates
            logger.info("  Starting training...")
            result = trainer.train(price_data=prices)
            
            logger.info(f"  ✅ RL Agent trained!")
            logger.info(f"     Episodes: {result.get('episodes_completed', 0)}")
            logger.info(f"     Best Reward: {result.get('best_reward', 0):.2f}")
            logger.info(f"     Time: {result.get('training_time_seconds', 0):.1f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"  ❌ RL training failed: {e}")
            return None
    
    async def train_transformer(self, data: dict):
        """Train the Transformer price predictor."""
        logger.info("\n🔮 Training Transformer Predictor...")
        
        try:
            from ml.training.transformer_trainer import (
                TransformerTrainer, TransformerTrainingConfig, PredictionHorizon
            )
            import numpy as np
            
            # Prepare OHLCV data
            all_ohlcv = []
            for symbol, df in data.items():
                ohlcv = df[['open', 'high', 'low', 'close', 'volume']].values
                all_ohlcv.append(ohlcv)
            
            if not all_ohlcv:
                logger.warning("No OHLCV data available, using synthetic data")
                np.random.seed(42)
                n_samples = 5000
                ohlcv = np.column_stack([
                    100 + np.cumsum(np.random.randn(n_samples) * 0.5),  # open
                    100 + np.cumsum(np.random.randn(n_samples) * 0.5) + 1,  # high
                    100 + np.cumsum(np.random.randn(n_samples) * 0.5) - 1,  # low
                    100 + np.cumsum(np.random.randn(n_samples) * 0.5),  # close
                    np.random.randint(1000000, 5000000, n_samples),  # volume
                ])
            else:
                ohlcv = np.vstack(all_ohlcv)
            
            # Configure for quick training
            config = TransformerTrainingConfig(
                d_model=32,
                n_heads=4,
                n_layers=2,
                sequence_length=30,
                epochs=20,                # Quick training
                batch_size=32,
                learning_rate=0.001,
                early_stopping_patience=5,
                checkpoint_dir=str(self.model_dir / 'transformer_checkpoints'),
            )
            
            trainer = TransformerTrainer(config)
            
            # Prepare data
            logger.info("  Preparing features...")
            feature_set = trainer.prepare_data(ohlcv, horizon=PredictionHorizon.HOUR_1)
            
            # Train
            logger.info("  Starting training...")
            result = trainer.train(feature_set)
            
            logger.info(f"  ✅ Transformer trained!")
            logger.info(f"     Epochs: {len(result.training_history)}")
            logger.info(f"     Best Loss: {result.best_val_loss:.4f}")
            logger.info(f"     Time: {result.training_time_seconds:.1f}s")
            
            # Save model
            model_path = self.model_dir / 'transformer_model.pkl'
            trainer.save_model(str(model_path))
            logger.info(f"     Model saved to: {model_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"  ❌ Transformer training failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def run(self):
        """Run the complete training pipeline."""
        logger.info("""
╔═══════════════════════════════════════════════════════════════╗
║              GNOSIS Quick Model Training                       ║
╚═══════════════════════════════════════════════════════════════╝
        """)
        
        start_time = datetime.now()
        
        # Step 1: Download historical data
        data = await self.download_historical_data()
        
        if not data:
            logger.warning("No historical data downloaded, will use synthetic data")
            data = {}
        
        # Step 2: Train RL Agent
        rl_result = await self.train_rl_agent(data)
        
        # Step 3: Train Transformer
        transformer_result = await self.train_transformer(data)
        
        # Summary
        elapsed = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"""
╔═══════════════════════════════════════════════════════════════╗
║                    Training Complete!                          ║
╠═══════════════════════════════════════════════════════════════╣
║  Total Time: {elapsed/60:.1f} minutes                                      
║  RL Agent: {'✅ Trained' if rl_result else '❌ Failed'}                                       
║  Transformer: {'✅ Trained' if transformer_result else '❌ Failed'}                                  
║                                                                
║  Models saved to: {self.model_dir}                            
╚═══════════════════════════════════════════════════════════════╝

🚀 Ready to trade! Run: python scripts/launch.py
        """)
        
        return rl_result is not None or transformer_result is not None


async def quick_train():
    """Quick training function for import."""
    trainer = QuickTrainer()
    return await trainer.run()


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='GNOSIS Quick Model Training')
    parser.add_argument('--symbols', type=str, 
                        help='Comma-separated list of symbols')
    parser.add_argument('--days', type=int, default=365,
                        help='Days of historical data to download')
    
    args = parser.parse_args()
    
    symbols = args.symbols.split(',') if args.symbols else None
    
    trainer = QuickTrainer(symbols=symbols, days=args.days)
    success = await trainer.run()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    asyncio.run(main())
