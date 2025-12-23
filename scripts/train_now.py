#!/usr/bin/env python3
"""
GNOSIS - Fast Model Training Script
Trains models quickly with real Alpaca data.
"""

import os
import sys
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

MODELS_DIR = Path('models/trained')
MODELS_DIR.mkdir(parents=True, exist_ok=True)


async def fetch_alpaca_data():
    """Fetch historical data from Alpaca."""
    import aiohttp
    
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    data_url = 'https://data.alpaca.markets/v2'
    
    symbols = ['SPY', 'QQQ', 'AAPL', 'NVDA', 'MSFT', 'AMD', 'TSLA', 'AMZN']
    all_data = []
    
    print("Fetching historical data from Alpaca...")
    
    async with aiohttp.ClientSession() as session:
        for symbol in symbols:
            url = f"{data_url}/stocks/{symbol}/bars"
            params = {
                'start': (datetime.now() - timedelta(days=500)).strftime('%Y-%m-%dT00:00:00Z'),
                'end': datetime.now().strftime('%Y-%m-%dT00:00:00Z'),
                'timeframe': '1Day',
                'limit': 1000
            }
            headers = {
                'APCA-API-KEY-ID': api_key,
                'APCA-API-SECRET-KEY': secret_key
            }
            
            async with session.get(url, params=params, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    bars = data.get('bars', [])
                    if bars:
                        arr = np.array([[b['o'], b['h'], b['l'], b['c'], b['v']] for b in bars])
                        all_data.append(arr)
                        print(f"  {symbol}: {len(arr)} bars")
            await asyncio.sleep(0.2)
    
    if all_data:
        combined = np.vstack(all_data)
        print(f"Total: {len(combined)} bars from {len(all_data)} symbols")
        return combined
    return None


def train_rl_fast(data: np.ndarray):
    """Train RL agent quickly."""
    from ml.training.rl_trainer import RLTrainer, RLTrainingConfig
    
    print("\n" + "="*60)
    print("TRAINING RL AGENT")
    print("="*60)
    
    config = RLTrainingConfig(
        num_episodes=50,  # Fast training
        max_steps_per_episode=100,
        eval_frequency=10,
        log_frequency=10,
        initial_capital=100000,
        transaction_cost=0.001,
        early_stopping_patience=20,
        checkpoint_dir=str(MODELS_DIR / 'rl_checkpoints'),
    )
    
    trainer = RLTrainer(config)
    
    # Create episodes from data
    episode_length = 150
    price_data = []
    for i in range(0, len(data) - episode_length, episode_length // 3):
        episode = data[i:i+episode_length, 3].tolist()
        price_data.append(episode)
    
    # Add random samples
    for _ in range(30):
        start = np.random.randint(0, len(data) - episode_length)
        episode = data[start:start+episode_length, 3].tolist()
        price_data.append(episode)
    
    print(f"Training on {len(price_data)} episodes...")
    
    result = trainer.train(price_data=price_data)
    
    # Save model
    model_path = MODELS_DIR / f"rl_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    trainer.save_model(str(model_path))
    
    latest_path = MODELS_DIR / "rl_agent_latest.pkl"
    trainer.save_model(str(latest_path))
    
    print(f"\nRL Training Complete!")
    print(f"  Episodes: {result.get('episodes_completed', 0)}")
    print(f"  Best Reward: {result.get('best_reward', 0):.2f}")
    print(f"  Model: {model_path}")
    
    return result


def train_transformer_fast(data: np.ndarray):
    """Train Transformer quickly."""
    from ml.training.transformer_trainer import (
        TransformerTrainer, TransformerTrainingConfig, PredictionHorizon
    )
    
    print("\n" + "="*60)
    print("TRAINING TRANSFORMER PREDICTOR")
    print("="*60)
    
    config = TransformerTrainingConfig(
        epochs=30,  # Fast training
        d_model=64,
        n_heads=4,
        n_layers=2,
        sequence_length=20,
        batch_size=32,
        learning_rate=0.001,
        early_stopping_patience=8,
    )
    
    trainer = TransformerTrainer(config)
    
    print("Preparing features...")
    feature_set = trainer.prepare_data(prices=data, horizon=PredictionHorizon.HOUR_1)
    
    print(f"Training for {config.epochs} epochs...")
    result = trainer.train(feature_set)
    
    # Save model
    model_path = MODELS_DIR / f"transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    trainer.save_model(str(model_path))
    
    latest_path = MODELS_DIR / "transformer_latest.pkl"
    trainer.save_model(str(latest_path))
    
    print(f"\nTransformer Training Complete!")
    print(f"  Epochs: {len(result.training_history)}")
    print(f"  Best Loss: {result.best_loss:.6f}")
    print(f"  Direction Accuracy: {result.final_metrics.get('direction_accuracy', 0)*100:.1f}%")
    print(f"  Model: {model_path}")
    
    return result


async def main():
    print("="*60)
    print("  GNOSIS FAST MODEL TRAINING")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    start_time = datetime.now()
    
    # Fetch data
    data = await fetch_alpaca_data()
    
    if data is None or len(data) < 500:
        print("Generating synthetic data as fallback...")
        data = np.random.randn(2000, 5) * 10 + 100
        data = np.abs(data)
    
    # Train models
    rl_result = train_rl_fast(data)
    tf_result = train_transformer_fast(data)
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'training_time_seconds': (datetime.now() - start_time).total_seconds(),
        'data_samples': len(data),
        'rl_result': {
            'episodes': rl_result.get('episodes_completed', 0),
            'best_reward': rl_result.get('best_reward', 0),
        },
        'transformer_result': {
            'epochs': len(tf_result.training_history),
            'best_loss': tf_result.best_loss,
        }
    }
    
    with open(MODELS_DIR / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    total_time = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "="*60)
    print("  TRAINING COMPLETE!")
    print("="*60)
    print(f"  Total Time: {total_time/60:.1f} minutes")
    print(f"  Models saved to: {MODELS_DIR}")
    print("\n  Start trading with: ./start.sh local")
    print("="*60)


if __name__ == '__main__':
    asyncio.run(main())
