#!/usr/bin/env python3
"""Quick Transformer training."""

import os
import sys
import asyncio
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

MODELS_DIR = Path('models/trained')
MODELS_DIR.mkdir(parents=True, exist_ok=True)


async def fetch_data():
    import aiohttp
    
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    data_url = 'https://data.alpaca.markets/v2'
    
    all_data = []
    
    async with aiohttp.ClientSession() as session:
        for symbol in ['SPY', 'QQQ', 'AAPL', 'NVDA']:
            url = f"{data_url}/stocks/{symbol}/bars"
            params = {
                'start': (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%dT00:00:00Z'),
                'end': datetime.now().strftime('%Y-%m-%dT00:00:00Z'),
                'timeframe': '1Day',
                'limit': 500
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
            await asyncio.sleep(0.1)
    
    return np.vstack(all_data) if all_data else None


def train():
    from ml.training.transformer_trainer import (
        TransformerTrainer, TransformerTrainingConfig, PredictionHorizon
    )
    
    print("Fetching data...")
    data = asyncio.run(fetch_data())
    
    if data is None:
        print("Using synthetic data")
        data = np.abs(np.random.randn(1000, 5) * 10 + 100)
    
    print(f"Data shape: {data.shape}")
    
    config = TransformerTrainingConfig(
        epochs=20,
        d_model=32,
        n_heads=2,
        n_layers=1,
        sequence_length=15,
        batch_size=16,
        learning_rate=0.002,
        early_stopping_patience=5,
    )
    
    trainer = TransformerTrainer(config)
    
    print("Preparing features...")
    feature_set = trainer.prepare_data(prices=data, horizon=PredictionHorizon.MIN_15)
    
    print("Training...")
    result = trainer.train(feature_set)
    
    # Save
    model_path = MODELS_DIR / f"transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    trainer.save_model(str(model_path))
    trainer.save_model(str(MODELS_DIR / "transformer_latest.pkl"))
    
    print(f"\nDone! Loss: {result.best_loss:.6f}")
    print(f"Model: {model_path}")


if __name__ == '__main__':
    train()
