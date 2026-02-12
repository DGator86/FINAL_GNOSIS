#!/usr/bin/env python3
"""
Minimal training script - trains models quickly.
"""

import os
import sys
import numpy as np
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

def main():
    print("=" * 60)
    print("  GNOSIS MINIMAL MODEL TRAINING")
    print("=" * 60)
    
    models_dir = Path('models/trained')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic data
    print("\n[1/4] Generating training data...")
    np.random.seed(42)
    n_samples = 1000
    price = 100.0
    data = []
    for _ in range(n_samples):
        change = np.random.normal(0.0002, 0.02)
        o, h, l, c = price, price*1.01, price*0.99, price*(1+change)
        v = np.random.uniform(1e6, 1e8)
        data.append([o, h, l, c, v])
        price = c
    combined_data = np.array(data)
    print(f"  Generated {len(combined_data)} samples")
    
    # Train RL Agent
    print("\n[2/4] Training RL Agent...")
    from ml.training.rl_trainer import RLTrainer, RLTrainingConfig
    
    rl_config = RLTrainingConfig(
        num_episodes=30,
        max_steps_per_episode=100,
        eval_frequency=10,
        log_frequency=10,
        early_stopping_patience=20,
        checkpoint_dir=str(models_dir / 'rl_checkpoints')
    )
    
    # Create episodes
    episode_len = 150
    price_data = []
    for i in range(0, len(combined_data) - episode_len, 50):
        episode = combined_data[i:i+episode_len, 3].tolist()
        price_data.append(episode)
    
    print(f"  Episodes: {len(price_data)}")
    
    trainer = RLTrainer(rl_config)
    rl_result = trainer.train(price_data=price_data)
    
    rl_path = models_dir / f"rl_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    trainer.save_model(str(rl_path))
    trainer.save_model(str(models_dir / "rl_agent_latest.pkl"))
    
    print(f"  RL Agent saved: {rl_path}")
    print(f"  Best Reward: {rl_result.get('best_reward', 'N/A')}")
    
    # Train Transformer
    print("\n[3/4] Training Transformer...")
    from ml.training.transformer_trainer import (
        TransformerTrainer, TransformerTrainingConfig, PredictionHorizon
    )
    
    tf_config = TransformerTrainingConfig(
        epochs=20,
        d_model=32,
        n_heads=2,
        n_layers=1,
        sequence_length=20,
        batch_size=16,
        learning_rate=0.001,
        early_stopping_patience=5
    )
    
    tf_trainer = TransformerTrainer(tf_config)
    feature_set = tf_trainer.prepare_data(combined_data, horizon=PredictionHorizon.MIN_15)
    tf_result = tf_trainer.train(feature_set)
    
    tf_path = models_dir / f"transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    tf_trainer.save_model(str(tf_path))
    tf_trainer.save_model(str(models_dir / "transformer_latest.pkl"))
    
    print(f"  Transformer saved: {tf_path}")
    print(f"  Best Loss: {tf_result.best_loss:.6f}")
    
    # Summary
    print("\n[4/4] Training Complete!")
    print("=" * 60)
    print("  Models saved to: models/trained/")
    print("  - rl_agent_latest.pkl")
    print("  - transformer_latest.pkl")
    print("=" * 60)
    print("\n  Ready to trade! Run: ./start.sh local")

if __name__ == '__main__':
    main()
