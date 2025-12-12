#!/usr/bin/env python3
"""
CLI script for training LSTM lookahead models

Usage:
    python cli/lstm_train.py --symbol SPY --start 2024-01-01 --end 2024-12-01 --save models/lstm_spy.pth
    python cli/lstm_train.py --symbol AAPL --days 180 --save models/lstm_aapl.pth
    python cli/lstm_train.py --config config/lstm_lookahead_example.yaml
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engines.inputs.market_data_adapter import MarketDataAdapter
from engines.ml.lstm_engine import LSTMPredictionEngine
from models.features.feature_builder import EnhancedFeatureBuilder
from models.lstm_lookahead import LookaheadConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train LSTM Lookahead model for stock price prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on 6 months of SPY data
  python cli/lstm_train.py --symbol SPY --days 180 --save models/lstm_spy.pth

  # Train on specific date range
  python cli/lstm_train.py --symbol AAPL --start 2024-01-01 --end 2024-12-01 --save models/lstm_aapl.pth

  # Custom architecture
  python cli/lstm_train.py --symbol QQQ --days 365 --hidden-dim 256 --layers 3 --save models/lstm_qqq.pth
        """
    )

    # Required arguments
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Trading symbol (e.g., SPY, AAPL, QQQ)"
    )

    # Date range (mutually exclusive groups)
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument(
        "--days",
        type=int,
        help="Number of days of historical data to use (from today backwards)"
    )
    date_group.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD). Must also specify --end"
    )

    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD). Only used with --start"
    )

    # Model configuration
    parser.add_argument(
        "--save",
        type=str,
        required=True,
        help="Path to save trained model"
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="LSTM hidden dimension (default: 128)"
    )

    parser.add_argument(
        "--layers",
        type=int,
        default=2,
        help="Number of LSTM layers (default: 2)"
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate (default: 0.3)"
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=60,
        help="Sequence length for input (default: 60)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum training epochs (default: 50)"
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)"
    )

    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 5, 15, 60],
        help="Forecast horizons in minutes (default: 1 5 15 60)"
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU training (default: use GPU if available)"
    )

    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()

    logger.info(f"Starting LSTM Lookahead training for {args.symbol}")
    logger.info(f"Configuration:")
    logger.info(f"  Hidden dim: {args.hidden_dim}")
    logger.info(f"  Layers: {args.layers}")
    logger.info(f"  Dropout: {args.dropout}")
    logger.info(f"  Sequence length: {args.sequence_length}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Max epochs: {args.epochs}")
    logger.info(f"  Forecast horizons: {args.horizons}")

    # Determine date range
    if args.days:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        logger.info(f"  Date range: Last {args.days} days")
    else:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
        end_date = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.now()
        logger.info(f"  Date range: {start_date.date()} to {end_date.date()}")

    # Create LSTM configuration
    config = LookaheadConfig(
        input_dim=150,  # Will be set by feature builder
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        dropout=args.dropout,
        forecast_horizons=args.horizons,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        device="cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"),
    )

    logger.info(f"  Device: {config.device}")

    # Initialize components
    logger.info("Initializing market data adapter and feature builder...")
    market_adapter = MarketDataAdapter()
    feature_builder = EnhancedFeatureBuilder()

    # Create LSTM engine
    logger.info("Initializing LSTM Prediction Engine...")
    engine = LSTMPredictionEngine(
        market_adapter=market_adapter,
        feature_builder=feature_builder,
        config=config,
    )

    # Train model
    logger.info("Starting training...")
    logger.info("This may take several minutes depending on data size and hardware...")

    try:
        history = engine.train_from_historical_data(
            symbol=args.symbol,
            start_date=start_date,
            end_date=end_date,
            save_path=args.save,
        )

        # Print training summary
        logger.info("\n" + "="*60)
        logger.info("Training completed successfully!")
        logger.info("="*60)
        logger.info(f"Final training loss: {history['train_loss'][-1]:.6f}")
        logger.info(f"Final validation loss: {history['val_loss'][-1]:.6f}")
        logger.info(f"Total epochs: {len(history['train_loss'])}")
        logger.info(f"Model saved to: {args.save}")
        logger.info("="*60)

        # Show loss progression
        logger.info("\nLoss progression (every 10 epochs):")
        for i in range(0, len(history['train_loss']), 10):
            logger.info(
                f"  Epoch {i+1:3d}: "
                f"Train Loss = {history['train_loss'][i]:.6f}, "
                f"Val Loss = {history['val_loss'][i]:.6f}"
            )

        logger.info("\nTo use this model:")
        logger.info(f"  from engines.ml.lstm_engine import LSTMPredictionEngine")
        logger.info(f"  engine = LSTMPredictionEngine(")
        logger.info(f"      market_adapter=market_adapter,")
        logger.info(f"      model_path='{args.save}'")
        logger.info(f"  )")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    import torch  # Import here to avoid issues if torch not installed
    sys.exit(main())
