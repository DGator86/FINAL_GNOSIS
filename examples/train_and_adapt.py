"""Train lookahead model from ledger and prepare for adaptation."""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger

from models.features.feature_builder import EnhancedFeatureBuilder, FeatureConfig
from models.predictors.lookahead_model import LookaheadModel

load_dotenv()

app = typer.Typer(help="Train ML models from the ledger and store artifacts")


@app.command("train-ml")
def train_ml(
    ledger: Path = typer.Option(Path("data/ledger.jsonl"), help="Path to ledger JSONL"),
    model_path: Path = typer.Option(Path("data/models/lookahead.pkl"), help="Where to save the trained model"),
) -> None:
    """Train the lookahead model using ledger data."""

    builder = EnhancedFeatureBuilder(FeatureConfig())
    features = builder.build_from_ledger(ledger)

    if features.empty:
        typer.echo("Ledger was empty; nothing to train")
        raise typer.Exit(1)

    model = LookaheadModel()
    train_score, test_score = model.train(features)
    model.save(model_path)

    typer.echo(f"Model saved to {model_path} | train={train_score:.3f} test={test_score:.3f}")


if __name__ == "__main__":
    app()
