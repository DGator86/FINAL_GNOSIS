"""Experiment tracking utilities for GNOSIS ML workflows."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sqlite3
import subprocess
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class Experiment:
    """Experiment metadata and results."""

    experiment_id: str
    experiment_name: str
    model_type: str
    parameters: Dict[str, Any]
    metrics: Dict[str, Any]
    artifacts: Dict[str, Dict[str, Any]]
    tags: List[str]
    status: str  # "running", "completed", "failed"
    created_at: datetime
    updated_at: datetime
    duration_seconds: Optional[float] = None
    notes: str = ""
    parent_experiment_id: Optional[str] = None
    git_commit: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the experiment to a JSON-serialisable dictionary."""

        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experiment":
        """Reconstruct an :class:`Experiment` from a dictionary."""

        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


class MetricLogger:
    """Real-time metric logging during training."""

    def __init__(self, experiment_id: str, tracker: "ExperimentTracker") -> None:
        self.experiment_id = experiment_id
        self.tracker = tracker
        self.metrics_history: List[Dict[str, Any]] = []
        self.step = 0

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a single metric value."""

        if step is None:
            step = self.step
            self.step += 1

        metric_entry = {
            "name": name,
            "value": float(value),
            "step": step,
            "timestamp": datetime.now().isoformat(),
        }

        self.metrics_history.append(metric_entry)
        self.tracker._log_metric_to_db(self.experiment_id, metric_entry)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics at once."""

        if step is None:
            step = self.step
            self.step += 1

        for name, value in metrics.items():
            self.log_metric(name, value, step)

    def log_hyperparameter(self, name: str, value: Any) -> None:
        """Log a hyperparameter value for the experiment."""

        self.tracker.log_hyperparameter(self.experiment_id, name, value)

    def log_artifact(self, name: str, path: str, artifact_type: str = "file") -> None:
        """Log a model artifact such as a model checkpoint or plot."""

        self.tracker.log_artifact(self.experiment_id, name, path, artifact_type)


class ExperimentTracker:
    """Main experiment tracking system backed by SQLite."""

    def __init__(self, base_path: str | Path = "./experiments") -> None:
        self.base_path = Path(base_path).expanduser()
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.db_path = self.base_path / "experiments.db"
        self._init_database()

        self.logger = self._setup_logger()
        self._lock = threading.Lock()
        self._active_experiments: Dict[str, Experiment] = {}

    def _setup_logger(self) -> logging.Logger:
        """Configure the experiment tracker logger."""

        logger = logging.getLogger("gnosis.experiment_tracker")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _init_database(self) -> None:
        """Initialize the SQLite database schema."""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    experiment_name TEXT,
                    model_type TEXT,
                    parameters TEXT,
                    metrics TEXT,
                    artifacts TEXT,
                    tags TEXT,
                    status TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    duration_seconds REAL,
                    notes TEXT,
                    parent_experiment_id TEXT,
                    git_commit TEXT
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metrics_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    step INTEGER,
                    timestamp TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS hyperparameters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    param_name TEXT,
                    param_value TEXT,
                    param_type TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
                """
            )

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_exp_name ON experiments(experiment_name)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_exp_model_type ON experiments(model_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_metrics_exp_id ON metrics_history(experiment_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_hyperparams_exp_id ON hyperparameters(experiment_id)"
            )

    def create_experiment(
        self,
        experiment_name: str,
        model_type: str,
        parameters: Dict[str, Any],
        tags: Optional[List[str]] = None,
        parent_experiment_id: Optional[str] = None,
    ) -> str:
        """Create a new experiment and persist its metadata."""

        experiment_id = self._generate_experiment_id(experiment_name, parameters)
        exp_dir = self.base_path / experiment_id
        exp_dir.mkdir(exist_ok=True)

        experiment = Experiment(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            model_type=model_type,
            parameters=parameters,
            metrics={},
            artifacts={},
            tags=tags or [],
            status="running",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            parent_experiment_id=parent_experiment_id,
            git_commit=self._get_git_commit(),
        )

        with self._lock:
            self._save_experiment_to_db(experiment)
            self._active_experiments[experiment_id] = experiment

        self.logger.info("Created experiment %s: %s", experiment_id, experiment_name)
        return experiment_id

    def _generate_experiment_id(self, name: str, parameters: Dict[str, Any]) -> str:
        """Generate a unique experiment identifier based on inputs and timestamp."""

        content = f"{name}_{json.dumps(parameters, sort_keys=True)}_{datetime.now().isoformat()}"
        hash_obj = hashlib.md5(content.encode(), usedforsecurity=False)
        return f"{name}_{hash_obj.hexdigest()[:8]}"

    def _get_git_commit(self) -> Optional[str]:
        """Return the current git commit hash if available."""

        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.base_path,
                check=False,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            return None
        return None

    def get_metric_logger(self, experiment_id: str) -> MetricLogger:
        """Return a convenience logger for streaming metrics."""

        return MetricLogger(experiment_id, self)

    def log_metric(
        self, experiment_id: str, name: str, value: float, step: Optional[int] = None
    ) -> None:
        """Log a metric for an experiment without creating a :class:`MetricLogger`."""

        metric_entry = {
            "name": name,
            "value": float(value),
            "step": step or 0,
            "timestamp": datetime.now().isoformat(),
        }

        self._log_metric_to_db(experiment_id, metric_entry)
        with self._lock:
            if experiment_id in self._active_experiments:
                exp = self._active_experiments[experiment_id]
                exp.metrics.setdefault(name, []).append(value)

    def log_hyperparameter(self, experiment_id: str, name: str, value: Any) -> None:
        """Persist a hyperparameter value to the database."""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO hyperparameters (experiment_id, param_name, param_value, param_type)
                VALUES (?, ?, ?, ?)
                """,
                (experiment_id, name, str(value), type(value).__name__),
            )

    def log_artifact(
        self, experiment_id: str, name: str, path: str, artifact_type: str = "file"
    ) -> None:
        """Record the path to an artifact produced by the experiment."""

        with self._lock:
            if experiment_id in self._active_experiments:
                exp = self._active_experiments[experiment_id]
                exp.artifacts[name] = {
                    "path": path,
                    "type": artifact_type,
                    "timestamp": datetime.now().isoformat(),
                }
                self._save_experiment_to_db(exp)

    def update_experiment_status(
        self, experiment_id: str, status: str, metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update the experiment status and capture final metrics if provided."""

        with self._lock:
            if experiment_id in self._active_experiments:
                exp = self._active_experiments[experiment_id]
                exp.status = status
                exp.updated_at = datetime.now()

                if status in {"completed", "failed"} and exp.created_at:
                    exp.duration_seconds = (
                        exp.updated_at - exp.created_at
                    ).total_seconds()

                if metrics:
                    exp.metrics.update(metrics)

                self._save_experiment_to_db(exp)

                if status in {"completed", "failed"}:
                    del self._active_experiments[experiment_id]

        self.logger.info("Updated experiment %s status to %s", experiment_id, status)

    def _save_experiment_to_db(self, experiment: Experiment) -> None:
        """Persist the experiment object to the database."""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiments (
                    experiment_id,
                    experiment_name,
                    model_type,
                    parameters,
                    metrics,
                    artifacts,
                    tags,
                    status,
                    created_at,
                    updated_at,
                    duration_seconds,
                    notes,
                    parent_experiment_id,
                    git_commit
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment.experiment_id,
                    experiment.experiment_name,
                    experiment.model_type,
                    json.dumps(experiment.parameters),
                    json.dumps(experiment.metrics),
                    json.dumps(experiment.artifacts),
                    json.dumps(experiment.tags),
                    experiment.status,
                    experiment.created_at.isoformat(),
                    experiment.updated_at.isoformat(),
                    experiment.duration_seconds,
                    experiment.notes,
                    experiment.parent_experiment_id,
                    experiment.git_commit,
                ),
            )

    def _log_metric_to_db(self, experiment_id: str, metric_entry: Dict[str, Any]) -> None:
        """Insert a metric entry into the database."""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO metrics_history (
                    experiment_id, metric_name, metric_value, step, timestamp
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    metric_entry["name"],
                    metric_entry["value"],
                    metric_entry["step"],
                    metric_entry["timestamp"],
                ),
            )

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Retrieve a single experiment by identifier."""

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
            )
            row = cursor.fetchone()

            if row:
                return self._row_to_experiment(row)
        return None

    def search_experiments(
        self,
        experiment_name: Optional[str] = None,
        model_type: Optional[str] = None,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[Experiment]:
        """Search experiments with optional filters and tag matching."""

        query = "SELECT * FROM experiments WHERE 1=1"
        params: List[Any] = []

        if experiment_name:
            query += " AND experiment_name LIKE ?"
            params.append(f"%{experiment_name}%")

        if model_type:
            query += " AND model_type = ?"
            params.append(model_type)

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        experiments: List[Experiment] = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            for row in rows:
                exp = self._row_to_experiment(row)

                if tags:
                    if any(tag in exp.tags for tag in tags):
                        experiments.append(exp)
                else:
                    experiments.append(exp)

        return experiments

    def _row_to_experiment(self, row: Tuple[Any, ...]) -> Experiment:
        """Convert a database row to an :class:`Experiment`."""

        return Experiment(
            experiment_id=row[0],
            experiment_name=row[1],
            model_type=row[2],
            parameters=json.loads(row[3]) if row[3] else {},
            metrics=json.loads(row[4]) if row[4] else {},
            artifacts=json.loads(row[5]) if row[5] else {},
            tags=json.loads(row[6]) if row[6] else [],
            status=row[7],
            created_at=datetime.fromisoformat(row[8]),
            updated_at=datetime.fromisoformat(row[9]),
            duration_seconds=row[10],
            notes=row[11] or "",
            parent_experiment_id=row[12],
            git_commit=row[13],
        )

    def get_experiment_metrics(self, experiment_id: str) -> pd.DataFrame:
        """Return the metrics history for an experiment as a DataFrame."""

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                """
                SELECT metric_name, metric_value, step, timestamp
                FROM metrics_history
                WHERE experiment_id = ?
                ORDER BY timestamp
                """,
                conn,
                params=(experiment_id,),
            )

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        return df

    def compare_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        """Compare experiments by final and best metric values."""

        comparison_data: List[Dict[str, Any]] = []

        for exp_id in experiment_ids:
            exp = self.get_experiment(exp_id)
            if exp:
                metrics_df = self.get_experiment_metrics(exp_id)

                exp_data: Dict[str, Any] = {
                    "experiment_id": exp_id,
                    "experiment_name": exp.experiment_name,
                    "model_type": exp.model_type,
                    "status": exp.status,
                    "duration_minutes": (
                        exp.duration_seconds / 60 if exp.duration_seconds else None
                    ),
                    "created_at": exp.created_at,
                }

                for metric_name in metrics_df["metric_name"].unique():
                    metric_data = metrics_df[metrics_df["metric_name"] == metric_name]
                    if not metric_data.empty:
                        final_value = metric_data["metric_value"].iloc[-1]
                        best_value = (
                            metric_data["metric_value"].min()
                            if "loss" in metric_name.lower()
                            or "error" in metric_name.lower()
                            else metric_data["metric_value"].max()
                        )
                        exp_data[f"{metric_name}_final"] = final_value
                        exp_data[f"{metric_name}_best"] = best_value

                comparison_data.append(exp_data)

        return pd.DataFrame(comparison_data)

    def get_best_experiment(
        self,
        metric_name: str,
        model_type: Optional[str] = None,
        minimize: bool = True,
    ) -> Optional[Experiment]:
        """Return the experiment with the best value for the requested metric."""

        experiments = self.search_experiments(model_type=model_type, status="completed")

        best_exp: Optional[Experiment] = None
        best_value = float("inf") if minimize else float("-inf")

        for exp in experiments:
            metrics_df = self.get_experiment_metrics(exp.experiment_id)
            metric_data = metrics_df[metrics_df["metric_name"] == metric_name]

            if not metric_data.empty:
                value = metric_data["metric_value"].iloc[-1]

                if minimize and value < best_value:
                    best_value = value
                    best_exp = exp
                elif not minimize and value > best_value:
                    best_value = value
                    best_exp = exp

        return best_exp

    def export_experiments(
        self, output_path: str | Path, experiment_ids: Optional[List[str]] = None
    ) -> None:
        """Export experiments and their metric history to JSON."""

        if experiment_ids:
            experiments = [self.get_experiment(exp_id) for exp_id in experiment_ids]
            experiments = [exp for exp in experiments if exp]
        else:
            experiments = self.search_experiments(limit=1000)

        export_data: List[Dict[str, Any]] = []
        for exp in experiments:
            exp_data = exp.to_dict()
            exp_data["metrics_history"] = (
                self.get_experiment_metrics(exp.experiment_id).to_dict("records")
            )
            export_data.append(exp_data)

        output_path = Path(output_path)
        output_path.write_text(json.dumps(export_data, indent=2, default=str))

        self.logger.info("Exported %s experiments to %s", len(export_data), output_path)

    def cleanup_old_experiments(self, days_old: int = 90, keep_completed: bool = True) -> None:
        """Remove experiments older than ``days_old`` from disk and the database."""

        cutoff_date = datetime.now() - timedelta(days=days_old)

        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT experiment_id FROM experiments WHERE created_at < ?"
            if keep_completed:
                query += " AND status != 'completed'"

            cursor = conn.execute(query, (cutoff_date.isoformat(),))
            exp_ids_to_delete = [row[0] for row in cursor.fetchall()]

            for exp_id in exp_ids_to_delete:
                conn.execute(
                    "DELETE FROM experiments WHERE experiment_id = ?", (exp_id,)
                )
                conn.execute(
                    "DELETE FROM metrics_history WHERE experiment_id = ?", (exp_id,)
                )
                conn.execute(
                    "DELETE FROM hyperparameters WHERE experiment_id = ?", (exp_id,)
                )

                exp_dir = self.base_path / exp_id
                if exp_dir.exists():
                    shutil.rmtree(exp_dir)

        self.logger.info("Cleaned up %s old experiments", len(exp_ids_to_delete))
