"""Flask-based real-time dashboard for the GNOSIS live trading engine."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from flask import Flask, jsonify, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from alpaca.data.timeframe import TimeFrame

from execution.broker_adapters.alpaca_trader import AlpacaConfig, AlpacaTrader
from gnosis.trading.live_trading_engine import GnosisLiveTradingEngine

app = Flask(__name__)
app.config["SECRET_KEY"] = "gnosis-secret-key-change-in-production"
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

trading_engine: Optional[GnosisLiveTradingEngine] = None
trader: Optional[AlpacaTrader] = None


def initialize_system() -> None:
    """Initialize trading system components with sensible defaults."""
    global trading_engine, trader

    config = {
        "symbols": ["SPY", "QQQ", "IWM", "AAPL", "MSFT"],
        "timeframe": "1Hour",
        "lookback_bars": 100,
        "trading_interval": 3600,
        "max_position_value": 5000,
        "use_bracket_orders": True,
        "agent_config": {
            "enabled_agents": [
                "ml_forecasting",
                "regime_detection",
                "risk_management",
                "sentiment_analysis",
            ],
            "agent_weights": {
                "ml_forecasting": 0.35,
                "regime_detection": 0.25,
                "risk_management": 0.25,
                "sentiment_analysis": 0.15,
            },
            "aggregation_strategy": "weighted_average",
            "min_agent_agreement": 0.5,
            "risk_management_config": {
                "initial_capital": 100000,
                "max_portfolio_risk": 0.02,
                "max_position_size": 0.10,
                "max_drawdown": 0.15,
                "max_daily_loss": 0.05,
                "max_daily_trades": 10,
            },
        },
    }

    try:
        trading_engine = GnosisLiveTradingEngine(config)
    except Exception as exc:  # pragma: no cover - defensive bootstrapping
        print(f"Failed to initialize trading engine: {exc}")
        trading_engine = None

    try:
        trader = AlpacaTrader(AlpacaConfig())
    except Exception as exc:  # pragma: no cover - defensive bootstrapping
        print(f"Failed to initialize Alpaca trader: {exc}")
        trader = None


def _serialize_signal(signal_data: Dict[str, Any]) -> Dict[str, Any]:
    signal_obj = signal_data.get("signal") if signal_data else None
    timestamp = signal_data.get("timestamp") if signal_data else None

    payload: Dict[str, Any] = {
        "timestamp": timestamp.isoformat() if timestamp else None,
        "price": signal_data.get("price") if signal_data else None,
    }

    if signal_obj:
        payload.update(
            {
                "signal_type": getattr(signal_obj, "signal_type", None),
                "confidence": getattr(signal_obj, "confidence", None),
                "target_price": getattr(signal_obj, "target_price", None),
                "stop_loss": getattr(signal_obj, "stop_loss", None),
                "take_profit": getattr(signal_obj, "take_profit", None),
            }
        )

    return payload


def _serialize_positions(raw_positions: Any) -> list[Dict[str, Any]]:
    positions: list[Dict[str, Any]] = []
    for pos in raw_positions or []:
        if hasattr(pos, "model_dump"):
            positions.append(pos.model_dump())
        else:
            positions.append(pos)
    return positions


def _timeframe_from_string(value: str) -> TimeFrame:
    mapping = {
        "1Min": TimeFrame.Minute,
        "5Min": TimeFrame.FiveMinutes,
        "15Min": TimeFrame.FifteenMinutes,
        "30Min": TimeFrame.ThirtyMinutes,
        "1Hour": TimeFrame.Hour,
        "1Day": TimeFrame.Day,
    }
    return mapping.get(value, TimeFrame.Hour)


# Initialize system on import so background tasks have dependencies
initialize_system()


def background_updates() -> None:
    """Send real-time updates to connected clients."""
    while True:
        try:
            if trading_engine and trading_engine.running and trader:
                account = trader.get_account()
                positions = trader.get_positions()
                summary = trading_engine.get_performance_summary()
                summary = {
                    **summary,
                    "positions": _serialize_positions(summary.get("positions")),
                    "last_signal": {
                        symbol: _serialize_signal(signal_data)
                        for symbol, signal_data in summary.get("last_signal", {}).items()
                    },
                }

                socketio.emit("account_update", account)
                socketio.emit("positions_update", positions)
                socketio.emit("performance_update", summary)

            socketio.sleep(5)
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"Error in background updates: {exc}")
            socketio.sleep(10)


socketio.start_background_task(background_updates)


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/status")
def get_status():
    try:
        is_running = bool(trading_engine and trading_engine.running)
        market_open = trader.is_market_open() if trader else False

        return jsonify(
            {
                "success": True,
                "system_running": is_running,
                "market_open": market_open,
                "timestamp": datetime.now().isoformat(),
            }
        )
    except Exception as exc:  # pragma: no cover - runtime guard
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/account")
def get_account():
    try:
        if not trader:
            return jsonify({"success": False, "error": "Trader not initialized"}), 500

        account = trader.get_account()
        return jsonify({"success": True, "data": account})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/positions")
def get_positions():
    try:
        if not trader:
            return jsonify({"success": False, "error": "Trader not initialized"}), 500

        positions = trader.get_positions()
        return jsonify({"success": True, "data": positions})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/agents")
def get_agents():
    try:
        if not trading_engine:
            return jsonify({"success": False, "error": "Trading engine not initialized"}), 500

        summaries: Dict[str, Dict[str, Any]] = {}
        signals = trading_engine.last_signal
        if signals:
            confidences = [s.get("signal").confidence for s in signals.values() if s.get("signal")]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            summaries[trading_engine.agent.agent_id] = {
                "total_signals": len(signals),
                "avg_confidence": avg_confidence,
            }
        else:
            summaries[trading_engine.agent.agent_id] = {"total_signals": 0, "avg_confidence": 0}

        return jsonify({"success": True, "data": summaries})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/signals")
def get_signals():
    try:
        if not trading_engine:
            return jsonify({"success": False, "error": "Trading engine not initialized"}), 500

        signals: Dict[str, Dict[str, Any]] = {}
        for symbol, signal_data in trading_engine.last_signal.items():
            signals[symbol] = _serialize_signal(signal_data)

        return jsonify({"success": True, "data": signals})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/performance")
def get_performance():
    try:
        if not trading_engine:
            return jsonify({"success": False, "error": "Trading engine not initialized"}), 500

        summary = trading_engine.get_performance_summary()
        if "positions" in summary:
            summary["positions"] = _serialize_positions(summary.get("positions"))
        if "last_signal" in summary:
            summary["last_signal"] = {
                symbol: _serialize_signal(signal_data)
                for symbol, signal_data in summary.get("last_signal", {}).items()
            }

        return jsonify({"success": True, "data": summary})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/market_data/<symbol>")
def get_market_data(symbol: str):
    try:
        if not trader:
            return jsonify({"success": False, "error": "Trader not initialized"}), 500

        timeframe_value = (
            _timeframe_from_string(trading_engine.timeframe)
            if trading_engine
            else _timeframe_from_string("1Hour")
        )
        data = trader.get_market_data(symbol, timeframe=timeframe_value, limit=100)
        if data.empty:
            return jsonify({"success": False, "error": "No data available"}), 404

        data_dict = data.to_dict("records")
        for record in data_dict:
            if "timestamp" in record:
                record["timestamp"] = record["timestamp"].isoformat()

        return jsonify({"success": True, "data": data_dict})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/watchlist")
def get_watchlist():
    try:
        if not trading_engine:
            return jsonify({"success": False, "error": "Trading engine not initialized"}), 500
        if not trader:
            return jsonify({"success": False, "error": "Trader not initialized"}), 500

        watchlist = []
        for symbol in trading_engine.symbols:
            data = trader.get_market_data(symbol, limit=1)
            if not data.empty:
                current_price = data["close"].iloc[-1]
                signal_data = trading_engine.last_signal.get(symbol, {})
                parsed_signal = _serialize_signal(signal_data)

                watchlist.append(
                    {
                        "symbol": symbol,
                        "price": float(current_price),
                        "last_signal": parsed_signal.get("signal_type", "none"),
                        "confidence": parsed_signal.get("confidence", 0) or 0,
                    }
                )

        return jsonify({"success": True, "data": watchlist})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/start", methods=["POST"])
def start_trading():
    try:
        if not trading_engine:
            return jsonify({"success": False, "error": "Trading engine not initialized"}), 500
        if trading_engine.running:
            return jsonify({"success": False, "error": "Already running"}), 400

        trading_engine.start()
        return jsonify({"success": True, "message": "Trading started"})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/stop", methods=["POST"])
def stop_trading():
    try:
        if not trading_engine:
            return jsonify({"success": False, "error": "Trading engine not initialized"}), 500
        if not trading_engine.running:
            return jsonify({"success": False, "error": "Not running"}), 400

        trading_engine.stop()
        return jsonify({"success": True, "message": "Trading stopped"})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/emergency_stop", methods=["POST"])
def emergency_stop():
    try:
        if not trading_engine:
            return jsonify({"success": False, "error": "Trading engine not initialized"}), 500

        trading_engine.emergency_stop()
        return jsonify({"success": True, "message": "Emergency stop executed"})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/close_position/<symbol>", methods=["POST"])
def close_position(symbol: str):
    try:
        if not trader:
            return jsonify({"success": False, "error": "Trader not initialized"}), 500

        success = trader.close_position(symbol)
        if success:
            return jsonify({"success": True, "message": f"Position closed: {symbol}"})
        return jsonify({"success": False, "error": "Failed to close position"}), 500
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@socketio.on("connect")
def handle_connect():
    print("Client connected")
    emit("connected", {"message": "Connected to GNOSIS"})


@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")


@socketio.on("request_update")
def handle_update_request():
    try:
        if trader:
            account = trader.get_account()
            positions = trader.get_positions()

            emit("account_update", account)
            emit("positions_update", positions)
    except Exception as exc:
        emit("error", {"message": str(exc)})


if __name__ == "__main__":  # pragma: no cover - script execution path
    print(
        """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║          🌐 GNOSIS WEB DASHBOARD STARTING 🌐             ║
    ║                                                           ║
    ║     Access at: http://localhost:5000                      ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    )
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)
