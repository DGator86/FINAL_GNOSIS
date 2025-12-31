#!/usr/bin/env python3
"""Simple standalone ticker viewer - generates an HTML file you can open in your browser."""

import json
from pathlib import Path
from datetime import datetime

import yaml


def load_tickers():
    """Load tickers from all available sources."""
    tickers = set()

    # Try loading from current_watchlist.json
    watchlist_path = Path("data/universe/current_watchlist.json")
    if watchlist_path.exists():
        try:
            with open(watchlist_path) as f:
                data = json.load(f)
                if isinstance(data, list):
                    tickers.update(data)
        except Exception:
            pass

    # Try loading from watchlist.yaml
    config_path = Path("config/watchlist.yaml")
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
                if config and "watchlist" in config:
                    for category, items in config["watchlist"].items():
                        if isinstance(items, list):
                            for item in items:
                                if isinstance(item, dict) and "symbol" in item:
                                    tickers.add(item["symbol"])
        except Exception:
            pass

    # If still empty, check environment variables from .env
    env_path = Path(".env")
    if env_path.exists() and not tickers:
        try:
            with open(env_path) as f:
                for line in f:
                    if line.startswith("TRADING_SYMBOLS="):
                        symbols = line.split("=", 1)[1].strip().strip('"\'')
                        tickers.update(s.strip() for s in symbols.split(",") if s.strip())
                        break
        except Exception:
            pass

    return sorted(tickers)


def generate_html(tickers):
    """Generate a standalone HTML file with the ticker list."""

    ticker_items = "".join(f'<div class="ticker-item">{ticker}</div>' for ticker in tickers)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Active Tickers - Super Gnosis</title>
    <style>
        :root {{
            --bg: #0f172a;
            --panel: #111827;
            --accent: #22d3ee;
            --text: #e5e7eb;
            --muted: #94a3b8;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            background: linear-gradient(135deg, #0b1222 0%, #0f172a 50%, #0b1222 100%);
            color: var(--text);
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 40px 20px;
        }}

        .container {{
            max-width: 900px;
            width: 100%;
        }}

        header {{
            text-align: center;
            margin-bottom: 40px;
        }}

        h1 {{
            font-size: 36px;
            margin-bottom: 12px;
            background: linear-gradient(90deg, #22d3ee, #0ea5e9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .subtitle {{
            color: var(--muted);
            font-size: 16px;
        }}

        .card {{
            background: var(--panel);
            border: 1px solid #1f2937;
            border-radius: 20px;
            padding: 32px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.4);
        }}

        .stats {{
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 28px;
            padding-bottom: 20px;
            border-bottom: 1px solid #1f2937;
        }}

        .stat-value {{
            font-size: 42px;
            font-weight: 700;
            color: var(--accent);
            margin-bottom: 6px;
        }}

        .stat-label {{
            color: var(--muted);
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}

        .ticker-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
            gap: 12px;
            margin-bottom: 24px;
        }}

        .ticker-item {{
            background: #0b1222;
            border: 1px solid #1f2937;
            border-radius: 12px;
            padding: 16px 12px;
            text-align: center;
            font-weight: 700;
            font-size: 18px;
            color: var(--accent);
            transition: all 0.2s ease;
        }}

        .ticker-item:hover {{
            background: #1a1f2e;
            border-color: var(--accent);
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(34, 211, 238, 0.2);
        }}

        .footer {{
            text-align: center;
            color: var(--muted);
            font-size: 13px;
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #1f2937;
        }}

        .empty-state {{
            text-align: center;
            padding: 60px 20px;
            color: var(--muted);
        }}

        .refresh-btn {{
            display: inline-block;
            margin-top: 12px;
            padding: 10px 20px;
            background: linear-gradient(90deg, #22d3ee, #0ea5e9);
            color: #0b1222;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 600;
            transition: transform 0.2s;
        }}

        .refresh-btn:hover {{
            transform: translateY(-2px);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Active Tickers</h1>
            <p class="subtitle">Symbols monitored by Super Gnosis</p>
        </header>

        <div class="card">
            <div class="stats">
                <div class="stat">
                    <div class="stat-value">{len(tickers)}</div>
                    <div class="stat-label">Active Symbols</div>
                </div>
            </div>

            {"<div class='ticker-grid'>" + ticker_items + "</div>" if tickers else "<div class='empty-state'><h3>No Active Tickers</h3><p>No tickers found in configuration files.</p></div>"}

            <div class="footer">
                <div>Generated: {datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")}</div>
                <a href="javascript:location.reload()" class="refresh-btn">🔄 Refresh Page</a>
            </div>
        </div>
    </div>
</body>
</html>"""

    return html


def main():
    """Main entry point."""
    print("📊 Super Gnosis Ticker Viewer")
    print("=" * 50)

    # Load tickers
    print("Loading tickers...")
    tickers = load_tickers()

    if tickers:
        print(f"Found {len(tickers)} tickers:")
        for ticker in tickers:
            print(f"  • {ticker}")
    else:
        print("No tickers found in configuration files.")

    # Generate HTML
    print("\nGenerating HTML file...")
    html = generate_html(tickers)

    output_file = Path("tickers.html")
    with open(output_file, "w") as f:
        f.write(html)

    print(f"\n✅ HTML file generated: {output_file.absolute()}")
    print("\n📖 To view tickers:")
    print(f"   Open this file in your browser: file://{output_file.absolute()}")
    print("\nOr just double-click: tickers.html")


if __name__ == "__main__":
    main()
