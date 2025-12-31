#!/usr/bin/env python3
"""Simple HTTP server to serve the ticker viewer."""

import http.server
import socketserver
from pathlib import Path
import os

PORT = 8080

os.chdir(Path(__file__).parent)

Handler = http.server.SimpleHTTPRequestHandler

print("=" * 60)
print("📊 Super Gnosis Ticker Viewer Server")
print("=" * 60)
print(f"\n✅ Server starting on port {PORT}...")
print(f"\n🌐 Open this URL in your browser:")
print(f"   http://YOUR_SERVER_IP:{PORT}/tickers.html")
print(f"\n   (Replace YOUR_SERVER_IP with your server's IP address)")
print(f"\n⌨️  Press Ctrl+C to stop the server\n")

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()
