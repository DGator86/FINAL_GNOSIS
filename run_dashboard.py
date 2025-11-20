"""
Quick start script for the GNOSIS Flask dashboard.
"""

if __name__ == "__main__":
    print(
        """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║          🌐 GNOSIS WEB DASHBOARD 🌐                      ║
    ║                                                           ║
    ║  1. Open browser: http://localhost:5000                   ║
    ║  2. Click 'Start Trading' to begin                        ║
    ║  3. Monitor positions and agents in real-time             ║
    ║  4. Use 'Emergency Stop' if needed                        ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    )

    from gnosis_dashboard import app, socketio  # pylint: disable=wrong-import-position

    socketio.run(app, debug=False, host="0.0.0.0", port=5000)
