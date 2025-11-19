#!/bin/bash
# Quick launcher for Super Gnosis DHPE v3 Dashboard

echo "🎯 Starting Super Gnosis DHPE v3 Dashboard..."
echo "================================================"
echo ""
echo "📊 Dashboard will open in your browser"
echo "🔄 Auto-refresh available in sidebar"
echo "💡 Tip: Enable auto-refresh for live updates"
echo ""
echo "================================================"
echo ""

# Check if streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "⚠️  Streamlit not found. Installing..."
    pip install streamlit plotly pandas -q
fi

# Run dashboard
streamlit run dashboard.py --server.headless true --server.port 8501
