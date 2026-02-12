import React, { useState, useEffect } from 'react';
import Head from 'next/head';
import { fetchSignals } from '../lib/api';
import SignalCard from '../components/SignalCard';
import { Filter, RefreshCw, Zap } from 'lucide-react';

export default function Dashboard() {
  const [signals, setSignals] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filterMode, setFilterMode] = useState('live'); // live, paper, backtest

  const loadData = async () => {
    setLoading(true);
    const data = await fetchSignals({ mode: filterMode, limit: 20 });
    // Sort strictly by time desc
    const sorted = data.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    setSignals(sorted);
    setLoading(false);
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 30000); // Auto-refresh every 30s
    return () => clearInterval(interval);
  }, [filterMode]);

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 font-sans">
      <Head>
        <title>Gnosis | Institutional Trade Signals</title>
      </Head>

      {/* Navbar */}
      <nav className="border-b border-gray-800 bg-gray-900/95 sticky top-0 z-50 backdrop-blur">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16 items-center">
            <div className="flex items-center gap-2">
              <Zap className="text-purple-500" />
              <span className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-blue-400">
                GNOSIS
              </span>
              <span className="ml-2 text-xs bg-gray-800 px-2 py-0.5 rounded text-gray-400 border border-gray-700">
                ALPHA v1.0
              </span>
            </div>
            <div className="flex items-center gap-4">
              <button className="text-sm text-gray-400 hover:text-white">Watchlist</button>
              <button className="text-sm text-gray-400 hover:text-white">Screener</button>
              <div className="h-8 w-px bg-gray-800"></div>
              <button className="bg-purple-600 hover:bg-purple-500 text-white px-4 py-2 rounded-md text-sm font-medium transition-colors">
                Upgrade to Pro
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
        {/* Header Section */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
          <div>
            <h1 className="text-3xl font-bold text-white">Live Market Signals</h1>
            <p className="text-gray-400 mt-1">Real-time AI analysis of equity and options flow.</p>
          </div>
          
          <div className="flex items-center gap-2 bg-gray-800 p-1 rounded-lg border border-gray-700">
            <button 
              onClick={() => setFilterMode('live')}
              className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all ${filterMode === 'live' ? 'bg-gray-700 text-white shadow' : 'text-gray-400 hover:text-gray-200'}`}
            >
              Live
            </button>
            <button 
              onClick={() => setFilterMode('paper')}
              className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all ${filterMode === 'paper' ? 'bg-gray-700 text-white shadow' : 'text-gray-400 hover:text-gray-200'}`}
            >
              Paper
            </button>
            <button 
              onClick={() => loadData()} 
              className="p-1.5 ml-2 text-gray-400 hover:text-white"
              title="Refresh"
            >
              <RefreshCw size={18} className={loading ? 'animate-spin' : ''} />
            </button>
          </div>
        </div>

        {/* Stats Row */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <div className="bg-gray-800/50 border border-gray-700 p-4 rounded-lg">
            <div className="text-gray-400 text-xs uppercase font-bold tracking-wider">Active Signals</div>
            <div className="text-2xl font-bold text-white mt-1">{signals.length}</div>
          </div>
          <div className="bg-gray-800/50 border border-gray-700 p-4 rounded-lg">
            <div className="text-gray-400 text-xs uppercase font-bold tracking-wider">Avg Confidence</div>
            <div className="text-2xl font-bold text-green-400 mt-1">78%</div>
          </div>
          <div className="bg-gray-800/50 border border-gray-700 p-4 rounded-lg">
            <div className="text-gray-400 text-xs uppercase font-bold tracking-wider">Market Regime</div>
            <div className="text-2xl font-bold text-blue-400 mt-1">Volatile</div>
          </div>
          <div className="bg-gray-800/50 border border-gray-700 p-4 rounded-lg">
            <div className="text-gray-400 text-xs uppercase font-bold tracking-wider">Options Flow</div>
            <div className="text-2xl font-bold text-purple-400 mt-1">Bullish</div>
          </div>
        </div>

        {/* Signals Grid */}
        {loading && signals.length === 0 ? (
          <div className="text-center py-20 text-gray-500">Loading market intelligence...</div>
        ) : signals.length === 0 ? (
          <div className="text-center py-20 text-gray-500 bg-gray-800/30 rounded-lg border border-gray-800">
            No active signals for this filter mode.
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {signals.map((signal) => (
              <SignalCard key={signal.id} signal={signal} />
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
