import { useEffect, useState } from 'react';
import { api } from '../api/client';
import { List, RefreshCw, Search, TrendingUp, TrendingDown, AlertCircle } from 'lucide-react';

export default function Watchlist() {
  const [watchlist, setWatchlist] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');

  const fetchWatchlist = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await api.getWatchlist();
      setWatchlist(data);
    } catch (err) {
      setError('Failed to fetch watchlist');
      console.error('Error fetching watchlist:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchWatchlist();
    const interval = setInterval(fetchWatchlist, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, []);

  const symbols = watchlist?.symbols || [];
  const filteredSymbols = symbols.filter(symbol =>
    symbol.toLowerCase().includes(searchTerm.toLowerCase())
  );

  if (loading && !watchlist) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <RefreshCw className="w-12 h-12 text-purple-500 animate-spin mx-auto mb-4" />
          <p className="text-gray-400">Loading watchlist...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-900 bg-opacity-20 border border-red-500 rounded-lg p-6">
        <div className="flex items-center space-x-3">
          <AlertCircle className="w-6 h-6 text-red-500" />
          <div>
            <h3 className="text-lg font-semibold text-red-500">Error Loading Watchlist</h3>
            <p className="text-gray-400 mt-1">{error}</p>
            <button
              onClick={fetchWatchlist}
              className="mt-4 px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg text-white transition-colors"
            >
              Retry
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Watchlist</h1>
          <p className="text-gray-400 mt-1">Active symbols in the trading universe</p>
        </div>
        <button
          onClick={fetchWatchlist}
          disabled={loading}
          className="flex items-center space-x-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 rounded-lg text-white transition-colors"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-gray-800 bg-opacity-50 rounded-lg border border-gray-700 p-6">
          <div className="flex items-center justify-between mb-2">
            <p className="text-gray-400 text-sm">Total Symbols</p>
            <List className="w-5 h-5 text-blue-500" />
          </div>
          <p className="text-2xl font-bold text-white">{symbols.length}</p>
        </div>
        <div className="bg-gray-800 bg-opacity-50 rounded-lg border border-gray-700 p-6">
          <div className="flex items-center justify-between mb-2">
            <p className="text-gray-400 text-sm">Source</p>
            <TrendingUp className="w-5 h-5 text-green-500" />
          </div>
          <p className="text-2xl font-bold text-white">{watchlist?.source || 'Universe'}</p>
        </div>
        <div className="bg-gray-800 bg-opacity-50 rounded-lg border border-gray-700 p-6">
          <div className="flex items-center justify-between mb-2">
            <p className="text-gray-400 text-sm">Status</p>
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
          </div>
          <p className="text-2xl font-bold text-white">Active</p>
        </div>
      </div>

      {/* Search */}
      <div className="bg-gray-800 bg-opacity-50 rounded-lg border border-gray-700 p-4">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
          <input
            type="text"
            placeholder="Search symbols..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-purple-500"
          />
        </div>
      </div>

      {/* Symbol Grid */}
      <div className="bg-gray-800 bg-opacity-50 rounded-lg border border-gray-700 p-6">
        <h2 className="text-xl font-semibold text-white mb-4">Active Symbols</h2>
        {filteredSymbols.length > 0 ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
            {filteredSymbols.map((symbol, idx) => (
              <div
                key={idx}
                className="bg-gray-900 bg-opacity-50 border border-gray-700 rounded-lg p-4 text-center hover:border-purple-500 transition-colors cursor-pointer card-glow"
              >
                <p className="text-white font-bold text-lg">{symbol}</p>
                <div className="flex items-center justify-center mt-2 space-x-1">
                  <div className="w-1.5 h-1.5 bg-green-500 rounded-full" />
                  <span className="text-xs text-gray-400">Active</span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center text-gray-400 py-12">
            <List className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p>{searchTerm ? 'No symbols match your search' : 'No symbols in watchlist'}</p>
          </div>
        )}
      </div>

      {/* Info Panel */}
      <div className="bg-gradient-to-r from-blue-900 to-purple-900 bg-opacity-50 rounded-lg border border-blue-700 p-6">
        <h3 className="text-lg font-semibold text-white mb-2">About the Watchlist</h3>
        <p className="text-gray-300 text-sm">
          The watchlist contains actively monitored symbols that are analyzed by the GNOSIS trading engines.
          Each symbol undergoes multi-engine analysis including Physics, Sentiment, Liquidity (PENTA), and Hedge positioning.
        </p>
      </div>
    </div>
  );
}
