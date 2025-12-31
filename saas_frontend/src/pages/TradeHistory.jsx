import { useEffect, useState } from 'react';
import { api } from '../api/client';
import { Activity, RefreshCw, TrendingUp, TrendingDown, AlertCircle, Calendar, DollarSign } from 'lucide-react';

export default function TradeHistory() {
  const [trades, setTrades] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [limit, setLimit] = useState(50);

  const fetchTrades = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await api.getTrades(limit);
      setTrades(data);
    } catch (err) {
      setError('Failed to fetch trade history');
      console.error('Error fetching trades:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTrades();
    const interval = setInterval(fetchTrades, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, [limit]);

  if (loading && trades.length === 0) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <RefreshCw className="w-12 h-12 text-purple-500 animate-spin mx-auto mb-4" />
          <p className="text-gray-400">Loading trade history...</p>
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
            <h3 className="text-lg font-semibold text-red-500">Error Loading Trades</h3>
            <p className="text-gray-400 mt-1">{error}</p>
            <button
              onClick={fetchTrades}
              className="mt-4 px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg text-white transition-colors"
            >
              Retry
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Calculate statistics
  const highConfidenceTrades = trades.filter(t => (t.confidence || 0) > 0.7).length;
  const avgConfidence = trades.length > 0
    ? (trades.reduce((sum, t) => sum + (t.confidence || 0), 0) / trades.length * 100)
    : 0;

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Trade History</h1>
          <p className="text-gray-400 mt-1">Pipeline execution ledger and analysis results</p>
        </div>
        <button
          onClick={fetchTrades}
          disabled={loading}
          className="flex items-center space-x-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 rounded-lg text-white transition-colors"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-gray-800 bg-opacity-50 rounded-lg border border-gray-700 p-6 card-glow">
          <div className="flex items-center justify-between mb-2">
            <p className="text-gray-400 text-sm">Total Executions</p>
            <Activity className="w-5 h-5 text-blue-500" />
          </div>
          <p className="text-2xl font-bold text-white">{trades.length}</p>
        </div>
        <div className="bg-gray-800 bg-opacity-50 rounded-lg border border-gray-700 p-6 card-glow">
          <div className="flex items-center justify-between mb-2">
            <p className="text-gray-400 text-sm">High Confidence</p>
            <TrendingUp className="w-5 h-5 text-green-500" />
          </div>
          <p className="text-2xl font-bold text-white">{highConfidenceTrades}</p>
          <p className="text-xs text-gray-400 mt-1">&gt;70% confidence</p>
        </div>
        <div className="bg-gray-800 bg-opacity-50 rounded-lg border border-gray-700 p-6 card-glow">
          <div className="flex items-center justify-between mb-2">
            <p className="text-gray-400 text-sm">Avg Confidence</p>
            <DollarSign className="w-5 h-5 text-purple-500" />
          </div>
          <p className="text-2xl font-bold text-white">{avgConfidence.toFixed(1)}%</p>
        </div>
      </div>

      {/* Filters */}
      <div className="flex items-center space-x-4">
        <label className="text-gray-400">Show:</label>
        <select
          value={limit}
          onChange={(e) => setLimit(Number(e.target.value))}
          className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-purple-500"
        >
          <option value={20}>Last 20</option>
          <option value={50}>Last 50</option>
          <option value={100}>Last 100</option>
          <option value={500}>Last 500</option>
        </select>
      </div>

      {/* Trade Table */}
      <div className="bg-gray-800 bg-opacity-50 rounded-lg border border-gray-700 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-900 bg-opacity-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Timestamp
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Symbol
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Confidence
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Regime
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Direction
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Status
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-700">
              {trades.length > 0 ? (
                trades.map((trade, idx) => {
                  const confidence = (trade.confidence || 0) * 100;
                  const isHighConfidence = confidence > 70;
                  const isMediumConfidence = confidence > 50 && confidence <= 70;

                  return (
                    <tr key={idx} className="hover:bg-gray-900 hover:bg-opacity-50 transition-colors">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center text-sm text-gray-300">
                          <Calendar className="w-4 h-4 mr-2 text-gray-500" />
                          {new Date(trade.timestamp || Date.now()).toLocaleString()}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className="text-white font-semibold">{trade.symbol || 'N/A'}</span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center space-x-2">
                          <div className={`w-2 h-2 rounded-full ${
                            isHighConfidence ? 'bg-green-500' :
                            isMediumConfidence ? 'bg-yellow-500' : 'bg-red-500'
                          }`} />
                          <span className={`font-semibold ${
                            isHighConfidence ? 'text-green-400' :
                            isMediumConfidence ? 'text-yellow-400' : 'text-red-400'
                          }`}>
                            {confidence.toFixed(1)}%
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className="px-2 py-1 text-xs font-medium rounded bg-gray-700 text-gray-300">
                          {trade.regime || 'Unknown'}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          {(trade.direction || 'neutral').toLowerCase().includes('bull') ? (
                            <TrendingUp className="w-4 h-4 text-green-500 mr-1" />
                          ) : (trade.direction || 'neutral').toLowerCase().includes('bear') ? (
                            <TrendingDown className="w-4 h-4 text-red-500 mr-1" />
                          ) : null}
                          <span className="text-gray-300">{trade.direction || 'Neutral'}</span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`px-2 py-1 text-xs font-medium rounded ${
                          trade.status === 'executed' ? 'bg-green-900 text-green-300' :
                          trade.status === 'pending' ? 'bg-yellow-900 text-yellow-300' :
                          'bg-blue-900 text-blue-300'
                        }`}>
                          {trade.status || 'Analyzed'}
                        </span>
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan={6} className="px-6 py-12 text-center">
                    <Activity className="w-12 h-12 text-gray-600 mx-auto mb-3" />
                    <p className="text-gray-400">No trade history available</p>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Info */}
      <div className="bg-gradient-to-r from-indigo-900 to-purple-900 bg-opacity-50 rounded-lg border border-indigo-700 p-6">
        <h3 className="text-lg font-semibold text-white mb-2">Trade Ledger</h3>
        <p className="text-gray-300 text-sm">
          Each entry represents a complete pipeline execution with multi-engine analysis.
          Confidence scores combine Physics, Sentiment, Liquidity (PENTA), and Hedge positioning signals
          to generate high-probability trading opportunities.
        </p>
      </div>
    </div>
  );
}
