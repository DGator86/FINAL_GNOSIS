import { useState } from 'react';
import { api } from '../api/client';
import {
  Play,
  Search,
  CheckCircle,
  XCircle,
  AlertCircle,
  Zap,
  TrendingUp,
  Activity,
  Gauge
} from 'lucide-react';

export default function Pipeline() {
  const [symbol, setSymbol] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);

  const handleRun = async (e) => {
    e.preventDefault();
    if (!symbol.trim()) {
      setError('Please enter a symbol');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setResult(null);

      const data = await api.runPipeline(symbol.toUpperCase());
      setResult(data);

      // Add to history
      setHistory(prev => [{
        symbol: symbol.toUpperCase(),
        timestamp: new Date().toISOString(),
        result: data,
      }, ...prev.slice(0, 9)]); // Keep last 10

      setSymbol(''); // Clear input
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to run pipeline. Make sure the backend is running.');
      console.error('Error running pipeline:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-white">Pipeline Execution</h1>
        <p className="text-gray-400 mt-1">Run the GNOSIS trading pipeline for any symbol</p>
      </div>

      {/* Execution Form */}
      <div className="bg-gradient-to-br from-purple-900 to-indigo-900 bg-opacity-30 rounded-lg border border-purple-700 p-8 card-glow">
        <form onSubmit={handleRun} className="space-y-6">
          <div>
            <label htmlFor="symbol" className="block text-sm font-medium text-gray-300 mb-2">
              Symbol
            </label>
            <div className="relative">
              <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                id="symbol"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                placeholder="Enter symbol (e.g., SPY, AAPL, TSLA)"
                className="w-full pl-12 pr-4 py-4 bg-gray-900 border border-gray-700 rounded-lg text-white text-lg placeholder-gray-500 focus:outline-none focus:border-purple-500 transition-colors"
                disabled={loading}
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading || !symbol.trim()}
            className="w-full flex items-center justify-center space-x-3 px-6 py-4 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg text-white font-semibold text-lg transition-colors"
          >
            {loading ? (
              <>
                <Activity className="w-6 h-6 animate-spin" />
                <span>Running Pipeline...</span>
              </>
            ) : (
              <>
                <Play className="w-6 h-6" />
                <span>Execute Pipeline</span>
              </>
            )}
          </button>
        </form>

        {/* Error Display */}
        {error && (
          <div className="mt-6 bg-red-900 bg-opacity-30 border border-red-500 rounded-lg p-4">
            <div className="flex items-center space-x-3">
              <XCircle className="w-6 h-6 text-red-500 flex-shrink-0" />
              <p className="text-red-300">{error}</p>
            </div>
          </div>
        )}

        {/* Result Display */}
        {result && (
          <div className="mt-6 space-y-4">
            <div className="bg-green-900 bg-opacity-30 border border-green-500 rounded-lg p-4">
              <div className="flex items-center space-x-3">
                <CheckCircle className="w-6 h-6 text-green-500" />
                <div>
                  <h3 className="text-lg font-semibold text-green-300">Pipeline Executed Successfully</h3>
                  <p className="text-gray-300 text-sm mt-1">{result.message || 'Analysis complete'}</p>
                </div>
              </div>
            </div>

            {result.analysis && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {result.analysis.confidence !== undefined && (
                  <ResultCard
                    icon={Gauge}
                    title="Confidence Score"
                    value={`${(result.analysis.confidence * 100).toFixed(1)}%`}
                    color={result.analysis.confidence > 0.7 ? 'green' : result.analysis.confidence > 0.5 ? 'yellow' : 'red'}
                  />
                )}
                {result.analysis.regime && (
                  <ResultCard
                    icon={Activity}
                    title="Market Regime"
                    value={result.analysis.regime}
                    color="blue"
                  />
                )}
                {result.analysis.direction && (
                  <ResultCard
                    icon={TrendingUp}
                    title="Direction"
                    value={result.analysis.direction}
                    color={result.analysis.direction.toLowerCase().includes('bull') ? 'green' : 'red'}
                  />
                )}
                {result.analysis.strategy && (
                  <ResultCard
                    icon={Zap}
                    title="Strategy"
                    value={result.analysis.strategy}
                    color="purple"
                  />
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Pipeline Info */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <InfoCard
          title="Physics Engine"
          description="Analyzes price as a particle with mass (market cap), velocity (momentum), and energy (volume)"
          icon={Activity}
        />
        <InfoCard
          title="PENTA Liquidity"
          description="5-engine confluence analysis using Wyckoff, ICT, Order Flow, Supply/Demand, and Liquidity concepts"
          icon={Zap}
        />
        <InfoCard
          title="Sentiment Analysis"
          description="Multi-source sentiment from technical indicators, news, options flow, and social media"
          icon={TrendingUp}
        />
      </div>

      {/* Execution History */}
      {history.length > 0 && (
        <div className="bg-gray-800 bg-opacity-50 rounded-lg border border-gray-700 p-6">
          <h2 className="text-xl font-semibold text-white mb-4">Recent Executions</h2>
          <div className="space-y-3">
            {history.map((item, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between p-4 bg-gray-900 bg-opacity-50 rounded-lg border border-gray-700"
              >
                <div className="flex items-center space-x-4">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <div>
                    <p className="text-white font-semibold">{item.symbol}</p>
                    <p className="text-xs text-gray-400">{new Date(item.timestamp).toLocaleString()}</p>
                  </div>
                </div>
                {item.result.analysis?.confidence && (
                  <div className="text-right">
                    <p className="text-sm text-gray-400">Confidence</p>
                    <p className="text-white font-semibold">
                      {(item.result.analysis.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Instructions */}
      <div className="bg-gradient-to-r from-blue-900 to-indigo-900 bg-opacity-50 rounded-lg border border-blue-700 p-6">
        <h3 className="text-lg font-semibold text-white mb-3 flex items-center">
          <AlertCircle className="w-5 h-5 mr-2" />
          How It Works
        </h3>
        <div className="space-y-2 text-sm text-gray-300">
          <p>1. Enter a stock symbol (e.g., SPY, AAPL, NVDA)</p>
          <p>2. The pipeline will analyze the symbol using multiple engines in parallel</p>
          <p>3. Results include market regime, direction, strategy recommendation, and confidence score</p>
          <p>4. Each execution is logged to the trade ledger for historical analysis</p>
        </div>
      </div>
    </div>
  );
}

function ResultCard({ icon: Icon, title, value, color }) {
  const colors = {
    green: 'border-green-500 bg-green-900',
    red: 'border-red-500 bg-red-900',
    blue: 'border-blue-500 bg-blue-900',
    purple: 'border-purple-500 bg-purple-900',
    yellow: 'border-yellow-500 bg-yellow-900',
  };

  const iconColors = {
    green: 'text-green-400',
    red: 'text-red-400',
    blue: 'text-blue-400',
    purple: 'text-purple-400',
    yellow: 'text-yellow-400',
  };

  return (
    <div className={`${colors[color]} bg-opacity-30 border rounded-lg p-4`}>
      <div className="flex items-center space-x-3 mb-2">
        <Icon className={`w-5 h-5 ${iconColors[color]}`} />
        <p className="text-gray-300 text-sm">{title}</p>
      </div>
      <p className="text-xl font-bold text-white">{value}</p>
    </div>
  );
}

function InfoCard({ title, description, icon: Icon }) {
  return (
    <div className="bg-gray-800 bg-opacity-50 rounded-lg border border-gray-700 p-6 card-glow">
      <div className="flex items-center space-x-3 mb-3">
        <Icon className="w-6 h-6 text-purple-500" />
        <h3 className="text-lg font-semibold text-white">{title}</h3>
      </div>
      <p className="text-gray-400 text-sm">{description}</p>
    </div>
  );
}
