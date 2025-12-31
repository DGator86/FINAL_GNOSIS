import { useEffect, useState } from 'react';
import { api } from '../api/client';
import {
  TrendingUp,
  TrendingDown,
  Activity,
  BarChart3,
  RefreshCw,
  AlertCircle,
  CheckCircle,
  Zap
} from 'lucide-react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function Dashboard() {
  const [health, setHealth] = useState(null);
  const [trades, setTrades] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(new Date());

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);
      const [healthData, tradesData] = await Promise.all([
        api.getHealth(),
        api.getTrades(10)
      ]);
      setHealth(healthData);
      setTrades(tradesData);
      setLastUpdate(new Date());
    } catch (err) {
      setError('Failed to fetch data. Make sure the backend is running on http://localhost:8000');
      console.error('Error fetching data:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  if (loading && !health) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <RefreshCw className="w-12 h-12 text-purple-500 animate-spin mx-auto mb-4" />
          <p className="text-gray-400">Loading system data...</p>
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
            <h3 className="text-lg font-semibold text-red-500">Connection Error</h3>
            <p className="text-gray-400 mt-1">{error}</p>
            <button
              onClick={fetchData}
              className="mt-4 px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg text-white transition-colors"
            >
              Retry Connection
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Calculate simple statistics
  const totalTrades = trades.length;
  const recentPnL = trades.slice(0, 5).reduce((sum, t) => {
    const confidence = t.confidence || 0;
    return sum + (confidence > 0.6 ? 1 : -0.5);
  }, 0);

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">System Dashboard</h1>
          <p className="text-gray-400 mt-1">Real-time trading intelligence and analytics</p>
        </div>
        <button
          onClick={fetchData}
          disabled={loading}
          className="flex items-center space-x-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 rounded-lg text-white transition-colors"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="System Status"
          value={health?.ok ? 'Active' : 'Offline'}
          icon={health?.ok ? CheckCircle : AlertCircle}
          trend={health?.ok ? 'positive' : 'negative'}
          color={health?.ok ? 'green' : 'red'}
        />
        <StatCard
          title="Active Symbols"
          value={health?.watchlist_size || 0}
          icon={BarChart3}
          trend="neutral"
          color="blue"
        />
        <StatCard
          title="Recent Trades"
          value={totalTrades}
          icon={Activity}
          trend="neutral"
          color="purple"
        />
        <StatCard
          title="Performance"
          value={recentPnL > 0 ? '+' + recentPnL.toFixed(1) : recentPnL.toFixed(1)}
          icon={recentPnL > 0 ? TrendingUp : TrendingDown}
          trend={recentPnL > 0 ? 'positive' : 'negative'}
          color={recentPnL > 0 ? 'green' : 'red'}
        />
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* System Health Details */}
        <div className="bg-gray-800 bg-opacity-50 rounded-lg border border-gray-700 p-6 card-glow">
          <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
            <Zap className="w-5 h-5 mr-2 text-yellow-500" />
            System Health
          </h2>
          <div className="space-y-3">
            <HealthItem label="API Status" value={health?.ok ? 'Connected' : 'Disconnected'} status={health?.ok} />
            <HealthItem label="Config Loaded" value={health?.config_loaded ? 'Yes' : 'No'} status={health?.config_loaded} />
            <HealthItem label="Watchlist Size" value={health?.watchlist_size || 0} status={true} />
            <HealthItem label="Ledger Entries" value={health?.ledger_size || 0} status={true} />
            <HealthItem label="Last Update" value={lastUpdate.toLocaleTimeString()} status={true} />
          </div>
        </div>

        {/* Recent Activity */}
        <div className="bg-gray-800 bg-opacity-50 rounded-lg border border-gray-700 p-6 card-glow">
          <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
            <Activity className="w-5 h-5 mr-2 text-blue-500" />
            Recent Pipeline Activity
          </h2>
          {trades.length > 0 ? (
            <div className="space-y-3">
              {trades.slice(0, 5).map((trade, idx) => (
                <div key={idx} className="flex items-center justify-between p-3 bg-gray-900 bg-opacity-50 rounded border border-gray-700">
                  <div className="flex items-center space-x-3">
                    <div className={`w-2 h-2 rounded-full ${
                      (trade.confidence || 0) > 0.7 ? 'bg-green-500' :
                      (trade.confidence || 0) > 0.5 ? 'bg-yellow-500' : 'bg-red-500'
                    }`} />
                    <div>
                      <p className="text-white font-medium">{trade.symbol || 'N/A'}</p>
                      <p className="text-xs text-gray-400">{trade.timestamp || new Date().toISOString()}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-gray-300">Confidence</p>
                    <p className="text-white font-semibold">{((trade.confidence || 0) * 100).toFixed(0)}%</p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center text-gray-400 py-8">
              <Activity className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p>No recent activity</p>
            </div>
          )}
        </div>
      </div>

      {/* Key Metrics Info */}
      <div className="bg-gradient-to-r from-purple-900 to-indigo-900 bg-opacity-50 rounded-lg border border-purple-700 p-6">
        <h2 className="text-xl font-semibold text-white mb-4">GNOSIS Trading Intelligence</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <MetricInfo
            title="Physics Engine"
            description="Price-as-Particle modeling with mass, velocity, and momentum analysis"
          />
          <MetricInfo
            title="PENTA Methodology"
            description="5-engine confluence: Wyckoff, ICT, Order Flow, Supply/Demand, Liquidity"
          />
          <MetricInfo
            title="Multi-Timeframe"
            description="Coordinated analysis across 1H, 4H, 1D, and 1W timeframes"
          />
        </div>
      </div>
    </div>
  );
}

function StatCard({ title, value, icon: Icon, trend, color }) {
  const colors = {
    green: 'border-green-500 bg-green-900',
    red: 'border-red-500 bg-red-900',
    blue: 'border-blue-500 bg-blue-900',
    purple: 'border-purple-500 bg-purple-900',
  };

  const iconColors = {
    green: 'text-green-500',
    red: 'text-red-500',
    blue: 'text-blue-500',
    purple: 'text-purple-500',
  };

  return (
    <div className={`${colors[color]} bg-opacity-20 border rounded-lg p-6 card-glow`}>
      <div className="flex items-center justify-between mb-2">
        <p className="text-gray-400 text-sm">{title}</p>
        <Icon className={`w-5 h-5 ${iconColors[color]}`} />
      </div>
      <p className="text-2xl font-bold text-white">{value}</p>
    </div>
  );
}

function HealthItem({ label, value, status }) {
  return (
    <div className="flex items-center justify-between p-3 bg-gray-900 bg-opacity-50 rounded border border-gray-700">
      <span className="text-gray-400">{label}</span>
      <div className="flex items-center space-x-2">
        <span className="text-white font-medium">{value}</span>
        <div className={`w-2 h-2 rounded-full ${status ? 'bg-green-500' : 'bg-red-500'}`} />
      </div>
    </div>
  );
}

function MetricInfo({ title, description }) {
  return (
    <div>
      <h3 className="text-white font-semibold mb-1">{title}</h3>
      <p className="text-gray-300 text-sm">{description}</p>
    </div>
  );
}
