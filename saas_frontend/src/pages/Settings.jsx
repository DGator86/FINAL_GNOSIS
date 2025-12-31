import { useState } from 'react';
import { Settings as SettingsIcon, Save, Server, Database, Bell } from 'lucide-react';

export default function Settings() {
  const [apiUrl, setApiUrl] = useState(import.meta.env.VITE_API_URL || 'http://localhost:8000');
  const [refreshInterval, setRefreshInterval] = useState(30);
  const [notifications, setNotifications] = useState(true);

  const handleSave = () => {
    // In a real app, you'd save these to localStorage or backend
    alert('Settings saved! (Refresh the page to apply changes)');
  };

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-white">Settings</h1>
        <p className="text-gray-400 mt-1">Configure your GNOSIS dashboard preferences</p>
      </div>

      {/* API Configuration */}
      <div className="bg-gray-800 bg-opacity-50 rounded-lg border border-gray-700 p-6">
        <div className="flex items-center space-x-3 mb-6">
          <Server className="w-6 h-6 text-blue-500" />
          <h2 className="text-xl font-semibold text-white">API Configuration</h2>
        </div>

        <div className="space-y-4">
          <div>
            <label htmlFor="apiUrl" className="block text-sm font-medium text-gray-300 mb-2">
              Backend API URL
            </label>
            <input
              type="text"
              id="apiUrl"
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              placeholder="http://localhost:8000"
              className="w-full px-4 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-purple-500"
            />
            <p className="text-xs text-gray-400 mt-1">
              The URL of your GNOSIS backend server
            </p>
          </div>
        </div>
      </div>

      {/* Dashboard Settings */}
      <div className="bg-gray-800 bg-opacity-50 rounded-lg border border-gray-700 p-6">
        <div className="flex items-center space-x-3 mb-6">
          <Database className="w-6 h-6 text-purple-500" />
          <h2 className="text-xl font-semibold text-white">Dashboard Settings</h2>
        </div>

        <div className="space-y-4">
          <div>
            <label htmlFor="refreshInterval" className="block text-sm font-medium text-gray-300 mb-2">
              Auto-Refresh Interval (seconds)
            </label>
            <select
              id="refreshInterval"
              value={refreshInterval}
              onChange={(e) => setRefreshInterval(Number(e.target.value))}
              className="w-full px-4 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-purple-500"
            >
              <option value={10}>10 seconds</option>
              <option value={30}>30 seconds</option>
              <option value={60}>1 minute</option>
              <option value={300}>5 minutes</option>
            </select>
          </div>

          <div className="flex items-center justify-between p-4 bg-gray-900 bg-opacity-50 rounded-lg border border-gray-700">
            <div>
              <p className="text-white font-medium">Enable Notifications</p>
              <p className="text-sm text-gray-400">Receive alerts for high-confidence signals</p>
            </div>
            <button
              onClick={() => setNotifications(!notifications)}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                notifications ? 'bg-purple-600' : 'bg-gray-600'
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  notifications ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>
        </div>
      </div>

      {/* System Information */}
      <div className="bg-gray-800 bg-opacity-50 rounded-lg border border-gray-700 p-6">
        <div className="flex items-center space-x-3 mb-6">
          <SettingsIcon className="w-6 h-6 text-green-500" />
          <h2 className="text-xl font-semibold text-white">System Information</h2>
        </div>

        <div className="space-y-3">
          <InfoRow label="Version" value="1.0.0" />
          <InfoRow label="Frontend" value="React + Vite" />
          <InfoRow label="Backend" value="FastAPI + Python" />
          <InfoRow label="Trading System" value="GNOSIS v3.1" />
        </div>
      </div>

      {/* Save Button */}
      <div className="flex justify-end">
        <button
          onClick={handleSave}
          className="flex items-center space-x-2 px-6 py-3 bg-purple-600 hover:bg-purple-700 rounded-lg text-white font-semibold transition-colors"
        >
          <Save className="w-5 h-5" />
          <span>Save Settings</span>
        </button>
      </div>

      {/* Info */}
      <div className="bg-gradient-to-r from-purple-900 to-indigo-900 bg-opacity-50 rounded-lg border border-purple-700 p-6">
        <h3 className="text-lg font-semibold text-white mb-2">About GNOSIS</h3>
        <p className="text-gray-300 text-sm">
          GNOSIS (Great Neural Optimization System for Intelligent Speculation) is a comprehensive
          algorithmic trading system that combines Physics-based modeling, PENTA liquidity analysis,
          multi-source sentiment, and hedge positioning to generate high-confidence trading signals.
        </p>
      </div>
    </div>
  );
}

function InfoRow({ label, value }) {
  return (
    <div className="flex items-center justify-between p-3 bg-gray-900 bg-opacity-50 rounded border border-gray-700">
      <span className="text-gray-400">{label}</span>
      <span className="text-white font-medium">{value}</span>
    </div>
  );
}
