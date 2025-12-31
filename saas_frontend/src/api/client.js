import axios from 'axios';

// Base URL for the API - defaults to localhost:8000 for development
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// API methods
export const api = {
  // Get system health status
  getHealth: async () => {
    const response = await apiClient.get('/api/health');
    return response.data;
  },

  // Get active watchlist
  getWatchlist: async () => {
    const response = await apiClient.get('/api/watchlist');
    return response.data;
  },

  // Get recent trades
  getTrades: async (limit = 20) => {
    const response = await apiClient.get('/api/trades', { params: { limit } });
    return response.data;
  },

  // Trigger pipeline run for a symbol
  runPipeline: async (symbol) => {
    const response = await apiClient.post('/api/run', { symbol });
    return response.data;
  },
};

export default apiClient;
