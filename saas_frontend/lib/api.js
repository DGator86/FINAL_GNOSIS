/**
 * API client for Gnosis backend
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

/**
 * Fetch signals (trade decisions) from the backend
 * @param {Object} options - Query options
 * @param {string} options.mode - Filter by mode: 'live', 'paper', or 'backtest'
 * @param {number} options.limit - Maximum number of results
 * @returns {Promise<Array>} Array of signal objects
 */
export async function fetchSignals({ mode, limit = 20 } = {}) {
  try {
    const params = new URLSearchParams();
    if (mode) params.append('mode', mode);
    if (limit) params.append('limit', limit.toString());

    const url = `${API_BASE_URL}/trades/decisions?${params.toString()}`;
    const response = await fetch(url);

    if (!response.ok) {
      console.error(`API error: ${response.status}`);
      return [];
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Failed to fetch signals:', error);
    return [];
  }
}

/**
 * Fetch a single trade decision by ID
 * @param {string} tradeId - UUID of the trade decision
 * @returns {Promise<Object|null>} Trade decision object or null
 */
export async function fetchSignalById(tradeId) {
  try {
    const url = `${API_BASE_URL}/trades/decisions/${tradeId}`;
    const response = await fetch(url);

    if (!response.ok) {
      console.error(`API error: ${response.status}`);
      return null;
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Failed to fetch signal:', error);
    return null;
  }
}

/**
 * Get backend health status
 * @returns {Promise<Object>} Health status object
 */
export async function getHealthStatus() {
  try {
    const url = `${API_BASE_URL}/health`;
    const response = await fetch(url);

    if (!response.ok) {
      return { status: 'unhealthy' };
    }

    const data = await response.json();
    return data;
  } catch (error) {
    return { status: 'error', error: error.message };
  }
}
