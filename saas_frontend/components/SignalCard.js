import React from 'react';
import { ArrowUpCircle, ArrowDownCircle, Activity, Shield, BarChart2 } from 'lucide-react';

const SignalCard = ({ signal }) => {
  const isLong = signal.direction === 'long';
  const confidence = signal.composer_decision?.sizing?.confidence || 0.5;
  const score = Math.round(confidence * 100);
  
  // Format price
  const price = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(signal.price);

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 hover:bg-gray-750 transition-colors shadow-lg">
      <div className="flex justify-between items-start mb-4">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-2xl font-bold text-white">{signal.symbol}</h3>
            <span className={`px-2 py-1 rounded text-xs font-bold uppercase ${
              isLong ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'
            }`}>
              {signal.direction}
            </span>
          </div>
          <p className="text-gray-400 text-sm mt-1">{new Date(signal.timestamp).toLocaleString()}</p>
        </div>
        <div className="text-right">
          <div className="text-xl font-mono text-white">{price}</div>
          <div className="text-xs text-gray-500">Entry Price</div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4 mb-4">
        <div className="bg-gray-900/50 p-3 rounded">
          <div className="flex items-center gap-2 text-gray-400 text-xs mb-1">
            <Activity size={14} /> Sentiment
          </div>
          <div className="text-sm font-semibold text-white">
            {signal.sentiment_agent_vote?.risk_posture || 'Neutral'}
          </div>
        </div>
        <div className="bg-gray-900/50 p-3 rounded">
          <div className="flex items-center gap-2 text-gray-400 text-xs mb-1">
            <Shield size={14} /> Hedge
          </div>
          <div className="text-sm font-semibold text-white">
            {signal.hedge_agent_vote?.bias || 'None'}
          </div>
        </div>
        <div className="bg-gray-900/50 p-3 rounded">
          <div className="flex items-center gap-2 text-gray-400 text-xs mb-1">
            <BarChart2 size={14} /> Liq Score
          </div>
          <div className="text-sm font-semibold text-white">
            {signal.options_liq_score || 'N/A'}
          </div>
        </div>
      </div>

      <div className="space-y-2">
        <div className="flex justify-between text-sm">
          <span className="text-gray-400">Confidence Score</span>
          <span className={`font-bold ${score > 70 ? 'text-green-400' : 'text-yellow-400'}`}>
            {score}/100
          </span>
        </div>
        <div className="w-full bg-gray-700 h-2 rounded-full overflow-hidden">
          <div 
            className={`h-full ${score > 70 ? 'bg-green-500' : 'bg-yellow-500'}`} 
            style={{ width: `${score}%` }}
          />
        </div>
      </div>
      
      <div className="mt-4 pt-4 border-t border-gray-700">
        <p className="text-xs text-gray-400 line-clamp-2">
          {signal.composer_decision?.reason_codes?.join(', ') || 'AI Analysis complete based on multi-factor inputs.'}
        </p>
      </div>
    </div>
  );
};

export default SignalCard;
