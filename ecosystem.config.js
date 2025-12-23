/**
 * PM2 Ecosystem Configuration for GNOSIS SaaS
 * 
 * Usage:
 *   pm2 start ecosystem.config.js
 *   pm2 save
 *   pm2 startup
 */

module.exports = {
  apps: [
    {
      name: 'gnosis-saas',
      script: 'scripts/gnosis_service.py',
      interpreter: 'python3',
      cwd: '/home/root/webapp',
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '500M',
      env: {
        GNOSIS_PORT: 8888,
        GNOSIS_HOST: '0.0.0.0',
        PYTHONUNBUFFERED: '1'
      },
      error_file: 'logs/pm2_error.log',
      out_file: 'logs/pm2_out.log',
      log_file: 'logs/pm2_combined.log',
      time: true,
      // Restart on failure
      exp_backoff_restart_delay: 1000,
      max_restarts: 10,
      min_uptime: '10s',
    }
  ]
};
