import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    allowedHosts: [
      'localhost',
      '127.0.0.1',
      '5175-iwr6u76ijtfw2zxk87bmc-82b888ba.sandbox.novita.ai',
    ],
  },
});
