import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.jsx';
import { resolveApiBase } from './api.js';
import './styles.css';

// Biết địa chỉ backend trước khi render: gist công bố của phiên Colab có thể vừa đổi link tunnel.
resolveApiBase().finally(() => {
  createRoot(document.getElementById('root')).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
});
