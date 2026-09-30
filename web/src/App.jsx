import { useEffect, useState } from 'react';
import { API_BASE, API_SOURCE, getHealth } from './api.js';
import Classify from './features/Classify.jsx';
import Detect from './features/Detect.jsx';
import Search from './features/Search.jsx';
import Chat from './features/Chat.jsx';

const TABS = [
  { id: 'classify', label: 'Phân loại ảnh', model: 'classifier', Component: Classify },
  { id: 'detect', label: 'Phát hiện đối tượng', model: 'detector', Component: Detect },
  { id: 'search', label: 'Tìm kiếm ảnh', model: 'retrieval', Component: Search },
  { id: 'chat', label: 'Chatbot RAG', model: 'llm', Component: Chat },
];

const SOURCE_LABEL = {
  query: 'từ ?api=',
  window: 'từ window.API_URL',
  gist: 'từ gist công bố',
  storage: 'nhớ từ lần trước',
  build: 'từ lúc build',
  origin: 'cùng origin',
};

// Backend có thể là link tunnel Colab vừa khởi động (nạp mô hình mất 1–3 phút) nên phải hỏi lại.
const RETRY_MS = 3000;
const MAX_TRIES = 20;

export default function App() {
  const [tab, setTab] = useState('classify');
  const [health, setHealth] = useState(null);
  const [tries, setTries] = useState(0);

  useEffect(() => {
    let stop = false;
    let timer;

    async function check(n) {
      try {
        const body = await getHealth();
        if (!stop) setHealth(body);
      } catch {
        if (stop) return;
        setTries(n);
        if (n < MAX_TRIES) timer = setTimeout(() => check(n + 1), RETRY_MS);
        else setHealth({ status: 'down', models: {} });
      }
    }

    check(1);
    return () => { stop = true; clearTimeout(timer); };
  }, []);

  const current = TABS.find((t) => t.id === tab);
  const online = health?.status === 'ok';
  const ready = online && health.models?.[current.model];
  const status = online ? `đang chạy (${health.device})`
    : health ? 'không kết nối'
      : tries ? `đang khởi động… (thử lại ${tries}/${MAX_TRIES})` : 'đang kiểm tra…';

  return (
    <div className="app">
      <header>
        <h1>AI Web Apps</h1>
        <p className="muted">Backend: {status}</p>
        <p className="muted">
          API: {API_BASE || 'cùng origin'}{API_BASE ? ` · ${SOURCE_LABEL[API_SOURCE]}` : ''}
        </p>
      </header>
      <nav className="tabs" role="tablist">
        {TABS.map((t) => (
          <button key={t.id} role="tab" aria-selected={tab === t.id} className={tab === t.id ? 'active' : ''}
                  onClick={() => setTab(t.id)}>
            {t.label}{online && !health.models?.[t.model] ? ' (tắt)' : ''}
          </button>
        ))}
      </nav>
      <main>
        {health?.status === 'down' && (
          <p className="error">
            Không gọi được backend ({API_BASE || 'cùng origin'}). Kiểm <code>CORS_ORIGINS</code> phía API,
            hoặc trỏ sang địa chỉ khác bằng <code>?api=&lt;địa chỉ&gt;</code> — với bản deploy tĩnh còn có
            thể đặt nguồn gist bằng <code>?discovery=&lt;url gist&gt;</code>.
          </p>
        )}
        {online && !ready && <p className="error">Mô hình “{current.model}” chưa được nạp ở backend.</p>}
        <current.Component />
      </main>
    </div>
  );
}
