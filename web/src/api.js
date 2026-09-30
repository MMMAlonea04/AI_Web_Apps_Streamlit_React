// Mọi lời gọi backend nằm ở đây. Địa chỉ backend xác định LÚC CHẠY, theo thứ tự ưu tiên:
// ?api=<địa chỉ> → window.API_URL → gist công bố (VITE_API_DISCOVERY) → localStorage
// → VITE_API_URL (lúc build) → rỗng = cùng origin.
const STORAGE_KEY = 'api_base';
const DISCOVERY_TIMEOUT_MS = 4000;

export let API_BASE = '';

const clean = (base) => String(base ?? '').trim().replace(/\/+$/, '');

function remember(base) {
  try { localStorage.setItem(STORAGE_KEY, base); } catch { /* chế độ riêng tư: bỏ qua */ }
}

function stored() {
  try { return localStorage.getItem(STORAGE_KEY); } catch { return null; }
}

// Gist công bố của phiên Colab đang chạy, do scripts/serve.py cập nhật:
// {"files": {"api.json": {"content": "{\"api\": \"https://…\"}"}}}
async function discovered() {
  const endpoint = import.meta.env.VITE_API_DISCOVERY;
  if (!endpoint) return null;
  try {
    const res = await fetch(endpoint, { cache: 'no-store', signal: AbortSignal.timeout(DISCOVERY_TIMEOUT_MS) });
    if (!res.ok) return null;
    const body = await res.json();
    const content = body?.files?.['api.json']?.content ?? body?.content;
    const api = content ? JSON.parse(content).api : body?.api;
    return api || null;
  } catch {
    return null;
  }
}

export async function resolveApiBase() {
  const asked = clean(new URLSearchParams(window.location.search).get('api'));
  if (asked) {
    remember(asked);
    API_BASE = asked;
    return API_BASE;
  }
  const found = clean(window.API_URL) || await discovered() || clean(stored()) || clean(import.meta.env.VITE_API_URL);
  if (found) remember(found);
  API_BASE = found;
  return API_BASE;
}

async function handle(res) {
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch { /* không phải JSON */ }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

export function postImage(path, file, fields = {}) {
  const form = new FormData();
  form.append('file', file);
  Object.entries(fields).forEach(([k, v]) => form.append(k, String(v)));
  return fetch(`${API_BASE}${path}`, { method: 'POST', body: form }).then(handle);
}

export function postJson(path, body) {
  return fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then(handle);
}

export const getHealth = () => fetch(`${API_BASE}/api/health`).then(handle);

// Đọc Server-Sent Events từ POST /api/chat. EventSource không hỗ trợ POST nên đọc stream thủ công.
export async function streamChat({ message, history, onSources, onToken, signal }) {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, history }),
    signal,
  });
  if (!res.ok) throw new Error(`${res.status}: ${res.statusText}`);
  const reader = res.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split('\n\n');
    buffer = events.pop(); // phần chưa trọn vẹn giữ lại cho lần đọc sau
    for (const ev of events) {
      if (!ev.startsWith('data: ')) continue;
      const data = JSON.parse(ev.slice(6));
      if (data.type === 'sources') onSources?.(data.items);
      if (data.type === 'token') onToken?.(data.text);
    }
  }
}
