// Mọi lời gọi backend nằm ở đây. Địa chỉ backend xác định LÚC CHẠY, theo thứ tự ưu tiên:
// ?api=<địa chỉ> → window.API_URL → gist công bố → localStorage → VITE_API_URL (lúc build) → cùng origin.
// Nguồn gist lấy từ ?discovery=<url gist> (nhớ cho lần sau) → localStorage → VITE_API_DISCOVERY (lúc build).
const STORAGE_KEY = 'api_base';
const DISCOVERY_KEY = 'api_discovery';
const DISCOVERY_TIMEOUT_MS = 4000;

export let API_BASE = '';
export let API_SOURCE = '';

const clean = (value) => String(value ?? '').trim().replace(/\/+$/, '');

function stored(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}

function remember(key, value) {
  try { localStorage.setItem(key, value); } catch { /* chế độ riêng tư: bỏ qua */ }
}

function discoveryEndpoint() {
  const asked = clean(new URLSearchParams(window.location.search).get('discovery'));
  if (asked) {
    remember(DISCOVERY_KEY, asked);
    return asked;
  }
  return clean(stored(DISCOVERY_KEY)) || clean(import.meta.env.VITE_API_DISCOVERY);
}

// Gist công bố của phiên Colab đang chạy, do scripts/serve.py cập nhật:
// {"files": {"api.json": {"content": "{\"api\": \"https://…\"}"}}}
async function discovered() {
  const endpoint = discoveryEndpoint();
  if (!endpoint) return null;
  try {
    const res = await fetch(endpoint, { cache: 'no-store', signal: AbortSignal.timeout(DISCOVERY_TIMEOUT_MS) });
    if (!res.ok) return null;
    const body = await res.json();
    const files = Object.values(body?.files ?? {});
    const chosen = body?.files?.['api.json'] ?? files.find((f) => /\.json$/i.test(f?.filename ?? '')) ?? files[0];
    const text = chosen?.content ?? body?.content;
    const api = text ? JSON.parse(text).api : body?.api;
    return api || null;
  } catch {
    return null;
  }
}

export async function resolveApiBase() {
  const wanted = new URLSearchParams(window.location.search);

  const asked = clean(wanted.get('api'));
  if (asked) return useBase('query', asked);

  const explicit = clean(window.API_URL);
  if (explicit) return useBase('window', explicit);

  const fromGist = clean(await discovered());
  if (fromGist) return useBase('gist', fromGist);

  const cached = clean(stored(STORAGE_KEY));
  if (cached) return useBase('storage', cached);

  const built = clean(import.meta.env.VITE_API_URL);
  return useBase(built ? 'build' : 'origin', built);
}

function useBase(source, base) {
  API_SOURCE = source;
  API_BASE = base;
  if (base) remember(STORAGE_KEY, base);
  return base;
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
