// Mọi lời gọi backend nằm ở đây. Địa chỉ backend xác định lúc chạy, theo thứ tự ưu tiên:
// ?api=<địa chỉ> (lưu lại cho lần sau) → window.API_URL → VITE_API_URL (lúc build) → rỗng = cùng origin.
const STORAGE_KEY = 'api_base';

function runtimeApi() {
  const asked = new URLSearchParams(window.location.search).get('api');
  if (asked) {
    try { localStorage.setItem(STORAGE_KEY, asked); } catch { /* chế độ riêng tư: bỏ qua */ }
    return asked;
  }
  try { return localStorage.getItem(STORAGE_KEY) ?? window.API_URL ?? null; } catch { return window.API_URL ?? null; }
}

export const API_BASE = (runtimeApi() ?? import.meta.env.VITE_API_URL ?? '').trim().replace(/\/+$/, '');

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
