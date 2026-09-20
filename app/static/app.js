const messagesEl = document.getElementById("messages");
const statusEl = document.getElementById("status");
const form = document.getElementById("composer");
const input = document.getElementById("input");

let teammateColors = {};
const typingEls = new Map();

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function colorFor(sender) {
  return teammateColors[sender] || "#8a8f98";
}

function renderMessage(message) {
  const el = document.createElement("div");
  el.className = `msg ${message.sender_type}`;
  if (message.sender_type === "teammate") {
    el.style.borderColor = colorFor(message.sender);
  }

  const sender = document.createElement("span");
  sender.className = "sender";
  sender.textContent = message.sender;
  if (message.sender_type === "teammate") {
    sender.style.color = colorFor(message.sender);
  }

  const body = document.createElement("div");
  body.innerHTML = renderBody(message.text);

  el.appendChild(sender);
  el.appendChild(body);
  messagesEl.appendChild(el);
  scrollToBottom();
}

function renderBody(text) {
  // minimal, safe formatting: escape html, then turn ```code``` into <pre>
  const escaped = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  return escaped.replace(/```([\s\S]*?)```/g, "<pre><code>$1</code></pre>");
}

function showTyping(sender) {
  if (typingEls.has(sender)) return;
  const el = document.createElement("div");
  el.className = "msg teammate typing";
  el.style.borderColor = colorFor(sender);
  el.textContent = `${sender} is thinking…`;
  messagesEl.appendChild(el);
  typingEls.set(sender, el);
  scrollToBottom();
}

function clearTyping(sender) {
  const el = typingEls.get(sender);
  if (el) {
    el.remove();
    typingEls.delete(sender);
  }
}

const protocol = location.protocol === "https:" ? "wss:" : "ws:";
const ws = new WebSocket(`${protocol}//${location.host}/ws`);

ws.onopen = () => {
  statusEl.textContent = "connected";
  statusEl.classList.add("connected");
};

ws.onclose = () => {
  statusEl.textContent = "disconnected";
  statusEl.classList.remove("connected");
};

ws.onmessage = (event) => {
  const payload = JSON.parse(event.data);

  if (payload.type === "history") {
    teammateColors = Object.fromEntries(payload.teammates.map((t) => [t.name, t.color]));
    payload.messages.forEach(renderMessage);
  } else if (payload.type === "typing") {
    showTyping(payload.sender);
  } else if (payload.type === "message") {
    clearTyping(payload.message.sender);
    renderMessage(payload.message);
  }
};

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text) return;

  ws.send(JSON.stringify({ type: "message", text }));
  input.value = "";
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});
