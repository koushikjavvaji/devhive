const statusEl = document.getElementById("status");
const roomsView = document.getElementById("rooms-view");
const chatView = document.getElementById("chat-view");

const roomMatch = location.pathname.match(/^\/r\/([\w-]+)$/);

if (roomMatch) {
  initChat(roomMatch[1]);
} else {
  initRoomsList();
}

async function initRoomsList() {
  roomsView.hidden = false;

  const listEl = document.getElementById("rooms-list");
  const newRoomBtn = document.getElementById("new-room");

  async function refresh() {
    const rooms = await fetch("/api/rooms").then((r) => r.json());
    listEl.innerHTML = "";

    if (rooms.length === 0) {
      const empty = document.createElement("li");
      empty.className = "empty";
      empty.textContent = "No war rooms yet — start one.";
      listEl.appendChild(empty);
      return;
    }

    for (const room of rooms) {
      const li = document.createElement("li");
      const a = document.createElement("a");
      a.href = `/r/${room.id}`;
      a.className = "room-title";
      a.textContent = room.title || "untitled";

      const meta = document.createElement("span");
      meta.className = "room-meta";
      meta.textContent = `${room.message_count} message${room.message_count === 1 ? "" : "s"}`;

      li.appendChild(a);
      li.appendChild(meta);
      listEl.appendChild(li);
    }
  }

  newRoomBtn.addEventListener("click", async () => {
    const resp = await fetch("/api/rooms", { method: "POST" });
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      alert(body.detail || "Couldn't create a room — try again.");
      return;
    }
    const room = await resp.json();
    location.href = `/r/${room.id}`;
  });

  refresh();
}

function storageGet(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function storageSet(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // private mode / blocked storage — the name just won't be remembered
  }
}

function escapeHtml(text) {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// teammates answer in markdown; marked + DOMPurify come from a CDN, so if either fails to
// load, fall back to escaping everything and only turning ``` fences into <pre> blocks
function renderMarkdown(text) {
  if (window.marked && window.DOMPurify) {
    return DOMPurify.sanitize(marked.parse(text, { breaks: true }));
  }
  return escapeHtml(text).replace(/```(?:[\w+-]+)?\n?([\s\S]*?)```/g, "<pre><code>$1</code></pre>");
}

// reactors end with a "VERDICT: AGREE/DISAGREE" line the server strips from the final
// message — hide it while it's still streaming in too, instead of it flashing up
function stripVerdict(text) {
  return text.replace(/\s*VERDICT:\s*(AGREE|DISAGREE)?\W*\s*$/i, "");
}

function initChat(roomId) {
  chatView.hidden = false;

  const messagesEl = document.getElementById("messages");
  const form = document.getElementById("composer");
  const input = document.getElementById("input");
  const sendBtn = form.querySelector("button[type=submit]");
  const nameInput = document.getElementById("name");

  nameInput.value = storageGet("devhive-name") || "";
  nameInput.addEventListener("change", () => storageSet("devhive-name", nameInput.value.trim()));

  let teammateColors = {};
  // sender -> { el, body, text } for a teammate that's thinking or mid-stream
  const pending = new Map();

  let ws = null;
  let retryDelay = 1000;
  let gone = false;

  function nearBottom() {
    return messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight < 80;
  }

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function append(el) {
    // don't yank the view down while someone's scrolled up reading an earlier answer
    const stick = nearBottom();
    messagesEl.appendChild(el);
    if (stick) scrollToBottom();
  }

  function colorFor(sender) {
    return teammateColors[sender] || "#8a8f98";
  }

  function renderMessage(message) {
    const isTeammate = message.sender_type === "teammate";
    const failed = isTeammate && message.status === "failed";

    const el = document.createElement("div");
    el.className = `msg ${message.sender_type}${failed ? " failed" : ""}`;

    const sender = document.createElement("span");
    sender.className = "sender";
    sender.textContent = message.sender;

    const body = document.createElement("div");
    if (isTeammate && !failed) {
      el.style.borderColor = colorFor(message.sender);
      sender.style.color = colorFor(message.sender);
      body.className = "body md";
      body.innerHTML = renderMarkdown(message.text);
      addCopyButtons(body);
    } else {
      // human messages are mostly pasted tracebacks — markdown would mangle the
      // underscores and asterisks in them, so they stay exactly as typed
      body.className = "body plain";
      body.textContent = message.text;
    }

    el.appendChild(sender);
    el.appendChild(body);
    append(el);
  }

  function addCopyButtons(container) {
    for (const pre of container.querySelectorAll("pre")) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "copy";
      btn.textContent = "copy";
      btn.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(pre.querySelector("code")?.textContent ?? pre.textContent);
          btn.textContent = "copied";
        } catch {
          btn.textContent = "can't copy";
        }
        setTimeout(() => (btn.textContent = "copy"), 1500);
      });
      pre.appendChild(btn);
    }
  }

  function notice(text, kind = "info") {
    const el = document.createElement("div");
    el.className = `notice ${kind}`;
    el.textContent = text;
    append(el);
    return el;
  }

  function pendingFor(sender) {
    let entry = pending.get(sender);
    if (entry) return entry;

    const el = document.createElement("div");
    el.className = "msg teammate typing";
    el.style.borderColor = colorFor(sender);

    const label = document.createElement("span");
    label.className = "sender";
    label.style.color = colorFor(sender);
    label.textContent = sender;

    const body = document.createElement("div");
    body.className = "body md";
    body.textContent = "thinking…";

    el.appendChild(label);
    el.appendChild(body);
    append(el);

    entry = { el, body, text: "" };
    pending.set(sender, entry);
    return entry;
  }

  function appendDelta(sender, chunk) {
    const entry = pendingFor(sender);
    const stick = nearBottom();
    entry.text += chunk;
    entry.el.classList.remove("typing");
    entry.el.classList.add("streaming");
    entry.body.innerHTML = renderMarkdown(stripVerdict(entry.text));
    if (stick) scrollToBottom();
  }

  function clearPending(sender) {
    const entry = pending.get(sender);
    if (entry) {
      entry.el.remove();
      pending.delete(sender);
    }
  }

  function clearAllPending() {
    for (const sender of [...pending.keys()]) clearPending(sender);
  }

  function setConnected(connected) {
    statusEl.textContent = connected ? "connected" : gone ? "room not found" : "reconnecting…";
    statusEl.classList.toggle("connected", connected);
    sendBtn.disabled = !connected;
  }

  function connect() {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${protocol}//${location.host}/ws/${roomId}`);

    ws.onopen = () => {
      retryDelay = 1000;
      setConnected(true);
    };

    ws.onclose = (event) => {
      if (event.code === 4404) {
        gone = true;
        setConnected(false);
        messagesEl.innerHTML = "";
        notice("This war room doesn't exist (or was lost in a server reset). Start a new one from the home page.", "error");
        return;
      }
      setConnected(false);
      // free-tier hosts drop idle sockets and sleep — keep retrying with backoff
      setTimeout(connect, retryDelay);
      retryDelay = Math.min(retryDelay * 2, 15000);
    };

    ws.onmessage = (event) => {
      const payload = JSON.parse(event.data);

      if (payload.type === "history") {
        // a reconnect resends the full history, so start from a clean slate
        teammateColors = Object.fromEntries(payload.teammates.map((t) => [t.name, t.color]));
        clearAllPending();
        messagesEl.innerHTML = "";
        payload.messages.forEach(renderMessage);
        scrollToBottom();
      } else if (payload.type === "typing") {
        pendingFor(payload.sender);
      } else if (payload.type === "delta") {
        appendDelta(payload.sender, payload.text);
      } else if (payload.type === "message") {
        clearPending(payload.message.sender);
        renderMessage(payload.message);
      } else if (payload.type === "idle") {
        clearAllPending();
      } else if (payload.type === "error") {
        notice(payload.text, "error");
      }
    };
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const text = input.value.trim();
    if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;

    ws.send(JSON.stringify({ type: "message", text, name: nameInput.value.trim() }));
    input.value = "";
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  setConnected(false);
  statusEl.textContent = "connecting…";
  connect();
}
