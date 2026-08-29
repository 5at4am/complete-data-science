(function () {
  const thread = document.getElementById("thread");
  const form = document.getElementById("form");
  const input = document.getElementById("input");
  const send = document.getElementById("send");

  function addMsg(text, kind) {
    const el = document.createElement("div");
    el.className = "msg " + kind;
    el.textContent = text;
    thread.appendChild(el);
    thread.scrollTop = thread.scrollHeight;
    return el;
  }

  async function chat() {
    const q = input.value.trim();
    if (!q) return;
    input.value = "";
    send.disabled = true;
    addMsg(q, "user");

    let res;
    try {
      const r = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: q }),
      });
      res = await r.json();
    } catch (e) {
      addMsg("Network error — is the API running?", "bot refused");
      send.disabled = false;
      return;
    }
    send.disabled = false;

    const bot = addMsg(res.answer, res.refused ? "bot refused" : "bot");

    const guards = (res.guard_results || [])
      .filter((g) => g.status !== "not_run")
      .map((g) => `${g.name}:${g.status} (${g.score})`)
      .join("  ·  ");
    const guardEl = document.createElement("div");
    guardEl.className = "guard";
    guardEl.textContent = guards ? "guards  " + guards : "";
    bot.appendChild(guardEl);

    const cites = (res.citations || []).slice(0, 3);
    if (cites.length && !res.refused) {
      const c = document.createElement("div");
      c.className = "cites";
      c.innerHTML = cites
        .map((x) => `<a href="${x.source_url}" target="_blank" rel="noopener">${x.source_title}</a>`)
        .join(" · ");
      bot.appendChild(c);
    }
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    chat();
  });
})();