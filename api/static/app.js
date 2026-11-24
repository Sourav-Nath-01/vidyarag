// VidyaRAG search page — talks to the same-origin FastAPI endpoints.
// All result text is inserted with textContent / text nodes (never innerHTML),
// so transcript or OCR content can't inject markup.
"use strict";

(() => {
  const $ = (sel) => document.querySelector(sel);
  const form = $("#search-form");
  const input = $("#q");
  const button = $("#search-btn");
  const topk = $("#topk");
  const useBm25 = $("#opt-bm25");
  const useRerank = $("#opt-rerank");
  const statusEl = $("#status");
  const resultsEl = $("#results");
  const introEl = $("#intro");

  const STOPWORDS = new Set((
    "the and but for with from into about how what why when which who does did are was were " +
    "this that these those can will would should explain define difference between using use"
  ).split(" "));

  // ── helpers ──────────────────────────────────────────────────────────────
  function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
      if (value === null || value === undefined || value === false) continue;
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else node.setAttribute(key, value === true ? "" : String(value));
    }
    for (const child of children) if (child !== null && child !== undefined) node.append(child);
    return node;
  }

  function fmtTime(seconds) {
    const total = Math.max(0, Math.floor(Number(seconds) || 0));
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = String(total % 60).padStart(2, "0");
    return h ? `${h}:${String(m).padStart(2, "0")}:${s}` : `${m}:${s}`;
  }

  function videoId(url) {
    const match = /(?:[?&]v=|youtu\.be\/)([\w-]{11})/.exec(url || "");
    return match ? match[1] : null;
  }

  function queryTerms(query) {
    const words = query.toLowerCase().match(/[a-z0-9]{3,}/g) || [];
    return [...new Set(words)].filter((w) => !STOPWORDS.has(w)).slice(0, 12);
  }

  // Wrap occurrences of the query terms in <mark>, building text nodes only.
  function highlighted(text, terms) {
    const frag = document.createDocumentFragment();
    if (!terms.length) { frag.append(text); return frag; }
    const escaped = terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    const re = new RegExp(`\\b(${escaped.join("|")})\\w*`, "gi");
    let last = 0;
    for (const match of text.matchAll(re)) {
      if (match.index > last) frag.append(text.slice(last, match.index));
      frag.append(el("mark", { text: match[0] }));
      last = match.index + match[0].length;
    }
    frag.append(text.slice(last));
    return frag;
  }

  function setStatus(message, kind = "") {
    statusEl.textContent = message;
    statusEl.dataset.kind = kind;
  }

  // ── result card ─────────────────────────────────────────────────────────
  function playInline(card, media, id, start) {
    const src = `https://www.youtube-nocookie.com/embed/${id}?start=${start}&autoplay=1&rel=0`;
    media.replaceChildren(el("iframe", {
      src,
      title: "Lecture video",
      allow: "autoplay; encrypted-media; picture-in-picture; fullscreen",
      allowfullscreen: true,
    }));
    card.classList.add("is-playing");
  }

  function renderCard(result, terms) {
    const id = videoId(result.youtube_url);
    const start = Math.floor(Number(result.start_sec) || 0);
    const title = result.lecture_title || "Untitled lecture";
    const media = el("div", { class: "media" });
    const card = el("article", { class: "card" });

    if (id) {
      const thumb = el("button", {
        class: "thumb", type: "button",
        "aria-label": `Play “${title}” from ${fmtTime(start)}`,
      },
        el("img", { src: `https://i.ytimg.com/vi/${id}/mqdefault.jpg`, alt: "", loading: "lazy", width: 320, height: 180 }),
        el("span", { class: "play-icon", "aria-hidden": "true" }),
        el("span", { class: "ts", text: fmtTime(start) }),
      );
      thumb.addEventListener("click", () => playInline(card, media, id, start));
      media.append(thumb);
    } else {
      media.append(el("div", { class: "thumb thumb-empty", text: "No video link" }));
    }

    const chips = el("div", { class: "chips" },
      el("span", { class: "chip", text: result.course_name || "Unknown course" }),
      result.content_type
        ? el("span", { class: `chip type type-${result.content_type}`, text: result.content_type })
        : null,
    );

    const heading = el("h3", { class: "title" }, el("span", { class: "rank", text: `#${result.rank}` }), title);

    const metaParts = [result.instructor, `${fmtTime(result.start_sec)} – ${fmtTime(result.end_sec)}`].filter(Boolean);
    const meta = el("p", { class: "meta", text: metaParts.join(" · ") });

    const transcriptText = (result.transcript || "").trim();
    const transcript = el("p", { class: "transcript clamp" }, highlighted(transcriptText, terms));
    let toggle = null;
    if (transcriptText.length > 240) {
      toggle = el("button", { class: "linklike", type: "button", "aria-expanded": "false", text: "Show full transcript" });
      toggle.addEventListener("click", () => {
        const expanded = !transcript.classList.toggle("clamp");
        toggle.textContent = expanded ? "Show less" : "Show full transcript";
        toggle.setAttribute("aria-expanded", String(expanded));
      });
    }

    const ocrText = (result.ocr_text || "").trim();
    const ocr = ocrText
      ? el("details", { class: "ocr" }, el("summary", { text: "Slide text (OCR)" }), el("p", {}, highlighted(ocrText, terms)))
      : null;

    const actions = el("div", { class: "actions" });
    if (id) {
      const play = el("button", { class: "btn small", type: "button", text: `▶ Play from ${fmtTime(start)}` });
      play.addEventListener("click", () => playInline(card, media, id, start));
      actions.append(play);
    }
    if (result.youtube_deep_link) {
      actions.append(el("a", {
        class: "btn small ghost", href: result.youtube_deep_link,
        target: "_blank", rel: "noopener noreferrer", text: "Open on YouTube ↗",
      }));
    }
    actions.append(el("span", {
      class: "score", title: "Cross-encoder relevance score",
      text: `score ${Number(result.retrieval_score).toFixed(2)}`,
    }));

    card.append(media, el("div", { class: "body" }, chips, heading, meta, transcript, toggle, ocr, actions));
    return card;
  }

  // ── search ──────────────────────────────────────────────────────────────
  let inflight = null;

  async function search(query, { updateUrl = true } = {}) {
    query = query.trim();
    if (query.length < 2) {
      setStatus("Type at least 2 characters.", "warn");
      input.focus();
      return;
    }
    if (inflight) inflight.abort();
    const controller = new AbortController();
    inflight = controller;

    button.disabled = true;
    resultsEl.setAttribute("aria-busy", "true");
    setStatus("Searching…", "busy");
    const slowHint = setTimeout(
      () => setStatus("Still working — if nobody has searched for a while, the server takes a few seconds to wake up…", "busy"),
      6000,
    );
    const started = performance.now();

    try {
      const response = await fetch("/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          top_k: Number(topk.value),
          use_bm25: useBm25.checked,
          use_rerank: useRerank.checked,
        }),
        signal: controller.signal,
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = typeof body.detail === "string"
          ? body.detail
          : response.status === 422 ? "Please enter a question between 2 and 512 characters." : `HTTP ${response.status}`;
        throw new Error(detail);
      }

      const terms = queryTerms(query);
      introEl.hidden = true;
      resultsEl.replaceChildren(...body.results.map((r) => renderCard(r, terms)));
      if (!body.results.length) {
        resultsEl.append(el("p", { class: "empty", text: "No matching lecture segments. Try rephrasing the question." }));
      }

      // Server-side pipeline time (embed → retrieve → fuse → rerank), not network.
      const seconds = Number(body.latency_sec ?? (performance.now() - started) / 1000).toFixed(1);
      const intent = body.results[0] && body.results[0].query_intent;
      setStatus(
        `${body.n_results} result${body.n_results === 1 ? "" : "s"} · searched in ${seconds} s` +
        (intent ? ` · detected intent: ${intent}` : ""),
        "ok",
      );

      if (updateUrl) {
        const url = new URL(window.location.href);
        url.searchParams.set("q", query);
        if (url.href !== window.location.href) history.pushState({ q: query }, "", url);
      }
      document.title = `${query} — VidyaRAG`;
    } catch (err) {
      if (err.name === "AbortError") return;
      const network = err instanceof TypeError;
      setStatus(
        network
          ? "Couldn't reach the server — it may still be waking up. Try again in a few seconds."
          : `Search failed: ${err.message}`,
        "error",
      );
    } finally {
      clearTimeout(slowHint);
      if (inflight === controller) {
        inflight = null;
        button.disabled = false;
        resultsEl.removeAttribute("aria-busy");
      }
    }
  }

  // ── wiring ──────────────────────────────────────────────────────────────
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    search(input.value);
  });

  for (const example of document.querySelectorAll(".example")) {
    example.addEventListener("click", () => {
      input.value = example.textContent;
      search(input.value);
    });
  }

  document.addEventListener("keydown", (event) => {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName);
    if (event.key === "/" && !typing) {
      event.preventDefault();
      input.focus();
      input.select();
    }
  });

  // Shareable searches: /?q=backpropagation runs the search on load.
  function runFromUrl() {
    const q = new URL(window.location.href).searchParams.get("q");
    if (q) {
      input.value = q;
      search(q, { updateUrl: false });
    }
  }
  window.addEventListener("popstate", runFromUrl);
  runFromUrl();

  // Build identifier in the footer.
  fetch("/info")
    .then((r) => (r.ok ? r.json() : null))
    .then((info) => { if (info && info.version) $("#build").textContent = String(info.version).slice(0, 7); })
    .catch(() => {});
})();
