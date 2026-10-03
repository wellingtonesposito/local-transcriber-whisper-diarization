/* Transcriber front-end: Alpine components + live progress over SSE. No external requests. */
(function () {
  "use strict";

  async function api(method, url, body) {
    const opts = { method, headers: {} };
    if (body !== undefined) { opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(body); }
    const r = await fetch(url, opts);
    let data = null;
    try { data = await r.json(); } catch (e) { /* empty body */ }
    if (!r.ok) throw new Error((data && data.detail) || r.statusText || "Request failed");
    return data;
  }

  function toast(msg) {
    const el = document.createElement("div");
    el.className = "toast"; el.setAttribute("role", "status"); el.textContent = msg;
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 3600);
  }

  function refreshFiles() { htmx.trigger(document.body, "refresh-files"); }

  function store(key, value) { try { if (value === undefined) return localStorage.getItem(key); localStorage.setItem(key, value); } catch (e) { /* storage unavailable */ } return null; }

  // ---- live progress -------------------------------------------------------------------
  const STAGES = { prepare: "Preparing audio", transcribe: "Transcribing", diarize: "Identifying speakers", align: "Finishing" };

  function applyEvent(ev) {
    const row = document.querySelector('[data-media="' + ev.media_id + '"]');
    const page = document.querySelector("[data-project]");
    if (!row) {
      if (page && page.dataset.project === ev.project_id) refreshFiles();
      return;
    }
    if (row.dataset.status !== ev.status) { refreshFiles(); return; }
    const pct = Math.round((ev.progress || 0) * 1000) / 10;
    const bar = row.querySelector("[data-bar]");
    if (bar) bar.style.width = pct + "%";
    const pill = row.querySelector("[data-pill]");
    if (pill) pill.textContent = (STAGES[ev.stage] || "Working") + " · " + Math.round(pct) + "%";
    const msg = row.querySelector("[data-msg]");
    if (msg) msg.textContent = ev.message || "";
  }

  function connect() {
    const es = new EventSource("/events");
    es.onmessage = (e) => { try { applyEvent(JSON.parse(e.data)); } catch (err) { /* ignore */ } };
    // EventSource reconnects by itself; refresh once on reconnect to catch missed events.
    es.onopen = () => { if (document.querySelector("[data-project]")) refreshFiles(); };
  }
  document.addEventListener("DOMContentLoaded", connect);

  // ---- Alpine -----------------------------------------------------------------------------
  document.addEventListener("alpine:init", () => {
    Alpine.store("app", {
      post(url, body) { return api("POST", url, body === undefined ? {} : body).then(refreshFiles).catch((e) => toast(e.message)); },
      async transcribe(pid, ids) {
        try {
          const r = await api("POST", "/api/projects/" + pid + "/transcribe", ids ? { media_ids: ids } : {});
          if (!r.queued.length) toast("Nothing to transcribe.");
          refreshFiles();
        } catch (e) { toast(e.message); }
      },
      async setReviewed(id, value) {
        try { await api("PUT", "/api/media/" + id + "/reviewed", { reviewed: value }); refreshFiles(); }
        catch (e) { toast(e.message); }
      },
      async removeMedia(id, name, done) {
        const msg = done ? 'Remove "' + name + '" and its transcript? This cannot be undone.' : 'Remove "' + name + '"?';
        if (!confirm(msg)) return;
        try { await api("DELETE", "/api/media/" + id); refreshFiles(); } catch (e) { toast(e.message); }
      },
    });

    Alpine.store("theme", {
      mode: "Auto",
      init() { const t = store("transcriber.theme"); this.mode = t === "light" ? "Light" : t === "dark" ? "Dark" : "Auto"; },
      cycle() {
        this.mode = { Auto: "Light", Light: "Dark", Dark: "Auto" }[this.mode];
        const root = document.documentElement;
        if (this.mode === "Auto") { delete root.dataset.theme; try { localStorage.removeItem("transcriber.theme"); } catch (e) { /* ignore */ } }
        else { root.dataset.theme = this.mode.toLowerCase(); store("transcriber.theme", this.mode.toLowerCase()); }
      },
    });

    Alpine.store("exp", {
      cleanup: "verbatim", timestamps: true, precision: 1, voice: false, merge: true, formats: ["docx_table", "vtt"],
      init() {
        try {
          const d = JSON.parse(store("transcriber.exp") || "{}");
          for (const k of ["cleanup", "timestamps", "precision", "voice", "merge", "formats"]) if (d[k] !== undefined) this[k] = d[k];
          if (d.speaker_style === "voice") this.voice = true;  // older saved shape
        } catch (e) { /* ignore */ }
      },
      save() {
        store("transcriber.exp", JSON.stringify({ cleanup: this.cleanup, timestamps: this.timestamps, precision: this.precision, voice: this.voice, merge: this.merge, formats: this.formats }));
      },
      set(k, v) { this[k] = v; this.save(); },
      has(k) { return this.formats.includes(k); },
      toggle(k) { this.formats = this.has(k) ? this.formats.filter((x) => x !== k) : [...this.formats, k]; this.save(); },
      qs(extra) {
        return new URLSearchParams(Object.assign({
          cleanup: this.cleanup, timestamps: this.timestamps ? 1 : 0, precision: this.precision, speaker_style: this.voice ? "voice" : "prefix",
          merge_turns: this.merge ? 1 : 0,
        }, extra || {})).toString();
      },
    });

    // Export dialog. scope = { projectId, mediaIds: [...], count, name }
    Alpine.store("dlg", {
      open: false, scope: null,
      show(scope) { this.scope = scope; this.open = true; },
      close() { this.open = false; },
      get subtitle() {
        const s = this.scope; if (!s) return "";
        return s.count === 1 ? s.name : s.count + " transcribed files in " + s.name;
      },
      get buttonLabel() {
        const n = (this.scope ? this.scope.count : 0) * Alpine.store("exp").formats.length;
        return n > 1 ? "Download " + n + " files (.zip)" : "Download";
      },
      download() {
        const e = Alpine.store("exp"), s = this.scope;
        if (!s || !e.formats.length) return;
        if (s.count === 1 && e.formats.length === 1) {
          window.location.href = "/media/" + s.mediaIds[0] + "/export/" + e.formats[0] + "?" + e.qs();
        } else {
          window.location.href = "/projects/" + s.projectId + "/export.zip?" + e.qs({ formats: e.formats.join(","), media: s.mediaIds.join(",") });
        }
        this.close();
      },
    });

    Alpine.data("uploader", (pid) => ({
      items: [], dragging: false, auto: false, busy: false, seq: 0,
      add(fileList) {
        for (const f of Array.from(fileList)) this.items.push({ id: ++this.seq, file: f, name: f.name, pct: 0, state: "waiting", error: "" });
        this.pump();
      },
      async pump() {
        if (this.busy) return;
        this.busy = true;
        try {
          for (const it of this.items) { if (it.state === "waiting") await this.send(it); }
        } finally { this.busy = false; }
      },
      send(it) {
        return new Promise((resolve) => {
          it.state = "uploading";
          const xhr = new XMLHttpRequest();
          xhr.open("POST", "/api/projects/" + pid + "/upload?filename=" + encodeURIComponent(it.name));
          xhr.upload.onprogress = (e) => { if (e.lengthComputable) it.pct = Math.round((e.loaded / e.total) * 100); };
          xhr.onload = () => {
            let data = {};
            try { data = JSON.parse(xhr.responseText); } catch (e) { /* not json */ }
            if (xhr.status >= 200 && xhr.status < 300) {
              it.state = "done"; it.pct = 100; refreshFiles();
              if (this.auto) Alpine.store("app").transcribe(pid, [data.id]);
              setTimeout(() => { this.items = this.items.filter((x) => x.id !== it.id); }, 4000);
            } else {
              it.state = "error"; it.error = it.name + ": " + (data.detail || xhr.statusText || "upload failed");
              setTimeout(() => { this.items = this.items.filter((x) => x.id !== it.id); }, 12000);
            }
            resolve();
          };
          xhr.onerror = () => { it.state = "error"; it.error = it.name + ": network error"; resolve(); };
          xhr.send(it.file);
        });
      },
    }));

    const blank = (v) => (v === null || v === undefined || v === "" ? "" : Number(v));
    const PRESETS = {
      interview: { diarize: true, num_speakers: 2, min_speakers: "", max_speakers: "" },
      focus: { diarize: true, num_speakers: "", min_speakers: 3, max_speakers: 8 },
      draft: { diarize: false, num_speakers: "", min_speakers: "", max_speakers: "" },
    };

    Alpine.data("settingsForm", (pid, initial, hasToken) => ({
      s: Object.assign({}, initial, {
        language: initial.language || "", num_speakers: blank(initial.num_speakers),
        min_speakers: blank(initial.min_speakers), max_speakers: blank(initial.max_speakers),
      }),
      hasToken, status: "", timer: null,
      get active() {
        for (const [name, p] of Object.entries(PRESETS)) {
          if (this.s.diarize === p.diarize && blank(this.s.num_speakers) === p.num_speakers
              && blank(this.s.min_speakers) === p.min_speakers && blank(this.s.max_speakers) === p.max_speakers) return name;
        }
        return "";
      },
      applyPreset(name) { Object.assign(this.s, PRESETS[name]); this.save(); },
      save() {
        clearTimeout(this.timer);
        this.status = "Saving…";
        this.timer = setTimeout(async () => {
          try {
            const body = Object.assign({}, this.s);
            for (const k of ["num_speakers", "min_speakers", "max_speakers"]) if (body[k] === "") body[k] = null;
            await api("PUT", "/api/projects/" + pid + "/settings", body);
            this.status = "Saved";
          } catch (e) { this.status = "Could not save: " + e.message; }
        }, 300);
      },
    }));

    Alpine.data("tokenForm", (has) => ({
      has, token: "", msg: "", ok: false, busy: false,
      async save() {
        this.busy = true;
        try { await api("POST", "/api/setup/token", { token: this.token }); this.has = true; this.token = ""; this.ok = true; this.msg = "Token saved. Click “Test access” to verify it."; }
        catch (e) { this.ok = false; this.msg = e.message; }
        this.busy = false;
      },
      async test() {
        this.busy = true; this.msg = "Checking with Hugging Face…"; this.ok = true;
        try { const r = await api("POST", "/api/setup/test-token", {}); this.ok = r.ok; this.msg = r.message; }
        catch (e) { this.ok = false; this.msg = e.message; }
        this.busy = false;
      },
      async remove() {
        await api("POST", "/api/setup/token", { token: "" }); this.has = false; this.ok = true; this.msg = "Token removed.";
      },
    }));

    Alpine.data("review", (mid, reviewedInitial) => {
      // Kept outside Alpine's reactive state: thousands of word timings would make every update expensive.
      const words = new Map();    // segment id -> [[word, start, end], ...] as transcribed
      const timing = new Map();   // segment id -> { text, t: [[start, end], ...] } per displayed token
      const segTurn = new Map();  // segment id -> turn id
      let curEl = null, raf = 0, lastUserScroll = 0;

      const esc = (x) => x.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
      const tokens = (seg) => seg.text.split(/\s+/).filter(Boolean);

      // Start/end time for each displayed word. Exact when the text still matches the transcribed words;
      // after an edit or in the Clean view, the words are spread across the sentence by length.
      function tokenTimes(seg) {
        const hit = timing.get(seg.id);
        if (hit && hit.text === seg.text) return hit.t;
        const toks = tokens(seg), w = words.get(seg.id) || [];
        let t;
        if (w.length && w.length === toks.length) {
          t = w.map((x) => [x[1], x[2]]);
        } else {
          const s0 = w.length ? w[0][1] : seg.start, e0 = w.length ? w[w.length - 1][2] : seg.end;
          const weights = toks.map((x) => x.length + 1), total = weights.reduce((a, b) => a + b, 0) || 1;
          let acc = 0;
          t = weights.map((x) => { const a = s0 + ((e0 - s0) * acc) / total; acc += x; return [a, s0 + ((e0 - s0) * acc) / total]; });
        }
        timing.set(seg.id, { text: seg.text, t });
        return t;
      }

      return {
        rows: [], turns: [], speakers: [], info: {}, blocks: [],
        active: -1, activeTurn: -1, editing: null, loading: true, reviewed: !!reviewedInitial,
        playing: false, cur: 0, total: 0, rate: 1, rates: [0.75, 1, 1.25, 1.5], minRate: 0.25, maxRate: 3,
        volume: Number.isFinite(parseFloat(store("transcriber.volume"))) ? parseFloat(store("transcriber.volume")) : 1,
        muted: store("transcriber.muted") === "1",
        theater: store("transcriber.theater") === "1",

        get editable() { return Alpine.store("exp").cleanup === "verbatim"; },
        get pct() { return this.total ? Math.min(100, (this.cur / this.total) * 100) : 0; },

        init() {
          this.$watch("$store.exp.cleanup", () => { this.editing = null; this.load(); });
          // Don't fight the reader: pause auto-follow for a few seconds after they scroll themselves.
          const stamp = () => { lastUserScroll = Date.now(); };
          window.addEventListener("wheel", stamp, { passive: true });
          window.addEventListener("touchmove", stamp, { passive: true });
        },

        async load() {
          try {
            const d = await api("GET", "/api/media/" + mid + "/transcript?" + Alpine.store("exp").qs());
            words.clear(); timing.clear();
            for (const r of d.rows) { words.set(r.id, r.words); delete r.words; }
            this.rows = d.rows; this.speakers = d.speakers; this.info = d.media;
            if (!this.total) this.total = d.media.duration || 0;
            this.buildTurns(); this.buildBlocks();
          } catch (e) { toast(e.message); }
          this.loading = false;
        },
        // One block per speaker: consecutive segments from the same speaker are shown together.
        buildTurns() {
          const turns = []; segTurn.clear();
          for (const r of this.rows) {
            const last = turns[turns.length - 1];
            if (last && last.speaker === r.speaker) last.segs.push(r);
            else turns.push({ id: r.id, speaker: r.speaker, segs: [r] });
            segTurn.set(r.id, turns[turns.length - 1].id);
          }
          this.turns = turns;
        },
        onMeta() {
          const d = this.$refs.player.duration;
          if (isFinite(d) && d > 0) { this.total = d; this.buildBlocks(); }
          this.applyVolume();
        },

        fmt(t) {
          t = Math.max(0, Math.floor(t || 0));
          const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = t % 60;
          const mm = String(m).padStart(h ? 2 : 1, "0"), ss = String(s).padStart(2, "0");
          return h ? h + ":" + mm + ":" + ss : mm + ":" + ss;
        },
        speaker(id) { return this.speakers.find((x) => x.id === id); },
        name(id) { const s = this.speaker(id); return (s && s.name) || id; },
        initial(id) { const s = this.speaker(id), i = this.speakers.findIndex((x) => x.id === id); return s && s.name ? s.name.charAt(0).toUpperCase() : String(i + 1); },
        color(id) { const i = this.speakers.findIndex((x) => x.id === id); return "var(--spk" + ((Math.max(i, 0) % 5) + 1) + ")"; },
        share(id) {
          let mine = 0, all = 0;
          for (const r of this.rows) { if (r.deleted) continue; const d = r.end - r.start; all += d; if (r.speaker === id) mine += d; }
          return all ? Math.round((mine / all) * 100) : 0;
        },
        buildBlocks() {
          const last = this.rows.length ? this.rows[this.rows.length - 1].end : 0;
          const T = Math.max(this.total, last);
          if (!T) { this.blocks = []; return; }
          let c = 0; const b = [];
          for (const r of this.rows) {
            if (r.deleted) continue;
            if (r.start > c + 0.01) b.push({ flex: r.start - c, spk: null, color: "transparent" });
            b.push({ flex: Math.max(0.05, r.end - Math.max(r.start, c)), spk: r.speaker, color: this.color(r.speaker) });
            c = Math.max(c, r.end);
          }
          if (T > c) b.push({ flex: T - c, spk: null, color: "transparent" });
          this.blocks = b;
        },

        // ---- words ----
        segHtml(seg) {
          return tokens(seg).map((t, i) => '<span class="w" id="w-' + seg.id + "-" + i + '" data-i="' + i + '">' + esc(t) + "</span>").join(" ");
        },
        wordClick(e, seg) {
          const el = e.target.closest(".w");
          if (!el) return;
          const t = tokenTimes(seg)[Number(el.dataset.i)];
          if (t) { const p = this.$refs.player; p.currentTime = t[0]; p.play().catch(() => {}); }
        },

        // ---- playback ----
        toggle() { const p = this.$refs.player; if (p.paused) p.play().catch(() => {}); else p.pause(); },
        onPlay() { this.playing = true; cancelAnimationFrame(raf); const loop = () => { this.sync(); if (this.playing) raf = requestAnimationFrame(loop); }; loop(); },
        onPause() { this.playing = false; cancelAnimationFrame(raf); this.sync(); },
        seek(r) { const p = this.$refs.player; p.currentTime = r.start; p.play().catch(() => {}); },
        scrub(e) { const rc = e.currentTarget.getBoundingClientRect(); this.$refs.player.currentTime = Math.max(0, Math.min(1, (e.clientX - rc.left) / rc.width)) * this.total; },
        nudge(d) { const p = this.$refs.player; p.currentTime = Math.max(0, Math.min(this.total, p.currentTime + d)); },
        get rateLabel() { return Number(this.rate.toFixed(2)) + "×"; },
        setRate(r) { this.rate = Math.min(this.maxRate, Math.max(this.minRate, Math.round(r * 100) / 100)); this.$refs.player.playbackRate = this.rate; },
        stepRate(d) { this.setRate(this.rate + d); },

        // ---- volume ----
        applyVolume() {
          const p = this.$refs.player;
          if (p) { p.volume = Math.max(0, Math.min(1, this.volume)); p.muted = this.muted; }
          store("transcriber.volume", String(this.volume)); store("transcriber.muted", this.muted ? "1" : "0");
        },
        setVolume(v) { this.volume = v; if (v > 0) this.muted = false; this.applyVolume(); },
        toggleMute() { this.muted = !this.muted; this.applyVolume(); },

        // ---- video: small on the side (default) or "theater" (full width, transcript below) ----
        toggleTheater() {
          this.theater = !this.theater;
          store("transcriber.theater", this.theater ? "1" : "0");
          // The layout is pure CSS, so the video keeps playing. Re-centre the current word once it has reflowed.
          this.$nextTick(() => { if (curEl && curEl.isConnected) this.follow(curEl); });
        },
        fullscreen() { const v = this.$refs.player; const f = v.requestFullscreen || v.webkitRequestFullscreen; if (f) f.call(v); },
        // Keep the playing word in the visible part of the page (below the pinned video in theater mode).
        follow(el) {
          // In theater mode the video is pinned to the top of the window, so measure from there even if it has
          // not been scrolled into its pinned position yet.
          const media = this.theater ? document.querySelector(".media") : null;
          const top = Math.max(90, media ? media.getBoundingClientRect().height + 16 : 0);
          const bottom = window.innerHeight - 150;
          const r = el.getBoundingClientRect();
          if (r.top < top || r.bottom > bottom) window.scrollBy({ top: r.top - (top + (bottom - top) * 0.4), behavior: "smooth" });
        },

        async toggleReviewed() {
          try { const r = await api("PUT", "/api/media/" + mid + "/reviewed", { reviewed: !this.reviewed }); this.reviewed = r.reviewed; }
          catch (e) { toast(e.message); }
        },
        onTime() { this.sync(); },

        // Called every animation frame while playing (and on seeks): playhead, active turn, current word.
        sync() {
          const p = this.$refs.player, t = p.currentTime;
          if (Math.abs(t - this.cur) > 0.03) this.cur = t;
          let lo = 0, hi = this.rows.length - 1, found = -1;
          while (lo <= hi) { const mid = (lo + hi) >> 1; if (this.rows[mid].start <= t + 0.05) { found = mid; lo = mid + 1; } else hi = mid - 1; }
          const row = found >= 0 ? this.rows[found] : null;
          const id = row ? row.id : -1;
          if (id !== this.active) { this.active = id; this.activeTurn = segTurn.has(id) ? segTurn.get(id) : -1; }
          this.highlight(row, t);
        },
        highlight(row, t) {
          let el = null;
          if (row && !row.deleted && t <= row.end + 0.6) {
            const tm = tokenTimes(row);
            let lo = 0, hi = tm.length - 1, j = -1;
            while (lo <= hi) { const mid = (lo + hi) >> 1; if (tm[mid][0] <= t + 0.03) { j = mid; lo = mid + 1; } else hi = mid - 1; }
            if (j >= 0) el = document.getElementById("w-" + row.id + "-" + j);
          }
          if (el === curEl) return;
          if (curEl) curEl.classList.remove("now");
          curEl = el;
          if (el) {
            el.classList.add("now");
            if (this.playing && Date.now() - lastUserScroll > 4000) this.follow(el);
          }
        },

        // ---- editing ----
        startEdit(turn, seg) {
          if (!this.editable) return;
          this.$refs.player.pause();
          this.editing = turn.id;
          const target = seg || turn.segs[0];
          this.$nextTick(() => { const el = document.getElementById("edit-" + target.id); if (el) el.focus(); });
        },
        stopEdit() { this.editing = null; },
        async rename(s, value) {
          try { await api("PUT", "/api/media/" + mid + "/speakers", { [s.id]: value }); s.name = value.trim(); }
          catch (e) { toast(e.message); }
        },
        async commit(r, el) {
          if (!this.editable || r.deleted) return;
          const text = el.textContent.trim();
          if (text === r.text) return;
          if (!text) { el.textContent = r.text; toast("A segment can't be empty. Use Remove instead."); return; }
          try { const res = await api("PUT", "/api/media/" + mid + "/segments/" + r.id, { text }); r.text = text; r.raw = text; r.edited = res.edited; }
          catch (e) { el.textContent = r.text; toast(e.message); }
        },
        async toggleDelete(r) {
          try { await api("PUT", "/api/media/" + mid + "/segments/" + r.id, { deleted: !r.deleted }); r.deleted = !r.deleted; this.buildBlocks(); }
          catch (e) { toast(e.message); }
        },
      };
    });
  });
})();
