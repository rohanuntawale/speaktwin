/**
 * ============================================================
 *  SpeakTwin — Voiceprint
 * ============================================================
 *  The signature element: one circular instrument that is
 *  simultaneously the control, the live meter, and the score.
 *
 *  Two clocks drive it, deliberately:
 *
 *    60 fps  — radial bars read the browser's own AnalyserNode,
 *              so the instrument responds to your voice the
 *              instant you make a sound.
 *    2.5 s   — the outer arc and colour temperature come from
 *              the server's analysis of the last chunk.
 *
 *  Without the fast clock the interface only moves when the
 *  server answers, which reads as broken. Without the slow one
 *  it is a toy visualiser with nothing to say.
 */

(function (global) {
  "use strict";

  const TAU = Math.PI * 2;
  const BARS = 96;              // radial spokes
  const SMOOTH = 0.28;          // bar easing per frame
  const IDLE_BREATH = 0.055;    // resting amplitude

  /** Cold at rest, tungsten while speaking. */
  const ICE = [127, 212, 255];
  const EMBER = [255, 143, 77];
  const ROSE = [255, 95, 126];

  const lerp = (a, b, t) => a + (b - a) * t;
  const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

  function mix(a, b, t) {
    return [
      Math.round(lerp(a[0], b[0], t)),
      Math.round(lerp(a[1], b[1], t)),
      Math.round(lerp(a[2], b[2], t)),
    ];
  }

  const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

  class Voiceprint {
    constructor(canvas) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");

      this.analyser = null;
      this.freq = null;

      this.bars = new Float32Array(BARS);
      this.level = 0;          // smoothed live loudness, 0-1
      this.warmth = 0;         // 0 = ice, 1 = ember
      this.targetWarmth = 0;

      this.score = 0;          // drawn arc, 0-100
      this.targetScore = 0;
      this.listening = false;
      this.phase = 0;
      this.intro = 0;          // 0-1 entrance sweep
      this.startedAt = 0;      // set on the first drawn frame

      this.reduced = global.matchMedia
        ? global.matchMedia("(prefers-reduced-motion: reduce)").matches
        : false;

      this._resize = this._resize.bind(this);
      this._frame = this._frame.bind(this);

      this.size = 0;
      this._resize();

      // Measuring once is unreliable: the box is often still 0 while web
      // fonts load and the aspect-ratio parent settles. Observe instead.
      if (global.ResizeObserver) {
        new ResizeObserver(this._resize).observe(canvas);
      } else {
        global.addEventListener("resize", this._resize);
      }

      requestAnimationFrame(this._frame);
    }

    _resize() {
      const box = this.canvas.getBoundingClientRect();
      const size = Math.round(box.width);
      if (size < 2 || size === this.size) return;

      const dpr = Math.min(global.devicePixelRatio || 1, 2);
      this.canvas.width = size * dpr;
      this.canvas.height = size * dpr;
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.size = size;
    }

    /** Attach the live audio graph so bars can run at frame rate. */
    listen(analyser) {
      this.analyser = analyser;
      this.freq = new Uint8Array(analyser.frequencyBinCount);
      this.listening = true;
    }

    stop() {
      this.listening = false;
      this.analyser = null;
      this.targetWarmth = 0;
    }

    /** Server truth for the last chunk. */
    update({ score, energyDb, tooLoud }) {
      if (typeof score === "number") this.targetScore = clamp01(score / 100) * 100;

      if (typeof energyDb === "number") {
        // -45 dBFS (silence) → 0, -12 dBFS (loud) → 1
        this.targetWarmth = clamp01((energyDb + 45) / 33);
      }
      this.tooLoud = !!tooLoud;
    }

    reset() {
      this.targetScore = 0;
      this.score = 0;
      this.targetWarmth = 0;
      this.tooLoud = false;
    }

    /** Current instrument colour — CSS reads this to stay in step. */
    colour() {
      const base = mix(ICE, EMBER, this.warmth);
      return this.tooLoud ? mix(base, ROSE, 0.55) : base;
    }

    _sampleAudio() {
      if (!this.analyser || !this.freq) return null;
      this.analyser.getByteFrequencyData(this.freq);

      // Speech energy sits low in the spectrum; sampling the whole
      // range would flatten the response into noise.
      const usable = Math.floor(this.freq.length * 0.55);
      const out = new Float32Array(BARS);
      let sum = 0;

      for (let i = 0; i < BARS; i++) {
        // Logarithmic bin mapping so low frequencies get the detail
        const t = i / BARS;
        const from = Math.floor(Math.pow(t, 1.7) * usable);
        const to = Math.max(from + 1, Math.floor(Math.pow((i + 1) / BARS, 1.7) * usable));
        let peak = 0;
        for (let b = from; b < to && b < usable; b++) {
          if (this.freq[b] > peak) peak = this.freq[b];
        }
        const v = peak / 255;
        out[i] = v;
        sum += v;
      }

      return { bars: out, level: sum / BARS };
    }

    _frame(now) {
      const ctx = this.ctx;
      const size = this.size;

      // Nothing measurable yet — wait for layout rather than drawing at 0.
      if (!size) {
        requestAnimationFrame(this._frame);
        return;
      }

      const cx = size / 2;
      const cy = size / 2;

      ctx.clearRect(0, 0, size, size);

      // Drive the entrance and the idle breath from elapsed time, not
      // frame count. Counting frames ties the animation's speed to the
      // device's frame rate, so it crawls wherever frames are scarce.
      if (!this.startedAt) this.startedAt = now;
      const elapsed = (now - this.startedAt) / 1000;

      this.intro = this.reduced ? 1 : Math.min(1, elapsed / 0.75);
      this.phase = this.reduced ? 0 : elapsed * 0.4;

      // ── Advance state ──────────────────────────────────────
      const sample = this.listening ? this._sampleAudio() : null;

      for (let i = 0; i < BARS; i++) {
        let target;
        if (sample) {
          target = sample.bars[i];
        } else {
          // At rest the ring breathes rather than sitting dead flat.
          target = IDLE_BREATH * (1 + Math.sin(this.phase * 2 + i * 0.22) * 0.5);
        }
        this.bars[i] += (target - this.bars[i]) * SMOOTH;
      }

      const liveLevel = sample ? sample.level : 0;
      this.level += (liveLevel - this.level) * 0.14;

      // Warmth follows the live signal between server updates, so the
      // room reacts immediately rather than in 2.5 s steps.
      const wTarget = this.listening
        ? Math.max(this.targetWarmth, clamp01(this.level * 2.6))
        : 0;
      this.warmth += (wTarget - this.warmth) * 0.05;
      this.score += (this.targetScore - this.score) * 0.07;

      const colour = this.colour();
      const radius = size * 0.325;
      const reach = size * 0.125;
      const ease = this.intro * this.intro * (3 - 2 * this.intro);

      // ── Halo ───────────────────────────────────────────────
      const glow = ctx.createRadialGradient(cx, cy, radius * 0.35, cx, cy, radius * 1.7);
      glow.addColorStop(0, rgba(colour, 0.16 * (0.35 + this.level * 1.6) * ease));
      glow.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = glow;
      ctx.fillRect(0, 0, size, size);

      // ── Guide ring ─────────────────────────────────────────
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, TAU * ease);
      ctx.strokeStyle = "rgba(255,255,255,0.055)";
      ctx.lineWidth = 1;
      ctx.stroke();

      // ── Radial bars: the voice itself ──────────────────────
      const visible = Math.floor(BARS * ease);
      ctx.lineCap = "round";

      for (let i = 0; i < visible; i++) {
        // Start at 12 o'clock so growth reads as symmetrical
        const angle = (i / BARS) * TAU - Math.PI / 2;
        const amp = this.bars[i];
        const len = 2 + amp * reach;

        const x1 = cx + Math.cos(angle) * radius;
        const y1 = cy + Math.sin(angle) * radius;
        const x2 = cx + Math.cos(angle) * (radius + len);
        const y2 = cy + Math.sin(angle) * (radius + len);

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = rgba(colour, 0.3 + amp * 0.7);
        ctx.lineWidth = 2;
        ctx.stroke();

        // Mirror inward at low opacity — reads as a reflection,
        // which is the whole premise of the product.
        if (amp > 0.05) {
          const x3 = cx + Math.cos(angle) * (radius - len * 0.5);
          const y3 = cy + Math.sin(angle) * (radius - len * 0.5);
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x3, y3);
          ctx.strokeStyle = rgba(colour, amp * 0.16);
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
      }

      // ── Confidence arc ─────────────────────────────────────
      if (this.score > 0.5) {
        const arcR = radius + reach + size * 0.045;
        const sweep = (this.score / 100) * TAU * ease;

        ctx.beginPath();
        ctx.arc(cx, cy, arcR, 0, TAU);
        ctx.strokeStyle = "rgba(255,255,255,0.045)";
        ctx.lineWidth = 3;
        ctx.stroke();

        const grad = ctx.createLinearGradient(cx - arcR, cy - arcR, cx + arcR, cy + arcR);
        grad.addColorStop(0, rgba(mix(colour, ICE, 0.4), 0.9));
        grad.addColorStop(1, rgba(colour, 1));

        ctx.beginPath();
        ctx.arc(cx, cy, arcR, -Math.PI / 2, -Math.PI / 2 + sweep);
        ctx.strokeStyle = grad;
        ctx.lineWidth = 3;
        ctx.lineCap = "round";
        ctx.shadowColor = rgba(colour, 0.55);
        ctx.shadowBlur = 12;
        ctx.stroke();
        ctx.shadowBlur = 0;

        // Head of the arc
        const hx = cx + Math.cos(-Math.PI / 2 + sweep) * arcR;
        const hy = cy + Math.sin(-Math.PI / 2 + sweep) * arcR;
        ctx.beginPath();
        ctx.arc(hx, hy, 3, 0, TAU);
        ctx.fillStyle = rgba(colour, 1);
        ctx.fill();
      }

      requestAnimationFrame(this._frame);
    }
  }

  global.Voiceprint = Voiceprint;
})(window);
