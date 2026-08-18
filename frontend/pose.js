/**
 * ============================================================
 *  SpeakTwin — Posture & gesture capture
 * ============================================================
 *  MediaPipe Pose runs here, in the browser. Video frames never
 *  leave the machine; only 33 landmark points per frame are sent,
 *  which makes a 2.5 s batch about 25 KB instead of megabytes.
 *
 *  Detection runs at the camera's frame rate for a smooth skeleton
 *  overlay, but landmarks are only *sampled* for upload at
 *  SAMPLE_FPS. Gesture rate is counted from runs of movement
 *  between sampled frames, so that rate has to stay fixed or the
 *  measurement drifts with the hardware.
 */

const MP_VERSION = "0.10.14";

// Model tier. `full` is the default: `lite`'s depth (z) estimates are too
// noisy to judge forward-head carriage — the single most common bad webcam
// posture — which made the live cue blind to it. Override with
// window.SPEAKTWIN_POSE_MODEL = "lite" | "full" | "heavy".
const POSE_MODELS = {
  lite:  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
  full:  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
  heavy: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
};

const SAMPLE_FPS = 10;        // landmarks kept per second for upload
const BATCH_MS = 2500;        // matches the speech chunk cadence

// Skeleton connections worth drawing — torso and arms. Legs are usually
// out of frame for a seated speaker and add noise to the overlay.
const BONES = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24],
];
const JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 0];

// ── Live guidance ────────────────────────────────────────────
// A cut-down copy of the server's geometry, run every frame so the
// speaker sees a correction while they can still act on it. The server
// batch every 2.5 s remains the authoritative score; this is the nudge.
// Thresholds mirror backend/utils/helpers.py — keep them in step.
const T = {
  shoulderTilt: [5, 10],
  headTilt: [8, 15],
  torsoLean: [7, 14],
  openness: [0.62, 0.85],   // [closed, open] — note: higher is better
  // Head grown past your own neutral size — craning at the screen brings
  // only the head nearer the camera. Matches HEAD_SCALE_* in
  // backend/utils/helpers.py. Judged vs a self-calibrated baseline, never
  // an absolute: the old z-depth check nagged correctly-seated people
  // because webcam perspective biases everyone's ears toward the camera.
  headScale: [1.10, 1.20],
  // Anatomical ceiling on the baseline: ear span is at most ~half the
  // shoulder width at true neutral. A larger "neutral" means the camera
  // first saw you already leaning in — without this clamp that craned
  // pose gets enshrined as the baseline and is never flagged.
  headScaleCeiling: 0.52,
  // Hand landmark within this many shoulder-widths of the mouth counts
  // as covering the face (fingertips on the mouth ≈ 0-0.15; gesturing
  // at chest height ≈ 0.8+).
  handFace: 0.30,
  // Sustained drop of the head below the speaker's own session-best
  // height, as a fraction. Self-calibrated, so it survives different
  // bodies, chairs, and camera angles.
  headDrop: [0.88, 0.80],   // [watch below, fix below] × baseline
};

// How long a slump must persist before it is called (frames), and how
// long everything must stay clear before "Good posture" is earned.
const DROP_HOLD_FRAMES = 24;
const GOOD_HOLD_FRAMES = 30;

const TONE = {
  good:  "rgba(74, 222, 159, 0.95)",
  watch: "rgba(255, 194, 71, 0.95)",
  fix:   "rgba(255, 95, 126, 1.0)",
  idle:  "rgba(127, 212, 255, 0.85)",
};

// Glow colours matching TONE but more translucent for the shadow pass
const GLOW = {
  good:  "rgba(74, 222, 159, 0.35)",
  watch: "rgba(255, 194, 71, 0.35)",
  fix:   "rgba(255, 95, 126, 0.45)",
  idle:  "rgba(127, 212, 255, 0.28)",
};

const deg = (rad) => (rad * 180) / Math.PI;

/** How far a→b departs from level, in degrees. */
function tiltOf(a, b) {
  const dx = Math.abs(b[0] - a[0]);
  const dy = Math.abs(b[1] - a[1]);
  return dx < 1e-6 ? 90 : deg(Math.atan(dy / dx));
}

/** How far lower→upper departs from vertical, in degrees. */
function leanOf(lower, upper) {
  const dx = Math.abs(upper[0] - lower[0]);
  const dy = Math.abs(upper[1] - lower[1]);
  return dy < 1e-6 ? 90 : deg(Math.atan(dx / dy));
}

function band(value, [watch, fix]) {
  const m = Math.abs(value);
  if (m >= fix) return "fix";
  if (m >= watch) return "watch";
  return "good";
}

class PoseTracker {
  constructor({ video, canvas, onBatch, onStatus, onGuidance, onFps, onCalibration, onResolution }) {
    this.video = video;
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.onBatch = onBatch || (() => {});
    this.onStatus = onStatus || (() => {});
    this.onGuidance = onGuidance || (() => {});
    this.onFps = onFps || (() => {});
    this.onCalibration = onCalibration || (() => {});
    this.onResolution = onResolution || (() => {});

    this.cue = null;        // currently displayed guidance
    this.cueSince = 0;      // when it was raised
    this.tone = "idle";

    // Self-calibration: this speaker's own geometry is the reference.
    this.baseHead = 0;      // slump: best nose-above-shoulders ratio seen
    this.lowFrames = 0;     // consecutive frames below the slump line
    this.baseScale = Infinity; // chin: smallest head-size ratio seen
    this.fwdFrames = 0;     // consecutive frames past the chin line
    this.faceFrames = 0;    // consecutive frames with a hand at the face
    this.baseAge = 0;       // frames of baseline evidence collected
    this.clearFrames = 0;   // consecutive frames with nothing wrong

    this.landmarker = null;
    this.stream = null;
    this.running = false;
    this.buffer = [];
    this.lastSample = 0;
    this.batchStart = 0;
    this.lastVideoTime = -1;
    this._loop = this._loop.bind(this);

    // ── Video adjustment state ──────────────────────────────
    // Digital zoom: 1.0 = no zoom, 2.0 = 2× zoom.
    this.zoomLevel = 1.0;
    // Flip: true = natural view (not mirrored), false = mirror (default).
    this.flipped = false;
    // Framing guide overlay
    this.showGrid = false;

    // ── FPS tracking ────────────────────────────────────────
    this._fpsFrames = 0;
    this._fpsLast = 0;
    this._fpsValue = 0;
  }

  get active() {
    return this.running;
  }




  async start(deviceId) {
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      const error = new Error("Camera access requires localhost or HTTPS.");
      error.name = "InsecureContextError";
      throw error;
    }

    // Ask for the webcam before loading the remote pose model. This keeps
    // camera permission failures separate from CDN/model failures and lets
    // the user see the live preview as soon as the device is available.
    this.onStatus("Requesting camera");
    const videoConstraints = {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      facingMode: "user",
    };
    if (deviceId) {
      videoConstraints.deviceId = { exact: deviceId };
    }

    this.stream = await navigator.mediaDevices.getUserMedia({
      video: videoConstraints,
      audio: false,
    });

    try {
      this.video.srcObject = this.stream;
      await this.video.play();

      this.canvas.width = this.video.videoWidth || 640;
      this.canvas.height = this.video.videoHeight || 480;

      const resLabel = `${this.canvas.height}p`;
      this.onResolution(resLabel);
    } catch (error) {
      this.stop();
      throw error;
    }

    this.onStatus("Loading pose model");

    if (!this.landmarker) {
      // Loaded from CDN as an ES module — no build step, and the weights
      // are cached by the browser after the first run.
      const vision = await import(
        `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`
      );
      const files = await vision.FilesetResolver.forVisionTasks(
        `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`
      );

      const wanted = window.SPEAKTWIN_POSE_MODEL || "full";
      // Fall back down the tiers rather than failing the camera outright.
      const tiers = [...new Set([wanted, "full", "lite"])].filter((t) => POSE_MODELS[t]);

      let lastErr = null;
      for (const tier of tiers) {
        try {
          this.landmarker = await vision.PoseLandmarker.createFromOptions(files, {
            baseOptions: { modelAssetPath: POSE_MODELS[tier], delegate: "GPU" },
            runningMode: "VIDEO",
            numPoses: 1,
            minPoseDetectionConfidence: 0.5,
            minPosePresenceConfidence: 0.5,
            minTrackingConfidence: 0.5,
          });
          this.modelTier = tier;
          break;
        } catch (err) {
          console.warn(`[pose] ${tier} model failed to load`, err);
          lastErr = err;
        }
      }
      if (!this.landmarker) {
        this.stop();
        const error = new Error("The posture model could not be loaded.");
        error.name = "PoseModelError";
        error.cause = lastErr;
        throw error;
      }
    }

    // New session, new body position — recalibrate every baseline.
    this.baseHead = 0;
    this.lowFrames = 0;
    this.baseScale = Infinity;
    this.fwdFrames = 0;
    this.faceFrames = 0;
    this.baseAge = 0;
    this.clearFrames = 0;
    this._fpsFrames = 0;
    this._fpsLast = 0;
    this._fpsValue = 0;

    this.running = true;
    this.buffer = [];
    this.batchStart = performance.now();
    this.onStatus("Watching");
    this.onCalibration({ active: true, progress: 0 });
    requestAnimationFrame(this._loop);
  }

  stop() {
    this.running = false;
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.video.srcObject = null;
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.buffer = [];
    this._fpsValue = 0;
    this.onStatus("Off");
    this.onFps(0);
  }

  _loop(now) {
    if (!this.running) return;

    if (this.video.readyState >= 2 && this.video.currentTime !== this.lastVideoTime) {
      this.lastVideoTime = this.video.currentTime;
      let result = null;
      try {
        result = this.landmarker.detectForVideo(this.video, now);
      } catch (err) {
        console.warn("[pose]", err);
      }

      const landmarks = result && result.landmarks && result.landmarks[0];
      this._guide(landmarks, now);
      this._draw(landmarks);

      // FPS calculation (rolling 1-second window)
      this._fpsFrames++;
      if (now - this._fpsLast >= 1000) {
        this._fpsValue = Math.round(this._fpsFrames * 1000 / (now - this._fpsLast));
        this._fpsFrames = 0;
        this._fpsLast = now;
        this.onFps(this._fpsValue);
      }

      // Sample at a fixed rate regardless of how fast detection runs.
      if (landmarks && now - this.lastSample >= 1000 / SAMPLE_FPS) {
        this.lastSample = now;
        this.buffer.push({
          t: Number(((now - this.batchStart) / 1000).toFixed(3)),
          landmarks: landmarks.map((p) => ({
            // Three decimals is ~0.5 px at 640 wide — far finer than the
            // model's own precision, and it halves the payload.
            x: Number(p.x.toFixed(3)),
            y: Number(p.y.toFixed(3)),
            z: Number((p.z ?? 0).toFixed(3)),
            visibility: Number((p.visibility ?? 1).toFixed(2)),
          })),
        });
      }

      if (now - this.batchStart >= BATCH_MS) this._flush(now);
    }

    requestAnimationFrame(this._loop);
  }

  _flush(now) {
    const frames = this.buffer;
    const duration = (now - this.batchStart) / 1000;
    this.buffer = [];
    this.batchStart = now;

    if (frames.length < 3) return;   // too little to say anything
    this.onBatch({
      duration: Number(duration.toFixed(2)),
      aspect: this.canvas.width / this.canvas.height,
      frames,
    });
  }

  /**
   * Per-frame posture check driving the live cue and overlay colour.
   *
   * Aspect correction matters as much here as on the server: MediaPipe
   * normalises x and y independently, so without multiplying x by the
   * frame's aspect ratio, level shoulders in a 16:9 view measure as
   * tilted and the guidance would nag at a correctly-seated speaker.
   */
  _guide(landmarks, now) {
    if (!landmarks) {
      this._raise(null, "idle", now);
      return;
    }

    const aspect = this.canvas.width / this.canvas.height || 1;
    const at = (i) => {
      const p = landmarks[i];
      if (!p || (p.visibility ?? 1) < 0.5) return null;
      return [p.x * aspect, p.y];
    };

    const ls = at(11), rs = at(12);
    if (!ls || !rs) {
      this._raise("Step into frame", "watch", now);
      return;
    }

    const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    const shoulderMid = mid(ls, rs);
    const width = Math.hypot(rs[0] - ls[0], rs[1] - ls[1]) || 1e-6;

    // Worst problem wins — one instruction at a time is actionable,
    // a list of four is not.
    const checks = [];

    const shoulderTilt = tiltOf(ls, rs);
    checks.push({
      band: band(shoulderTilt, T.shoulderTilt),
      text: "Level your shoulders",
      severity: shoulderTilt / T.shoulderTilt[1],
    });

    const le = at(7), re = at(8);
    if (le && re) {
      const headTilt = tiltOf(le, re);
      checks.push({
        band: band(headTilt, T.headTilt),
        text: "Straighten your head",
        severity: headTilt / T.headTilt[1],
      });
    }

    const lh = at(23), rh = at(24);
    if (lh && rh) {
      const hipMid = mid(lh, rh);
      const lean = leanOf(hipMid, shoulderMid);
      checks.push({
        band: band(lean, T.torsoLean),
        text: "Centre your weight",
        severity: lean / T.torsoLean[1],
      });

      const torso = Math.hypot(shoulderMid[0] - hipMid[0], shoulderMid[1] - hipMid[1]);
      if (torso > 1e-4) {
        const openness = width / torso;
        const openBand =
          openness >= T.openness[1] ? "good"
          : openness >= T.openness[0] ? "watch" : "fix";
        checks.push({
          band: openBand,
          text: "Sit tall — open your chest",
          severity: openBand === "fix" ? 1.4 : openBand === "watch" ? 1.0 : 0,
        });
      }
    }

    // ── Forward head: the classic webcam slouch ──────────────────────
    let chinMeasured = false;
    if (le && re) {
      const headScale = Math.hypot(re[0] - le[0], re[1] - le[1]) / width;
      if (headScale > 0) {
        chinMeasured = true;
        this.baseScale = Math.min(
          this.baseScale * 1.00005, headScale, T.headScaleCeiling
        );
        this.baseAge++;

        const dev = headScale / this.baseScale;
        if (dev >= T.headScale[0]) this.fwdFrames++;
        else this.fwdFrames = 0;

        if (this.fwdFrames >= DROP_HOLD_FRAMES) {
          checks.push({
            band: dev >= T.headScale[1] ? "fix" : "watch",
            text: "Draw your chin back",
            severity: dev >= T.headScale[1] ? 1.4 : 1.05,
          });
        }
      }
    }

    // ── Hand over the mouth ──────────────────────────────────────────
    const mouthL = at(9), mouthR = at(10);
    if (mouthL && mouthR) {
      const mouth = mid(mouthL, mouthR);
      let nearest = Infinity;
      for (const j of [15, 16, 17, 18, 19, 20, 21, 22]) {
        const p = at(j);
        if (p) {
          const d = Math.hypot(p[0] - mouth[0], p[1] - mouth[1]) / width;
          if (d < nearest) nearest = d;
        }
      }
      if (nearest < T.handFace) this.faceFrames++;
      else this.faceFrames = 0;

      if (this.faceFrames >= 12) {
        checks.push({
          band: "fix",
          text: "Move your hand from your mouth",
          severity: 1.6,
        });
      }
    }

    // ── Slump: sinking below your own best height ────────────────────
    const nose = at(0);
    if (nose) {
      const headRatio = (shoulderMid[1] - nose[1]) / width;
      if (headRatio > 0) {
        this.baseHead = Math.max(this.baseHead * 0.9995, headRatio);

        const rel = this.baseHead > 1e-3 ? headRatio / this.baseHead : 1;
        if (rel < T.headDrop[0]) this.lowFrames++;
        else this.lowFrames = 0;

        if (this.lowFrames >= DROP_HOLD_FRAMES) {
          checks.push({
            band: rel < T.headDrop[1] ? "fix" : "watch",
            text: "Sit back up — you've sunk",
            severity: rel < T.headDrop[1] ? 1.5 : 1.1,
          });
        }
      }
    }

    const worst = checks
      .filter((c) => c.band !== "good")
      .sort((a, b) => b.severity - a.severity)[0];

    if (worst) {
      this.clearFrames = 0;
      this._raise(worst.text, worst.band, now);
      if (this.baseAge < 60) {
        this.onCalibration({ active: true, progress: Math.min(100, Math.round((this.baseAge / 60) * 100)) });
      } else {
        this.onCalibration({ active: false, progress: 100 });
      }
      return;
    }

    if (!chinMeasured || this.baseAge < 60) {
      this.onCalibration({ active: true, progress: Math.min(100, Math.round((this.baseAge / 60) * 100)) });
      this._raise(null, "idle", now);
      return;
    }

    this.onCalibration({ active: false, progress: 100 });
    this.clearFrames++;
    if (this.clearFrames >= GOOD_HOLD_FRAMES) {
      this._raise("Good posture", "good", now);
    }
  }

  /**
   * Hold a cue for a beat before replacing it.
   */
  _raise(text, tone, now) {
    const HOLD_MS = 1200;
    if (text === this.cue) {
      this.tone = tone;
      return;
    }
    // Corrections may interrupt the hold; praise may not.
    if (tone !== "fix" && this.cue !== null && now - this.cueSince < HOLD_MS) return;

    this.cue = text;
    this.tone = tone;
    this.cueSince = now;
    this.onGuidance({ text, tone });
  }

  /**
   * Skeleton overlay with digital zoom, glow effects, and gradient-tinted bones.
   * The overlay is always drawn to the full canvas; the video element's CSS
   * transform handles the actual mirroring, so we mirror the x coords here too.
   */
  _draw(landmarks) {
    const { ctx, canvas } = this;
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    const zoom = this.zoomLevel;
    const colour = TONE[this.tone] || TONE.idle;
    const glowColour = GLOW[this.tone] || GLOW.idle;

    // Apply digital zoom — scale from centre of canvas.
    ctx.save();
    if (zoom > 1.0) {
      ctx.translate(w / 2, h / 2);
      ctx.scale(zoom, zoom);
      ctx.translate(-w / 2, -h / 2);
    }

    // ── Framing guide grid ──────────────────────────────────────────
    if (this.showGrid) {
      ctx.save();
      ctx.strokeStyle = "rgba(127, 212, 255, 0.25)";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 4]);

      // Vertical thirds
      ctx.beginPath();
      ctx.moveTo(w / 3, 0); ctx.lineTo(w / 3, h);
      ctx.moveTo((2 * w) / 3, 0); ctx.lineTo((2 * w) / 3, h);
      // Horizontal thirds
      ctx.moveTo(0, h / 3); ctx.lineTo(w, h / 3);
      ctx.moveTo(0, (2 * h) / 3); ctx.lineTo(w, (2 * h) / 3);
      ctx.stroke();

      // Eye level annotation at 1/3 height
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(127, 212, 255, 0.5)";
      ctx.font = "8px monospace";
      ctx.fillText("EYE LEVEL", 8, h / 3 - 4);

      // Framing corner brackets
      const bracketSize = Math.max(12, w * 0.04);
      const pad = 10;
      ctx.strokeStyle = "rgba(127, 212, 255, 0.4)";
      ctx.lineWidth = 1.5;
      // Top-left
      ctx.beginPath();
      ctx.moveTo(pad, pad + bracketSize); ctx.lineTo(pad, pad); ctx.lineTo(pad + bracketSize, pad);
      // Top-right
      ctx.moveTo(w - pad - bracketSize, pad); ctx.lineTo(w - pad, pad); ctx.lineTo(w - pad, pad + bracketSize);
      // Bottom-left
      ctx.moveTo(pad, h - pad - bracketSize); ctx.lineTo(pad, h - pad); ctx.lineTo(pad + bracketSize, h - pad);
      // Bottom-right
      ctx.moveTo(w - pad - bracketSize, h - pad); ctx.lineTo(w - pad, h - pad); ctx.lineTo(w - pad, h - pad - bracketSize);
      ctx.stroke();
      ctx.restore();
    }

    if (!landmarks) {
      ctx.restore();
      return;
    }

    // Point mapper — mirrors unless "flipped" (natural) mode is on.
    // In mirrored mode (default): x is inverted so skeleton matches the
    // CSS-flipped video.  In natural/flip mode: x is left as-is.
    const pt = (i) => {
      const p = landmarks[i];
      if (!p || (p.visibility ?? 1) < 0.5) return null;
      const x = this.flipped ? p.x * w : (1 - p.x) * w;
      return [x, p.y * h];
    };

    const boneWidth = Math.max(2, w / 200);
    const jointRadius = Math.max(3.5, w / 155);

    // ── Glow pass (blurred, wide, semi-transparent) ──────────────────
    ctx.save();
    ctx.filter = `blur(${Math.round(boneWidth * 2.5)}px)`;
    ctx.lineWidth = boneWidth * 2.2;
    ctx.strokeStyle = glowColour;
    ctx.lineCap = "round";
    for (const [a, b] of BONES) {
      const pa = pt(a), pb = pt(b);
      if (!pa || !pb) continue;
      ctx.beginPath();
      ctx.moveTo(pa[0], pa[1]);
      ctx.lineTo(pb[0], pb[1]);
      ctx.stroke();
    }
    ctx.filter = "none";
    ctx.restore();

    // ── Crisp bone pass ──────────────────────────────────────────────
    ctx.lineWidth = boneWidth;
    ctx.strokeStyle = colour;
    ctx.lineCap = "round";

    for (const [a, b] of BONES) {
      const pa = pt(a), pb = pt(b);
      if (!pa || !pb) continue;
      ctx.beginPath();
      ctx.moveTo(pa[0], pa[1]);
      ctx.lineTo(pb[0], pb[1]);
      ctx.stroke();
    }

    // ── Level reference line across the shoulders ────────────────────
    // Makes tilt legible at a glance.
    const ls = pt(11);
    const rs = pt(12);
    if (ls && rs) {
      const midY = (ls[1] + rs[1]) / 2;
      const x1 = Math.min(ls[0], rs[0]) - 18;
      const x2 = Math.max(ls[0], rs[0]) + 18;

      ctx.save();
      ctx.setLineDash([4, 5]);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.28)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x1, midY);
      ctx.lineTo(x2, midY);
      ctx.stroke();
      ctx.restore();

      // Shade the wedge between level and actual.
      if (Math.abs(ls[1] - rs[1]) > h * 0.012) {
        ctx.save();
        ctx.globalAlpha = 0.14;
        ctx.fillStyle = colour;
        ctx.beginPath();
        ctx.moveTo(ls[0], ls[1]);
        ctx.lineTo(rs[0], rs[1]);
        ctx.lineTo(rs[0], midY);
        ctx.lineTo(ls[0], midY);
        ctx.closePath();
        ctx.fill();
        ctx.restore();
      }
    }

    // ── Joints ───────────────────────────────────────────────────────
    // Inner bright dot with a soft outer ring for readability.
    for (const i of JOINTS) {
      const p = pt(i);
      if (!p) continue;

      // Outer ring
      ctx.save();
      ctx.globalAlpha = 0.4;
      ctx.fillStyle = colour;
      ctx.beginPath();
      ctx.arc(p[0], p[1], jointRadius * 2.0, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      // Inner solid dot
      ctx.fillStyle = colour;
      ctx.beginPath();
      ctx.arc(p[0], p[1], jointRadius, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.restore(); // zoom transform
  }

}

window.PoseTracker = PoseTracker;
