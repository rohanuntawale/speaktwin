/**
 * ============================================================
 *  SpeakTwin – AI Communication Mirror
 *  Frontend Application Logic (Vanilla JS)
 * ============================================================
 *
 *  Handles:
 *    • Start / Stop microphone via API
 *    • Polling /api/analyze for real-time results
 *    • Non-blocking UI updates with async fetch
 *    • Animated meter bars, confidence ring, feedback list
 *    • Transcript display with filler highlighting
 */

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const API_BASE = "http://localhost:8002";    // Explicitly point to backend port 8002
const CHUNK_DURATION_MS = 2500;      // Audio snippet duration in ms
const POLL_INTERVAL_MS = 2800;      // Slightly under chunk duration for continuity
const METER_BAR_COUNT = 20;         // Number of bars in each meter

// ---------------------------------------------------------------------------
// DOM References
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);

const DOM = {
    micBtn:             $("mic-btn"),
    micIconSvg:         $("mic-icon-svg"),
    controlLabel:       $("control-label"),
    logoPulse:          $("logo-pulse"),
    statusBadge:        $("status-badge"),
    statusDot:          $("status-dot"),
    statusText:         $("status-text"),

    // Meters
    energyBars:         $("energy-bars"),
    pitchBars:          $("pitch-bars"),
    energyValue:        $("energy-value"),
    pitchValue:         $("pitch-value"),

    // Stats
    wpmValue:           $("wpm-value"),
    fillerValue:        $("filler-value"),
    pauseValue:         $("pause-value"),
    clarityValue:       $("clarity-value"),

    // Keywords
    keywordPool:        $("keyword-pool"),
    keywordPlaceholder: $("keyword-placeholder"),

    // Confidence
    scoreRingFill:      $("score-ring-fill"),
    scoreValue:         $("score-value"),
    bdWpm:              $("bd-wpm"),
    bdPitch:            $("bd-pitch"),
    bdEnergy:           $("bd-energy"),
    bdFiller:           $("bd-filler"),

    // Feedback
    liveDot:            $("live-dot"),
    feedbackPrimary:    $("feedback-primary"),
    feedbackList:       $("feedback-list"),

    // Transcript
    transcriptToggle:   $("transcript-toggle"),
    transcriptBody:     $("transcript-body"),
    transcriptPlaceholder: $("transcript-placeholder"),
    transcriptText:     $("transcript-text"),

    // Filler details
    fillerDetails:      $("filler-details"),
};


// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];
let recordingInterval = null;
let transcriptHistory = [];   // All transcript segments

// App cumulative state
let appState = {
    totalFillers: 0,
    fillerDetails: {},
    totalKeywords: 0,
    keywordDetails: {}
};

// ---------------------------------------------------------------------------
// Main Controls
// ---------------------------------------------------------------------------
DOM.micBtn.addEventListener("click", () => {
    if (!isRecording) {
        startClientRecording();
    } else {
        stopClientRecording();
    }
});

function setRecordingState(recording) {
    isRecording = recording;
    
    // Toggle mic button class
    DOM.micBtn.classList.toggle("recording", recording);
    DOM.controlLabel.textContent = recording ? "Tap to stop analysis" : "Tap to start analysis";
    
    // Toggle system status indicator
    if (DOM.statusBadge) DOM.statusBadge.classList.toggle("recording", recording);
    if (DOM.statusDot) DOM.statusDot.classList.toggle("recording", recording);
    if (DOM.statusText) DOM.statusText.textContent = recording ? "Listening" : "Idle";
    
    // Aesthetic animations
    if (DOM.logoPulse) DOM.logoPulse.classList.toggle("recording", recording);
    if (DOM.liveDot) DOM.liveDot.classList.toggle("active", recording);

    // Swap mic icon to stop square
    if (recording) {
        DOM.micIconSvg.innerHTML = `<rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" stroke="none"/>`;
    } else {
        DOM.micIconSvg.innerHTML = `<path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/>`;
        setTimeout(() => updatePrimaryFeedback("Session complete. Click mic to analyze another snippet.", "info"), 1000);
    }
}


// ---------------------------------------------------------------------------
// Audio Pipeline (Vanilla JS Web Audio -> WAV -> Backend)
// ---------------------------------------------------------------------------

/**
 * Creates a valid WAV file blob from Float32Array PCM data
 */
function encodeWav(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    const writeString = (view, offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM format
    view.setUint16(22, 1, true); // 1 channel
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);

    // Write PCM samples
    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }

    return new Blob([view], { type: 'audio/wav' });
}


async function startClientRecording() {
    try {
        console.log("[SpeakTwin] Requesting microphone access...");
        
        // Reset state on new session
        appState = { totalFillers: 0, fillerDetails: {}, totalKeywords: 0, keywordDetails: {} };
        if (DOM.fillerValue) DOM.fillerValue.textContent = "0";
        if (DOM.keywordPool) DOM.keywordPool.innerHTML = '<p class="transcript-placeholder" id="keyword-placeholder">Speak target words to highlight...</p>';
        if (DOM.fillerDetails) DOM.fillerDetails.innerHTML = '<p class="transcript-placeholder">No fillers detected yet</p>';
        if (DOM.transcriptText) DOM.transcriptText.innerHTML = "";
        if (DOM.transcriptPlaceholder) DOM.transcriptPlaceholder.style.display = "block";
        
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        console.log("[SpeakTwin] Microphone access granted.");
        
        audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        source = audioCtx.createMediaStreamSource(stream);

        processor = audioCtx.createScriptProcessor(4096, 1, 1);
        
        let sampleAccumulator = [];
        const requiredSamples = 16000 * (CHUNK_DURATION_MS / 1000); // e.g. 40000 for 2.5s

        processor.onaudioprocess = async (e) => {
            if (!isRecording) return;
            
            const pcmData = e.inputBuffer.getChannelData(0);
            sampleAccumulator.push(...pcmData);

            if (sampleAccumulator.length >= requiredSamples) {
                // Take a snapshot
                const samplesToSend = new Float32Array(sampleAccumulator.slice(0, requiredSamples));
                // Keep the rest (overlap or overflow)
                sampleAccumulator = sampleAccumulator.slice(requiredSamples);
                
                console.log(`[SpeakTwin] Sending ${samplesToSend.length} samples as WAV...`);
                const blob = encodeWav(samplesToSend, 16000);
                const result = await apiAnalyze(blob);
                console.log("[SpeakTwin] Analysis result:", result);
                if (result) handleAnalysisResult(result);
            }
        };

        source.connect(processor);
        processor.connect(audioCtx.destination);

        setRecordingState(true);
        isRecording = true;

        // Cleanup on stop
        window._stopRec = () => {
            stream.getTracks().forEach(t => t.stop());
            processor.disconnect();
            source.disconnect();
            audioCtx.close();
        };

    } catch (err) {
        console.error("[error] Microphone access denied:", err);
        updatePrimaryFeedback("Microphone access denied. Please check site permissions.", "poor");
    }
}

/**
 * Stops the client-side recording.
 */
function stopClientRecording() {
    isRecording = false;
    if (window._stopRec) window._stopRec();
    setRecordingState(false);
    updateMeterBars(DOM.energyBars, 0);
    updateMeterBars(DOM.pitchBars, 0);
    setServerStatus("Idle", "inactive");
    
    // Reset transient visual state optionally, or keep history.
    // For now, let's keep the history visible until the next "Start" clears it.
}

// ---------------------------------------------------------------------------
// API Calls
// ---------------------------------------------------------------------------

/**
 * POST the audio blob to the backend for analysis.
 */
async function apiAnalyze(blob) {
    try {
        setServerStatus("Analyzing...", "active");
        const formData = new FormData();
        formData.append("audio_file", blob, "chunk.wav");

        const res = await fetch(`${API_BASE}/api/analyze`, {
            method: "POST",
            body: formData,
        });

        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.message || "Server Error");
        }
        
        setServerStatus("Server OK", "active");
        return await res.json();
    } catch (err) {
        console.error("Analysis fetch error:", err);
        setServerStatus("Connection Error", "error");
        updatePrimaryFeedback(`Connection error: ${err.message}`, "poor");
        return null;
    }
}

function setServerStatus(text, state) {
    if (!DOM.statusText || !DOM.statusDot) return;
    DOM.statusText.textContent = text;
    DOM.statusDot.className = "status-dot";
    if (state === "active") DOM.statusDot.classList.add("recording");
}

// ---------------------------------------------------------------------------
// UI Update Functions
// ---------------------------------------------------------------------------

/**
 * Central handler for analysis data.
 */
function handleAnalysisResult(data) {
    if (!data || data.status === "error") return;

    // -- Update meters --
    const energyNorm = Math.min(data.energy / 0.15, 1);
    updateMeterBars(DOM.energyBars, energyNorm, 0.85);
    DOM.energyValue.textContent = data.energy.toFixed(3);

    const pitchNorm = data.pitch > 0 ? Math.min(data.pitch / 350, 1) : 0;
    updateMeterBars(DOM.pitchBars, pitchNorm, 0.85);
    DOM.pitchValue.textContent = data.pitch > 0 ? `${Math.round(data.pitch)}Hz` : "—";

    // -- Update stats --
    DOM.wpmValue.textContent = Math.round(data.wpm);
    DOM.pauseValue.textContent =
        data.pause_ratio !== undefined
            ? `${Math.round(data.pause_ratio * 100)}%`
            : "—";

    if (DOM.clarityValue && data.clarity !== undefined) {
        DOM.clarityValue.textContent = Math.round(data.clarity);
    }

    // -- Update Confidence Circle & Breakdown --
    updateConfidenceScore(data.confidence_score || 0);
    updateBreakdown(data.confidence_breakdown);

    // -- Update primary insight --
    updatePrimaryFeedback(data.message, data.status);

    // -- Update transcript with highlighting --
    if (data.transcript) {
        let highlightedText = data.transcript;
        
        // Emphasize keywords (green)
        if (data.keywords && data.keywords.keywords_list) {
            data.keywords.keywords_list.forEach(kw => {
                const regex = new RegExp(`\\b${kw}\\b`, 'gi');
                highlightedText = highlightedText.replace(regex, `<span class="keyword-highlight">$&</span>`);
            });
        }
        
        // Emphasize fillers (red)
        if (data.fillers && data.fillers.details) {
            Object.keys(data.fillers.details).forEach(fw => {
                const regex = new RegExp(`\\b${fw}\\b`, 'gi');
                highlightedText = highlightedText.replace(regex, `<span class="filler-highlight">$&</span>`);
            });
        }
        
        appendTranscriptHTML(highlightedText);
    }
    
    // Accumulate fillers
    if (data.fillers && data.fillers.total_fillers > 0) {
        appState.totalFillers += data.fillers.total_fillers;
        if (DOM.fillerValue) DOM.fillerValue.textContent = appState.totalFillers;
        
        for (const [fw, count] of Object.entries(data.fillers.details)) {
            appState.fillerDetails[fw] = (appState.fillerDetails[fw] || 0) + count;
        }
        updateFillerDetails(appState.fillerDetails);
    }
    
    // Accumulate keywords
    if (data.keywords && data.keywords.total_keywords > 0) {
        appState.totalKeywords += data.keywords.total_keywords;
        
        for (const [kw, count] of Object.entries(data.keywords.found_keywords)) {
            appState.keywordDetails[kw] = (appState.keywordDetails[kw] || 0) + count;
        }
        updateKeywordDetails(appState.keywordDetails);
    }
}

/**
 * Update the animated meter bars.
 */
function updateMeterBars(container, level, highThreshold = 0.8) {
    if (!container) return;
    const bars = container.querySelectorAll("span");
    const activeBars = Math.round(level * bars.length);

    bars.forEach((bar, i) => {
        if (i < activeBars) {
            bar.classList.add("active");
            const h = 4 + (level * 76) * (0.5 + Math.random() * 0.5);
            bar.style.height = `${Math.min(h, 80)}px`;
            if (level > highThreshold) bar.classList.add("high");
            else bar.classList.remove("high");
        } else {
            bar.classList.remove("active", "high");
            bar.style.height = `${4 + Math.random() * 3}px`;
        }
    });
}

function updateConfidenceScore(score) {
    if (!DOM.scoreRingFill || !DOM.scoreValue) return;
    const circumference = 2 * Math.PI * 52;
    const offset = circumference - (score / 100) * circumference;
    DOM.scoreRingFill.style.strokeDashoffset = offset;

    if (score >= 75) DOM.scoreRingFill.style.stroke = "var(--accent-green)";
    else if (score >= 50) DOM.scoreRingFill.style.stroke = "var(--accent-cyan)";
    else if (score >= 25) DOM.scoreRingFill.style.stroke = "var(--accent-yellow)";
    else DOM.scoreRingFill.style.stroke = "var(--accent-orange)";

    DOM.scoreValue.textContent = Math.round(score);
}

function updateBreakdown(breakdown) {
    if (!breakdown) return;
    const upd = (el, val) => { if (el) el.style.width = `${Math.round(val || 0)}%`; };
    upd(DOM.bdWpm, breakdown.wpm);
    upd(DOM.bdPitch, breakdown.pitch_variation);
    upd(DOM.bdEnergy, breakdown.energy);
    upd(DOM.bdFiller, breakdown.filler_usage);
}

function updatePrimaryFeedback(message, type) {
    if (!DOM.feedbackPrimary || !message) return;
    
    DOM.feedbackPrimary.style.opacity = 0;
    DOM.feedbackPrimary.style.transform = "translateY(10px)";
    
    setTimeout(() => {
        DOM.feedbackPrimary.textContent = message;
        DOM.feedbackPrimary.className = `feedback-primary ${type}`;
        DOM.feedbackPrimary.style.opacity = 1;
        DOM.feedbackPrimary.style.transform = "translateY(0)";
    }, 300);
}

function appendTranscriptHTML(htmlContent) {
    if (!DOM.transcriptText || !htmlContent.trim()) return;
    if (DOM.transcriptPlaceholder) DOM.transcriptPlaceholder.style.display = "none";

    const p = document.createElement("div");
    p.className = "new-segment";
    p.innerHTML = htmlContent;
    DOM.transcriptText.appendChild(p);
    
    // Smooth scroll to bottom
    setTimeout(() => {
        DOM.transcriptBody.scrollTo({
            top: DOM.transcriptBody.scrollHeight,
            behavior: "smooth"
        });
    }, 50);
}

function updateFillerDetails(details) {
    if (!DOM.fillerDetails) return;
    DOM.fillerDetails.innerHTML = "";
    if (!details || Object.keys(details).length === 0) {
        DOM.fillerDetails.innerHTML = '<p class="transcript-placeholder">No fillers detected yet</p>';
        return;
    }

    Object.entries(details)
        .sort((a, b) => b[1] - a[1])
        .forEach(([word, count]) => {
            const span = document.createElement("span");
            span.className = "filler-tag";
            span.innerHTML = `${word} <span class="filler-tag-count">${count}</span>`;
            DOM.fillerDetails.appendChild(span);
        });
}

function updateKeywordDetails(details) {
    if (!DOM.keywordPool) return;
    DOM.keywordPool.innerHTML = "";
    
    if (!details || Object.keys(details).length === 0) {
        DOM.keywordPool.innerHTML = '<p class="transcript-placeholder" id="keyword-placeholder">Speak target words to highlight...</p>';
        return;
    }

    Object.entries(details)
        .sort((a, b) => b[1] - a[1]) // highest first
        .forEach(([word, count]) => {
            const span = document.createElement("span");
            span.className = "keyword-pill";
            span.innerHTML = `${word} <span class="keyword-pill-count">${count}</span>`;
            DOM.keywordPool.appendChild(span);
        });
}

// End of SpeakTwin Application Logic
