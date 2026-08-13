"""End-to-end API behaviour through the FastAPI test client."""

from __future__ import annotations

import pytest

from conftest import sine, to_wav_bytes


def post_audio(client, wav: bytes, session_id: str | None = None):
    data = {"session_id": session_id} if session_id else None
    return client.post(
        "/api/analyze",
        files={"audio_file": ("chunk.wav", wav, "audio/wav")},
        data=data,
    )


# ---------------------------------------------------------------------------
# Health & status
# ---------------------------------------------------------------------------
def test_health_reports_real_state(client):
    body = client.get("/api/health").json()

    assert body["version"]
    assert body["status"] in {"ok", "degraded"}
    # The old endpoint hard-coded "ready"; this one reflects reality: no
    # cloud key is configured in tests, so the resolved engine is local.
    assert body["stt_engine"] == "local"
    assert body["stt_ready"] == body["local_model_loaded"]
    assert body["llm_enabled"] is False
    assert body["active_sessions"] == 0


def test_health_reports_key_presence_not_key_values(client):
    config = client.get("/api/health").json()["config"]

    assert config["groq_key_configured"] is False
    assert config["openai_key_configured"] is False
    assert "groq_api_key" not in config
    assert "openai_api_key" not in config
    assert "openrouter_api_key" not in config


def test_status_endpoint(client):
    body = client.get("/api/status").json()
    assert body["status"] == "ready"
    assert "stt_engine" in body


def test_every_response_carries_a_request_id(client):
    response = client.get("/api/status")
    assert response.headers.get("X-Request-ID")


# ---------------------------------------------------------------------------
# Upload validation
# ---------------------------------------------------------------------------
def test_oversized_upload_is_rejected(client, oversized_wav):
    response = post_audio(client, oversized_wav)
    assert response.status_code == 413
    assert "limit" in response.json()["message"].lower()


def test_undecodable_upload_is_rejected(client):
    response = post_audio(client, b"not audio at all")
    assert response.status_code == 400
    assert response.json()["status"] == "error"


def test_empty_upload_is_rejected(client):
    response = post_audio(client, b"")
    assert response.status_code == 400


def test_missing_file_is_a_validation_error(client):
    response = client.post("/api/analyze")
    assert response.status_code == 422
    assert response.json()["status"] == "error"


def test_errors_never_include_a_traceback(client):
    """The old handler returned traceback.format_exc() to the caller."""
    body = client.post("/api/analyze", files={
        "audio_file": ("chunk.wav", b"garbage", "audio/wav")
    }).json()
    serialised = str(body)
    assert "Traceback" not in serialised
    assert "File \"" not in serialised


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def test_analyze_returns_the_full_schema(client, tone_wav, stub_transcribe):
    body = post_audio(client, tone_wav).json()

    for key in ("message", "pitch", "pitch_std", "energy", "energy_db", "wpm",
                "fillers", "keywords", "clarity", "transcript",
                "confidence_score", "confidence_breakdown", "feedback",
                "status", "pause_ratio", "degraded", "warnings"):
        assert key in body, f"missing {key}"


def test_analyze_detects_fillers_and_keywords_in_the_transcript(client, tone_wav,
                                                               stub_transcribe):
    body = post_audio(client, tone_wav).json()
    assert body["fillers"]["total_fillers"] > 0
    assert "innovative" in body["keywords"]["found_keywords"]


def test_silence_skips_transcription(client, silence_wav, stub_transcribe):
    body = post_audio(client, silence_wav).json()
    assert body["transcript"] == ""
    assert "silence_skipped" in body["warnings"]
    assert not stub_transcribe, "STT should not run on silence"


def test_source_sample_rate_is_reported(client, stub_transcribe):
    wav = to_wav_bytes(sine(seconds=1.0, sr=44_100), sr=44_100)
    body = post_audio(client, wav).json()
    assert body["source_sample_rate"] == 44_100


def test_wpm_is_rate_correct_for_non_16k_input(client, stub_transcribe):
    """Same speech, two container rates -> the same WPM."""
    body_16k = post_audio(client, to_wav_bytes(sine(seconds=2.0))).json()
    body_48k = post_audio(
        client, to_wav_bytes(sine(seconds=2.0, sr=48_000), sr=48_000)
    ).json()
    assert body_16k["wpm"] == pytest.approx(body_48k["wpm"], rel=0.02)


# ---------------------------------------------------------------------------
# Degraded results
# ---------------------------------------------------------------------------
def test_stt_failure_is_flagged_not_disguised(client, tone_wav, failing_transcribe):
    body = post_audio(client, tone_wav).json()

    assert body["degraded"] is True
    assert "stt_unavailable" in body["warnings"]
    assert "confidence_acoustic_only" in body["warnings"]
    # Acoustic metrics are still genuine, so they must still be returned.
    assert body["energy"] > 0
    assert body["pitch"] > 0


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
def test_session_lifecycle(client, tone_wav, stub_transcribe):
    session_id = client.post("/api/session").json()["session_id"]

    for _ in range(3):
        body = post_audio(client, tone_wav, session_id).json()
        assert body["session_id"] == session_id
        assert body["session"]["chunk_count"] > 0

    summary = client.get(f"/api/session/{session_id}").json()
    assert summary["chunk_count"] == 3
    assert summary["total_fillers"] > 0
    assert summary["avg_confidence"] is not None

    report = client.delete(f"/api/session/{session_id}").json()
    assert report["transcript"]
    assert len(report["transcript_segments"]) == 3
    assert client.get(f"/api/session/{session_id}").status_code == 404


def test_smoothed_confidence_is_returned_only_with_a_session(client, tone_wav,
                                                             stub_transcribe):
    assert post_audio(client, tone_wav).json()["confidence_smoothed"] is None

    session_id = client.post("/api/session").json()["session_id"]
    body = post_audio(client, tone_wav, session_id).json()
    assert body["confidence_smoothed"] is not None


def test_transcript_context_is_carried_between_chunks(client, tone_wav,
                                                      stub_transcribe):
    session_id = client.post("/api/session").json()["session_id"]
    post_audio(client, tone_wav, session_id)
    post_audio(client, tone_wav, session_id)

    assert stub_transcribe[0]["prompt"] is None      # nothing to carry yet
    assert stub_transcribe[1]["prompt"], "second chunk should get context"


def test_unknown_session_is_reported_but_does_not_fail_analysis(client, tone_wav,
                                                                stub_transcribe):
    body = post_audio(client, tone_wav, "does-not-exist").json()
    assert "session_not_found" in body["warnings"]
    assert body["session"] is None
    assert body["confidence_score"] >= 0


def test_missing_session_endpoints_return_404(client):
    assert client.get("/api/session/nope").status_code == 404
    assert client.get("/api/session/nope/report").status_code == 404
    assert client.delete("/api/session/nope").status_code == 404


# ---------------------------------------------------------------------------
# Diarization
# ---------------------------------------------------------------------------
def test_diarize_reports_disabled_with_actionable_guidance(client, tone_wav):
    """Disabled by default; the error must say how to turn it on."""
    response = client.post(
        "/api/diarize", files={"audio_file": ("full.wav", tone_wav, "audio/wav")}
    )
    assert response.status_code == 503
    message = response.json()["message"]
    assert "ML_DIARIZATION_ENABLED" in message


def test_diarize_validates_before_checking_the_model(client):
    """A disabled feature should not accept junk uploads either."""
    response = client.post(
        "/api/diarize", files={"audio_file": ("x.wav", b"", "audio/wav")}
    )
    assert response.status_code in (400, 503)


# ---------------------------------------------------------------------------
# Speaker continuity
# ---------------------------------------------------------------------------
def test_speaker_similarity_is_null_when_tracking_is_off(client, tone_wav,
                                                         stub_transcribe):
    session_id = client.post("/api/session").json()["session_id"]
    body = post_audio(client, tone_wav, session_id).json()
    assert body["speaker_similarity"] is None
    assert "speaker_changed" not in body["warnings"]


def test_session_exposes_speaker_fields(client, tone_wav, stub_transcribe):
    session_id = client.post("/api/session").json()["session_id"]
    post_audio(client, tone_wav, session_id)
    summary = client.get(f"/api/session/{session_id}").json()
    assert "speaker_similarity" in summary
    assert summary["speaker_changes"] == 0


# ---------------------------------------------------------------------------
# Frontend serving
# ---------------------------------------------------------------------------
def test_index_is_served_at_the_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "SpeakTwin" in response.text


def test_relative_assets_resolve_from_the_root(client):
    """index.html loads these with relative paths; both must 200."""
    assert client.get("/style.css").status_code == 200
    assert client.get("/app.js").status_code == 200


def test_static_mount_still_works(client):
    assert client.get("/static/style.css").status_code == 200
