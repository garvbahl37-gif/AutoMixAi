import { useState, useRef } from "react";
import { Mic, MicOff, Music, ExternalLink, Search, Loader } from "lucide-react";

// Gradio-based Song Detection Space (dejavu fingerprinting)
const BACKEND_URL = "https://bharatverse11-song-detection.hf.space";
const RECORD_DURATION = 10000; // 10 seconds for better recognition

export default function ShazamPage() {
  const [listening, setListening] = useState(false);
  const [identifying, setIdentifying] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [status, setStatus] = useState("");
  const [timer, setTimer] = useState(0);

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);

  function startTimer() {
    setTimer(0);
    timerRef.current = setInterval(function() {
      setTimer(function(prev) { return prev + 1; });
    }, 1000);
  }

  function stopTimer() {
    if (timerRef.current) clearInterval(timerRef.current);
    setTimer(0);
  }

  async function startListening() {
    setListening(true);
    setResult(null);
    setError(null);
    setStatus("🎙 Listening...");
    chunksRef.current = [];

    try {
      var stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      var recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = function(e) {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = async function() {
        stream.getTracks().forEach(function(t) { t.stop(); });
        stopTimer();
        setListening(false);
        setIdentifying(true);
        setStatus("🔍 Identifying song...");
        var blob = new Blob(chunksRef.current, { type: "audio/webm" });
        await identify(blob);
        setIdentifying(false);
      };

      recorder.start();
      startTimer();

      setTimeout(function() {
        if (recorder.state === "recording") recorder.stop();
      }, RECORD_DURATION);

    } catch (e) {
      setError("Microphone access denied. Please allow mic permission in your browser.");
      setListening(false);
      setStatus("");
      stopTimer();
    }
  }

  function stopEarly() {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      mediaRecorderRef.current.stop();
    }
  }

  async function identify(blob) {
    try {
      // Send audio directly to /recognize endpoint
      const formData = new FormData();
      formData.append("file", blob, "audio.webm");

      const res = await fetch(BACKEND_URL + "/recognize", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error("Failed to connect to recognition service. The server may be starting up - please try again in a moment.");
      }

      const data = await res.json();

      if (data.status === "found") {
        setResult({
          status: "found",
          title: data.title,
          artist: data.artist || "Unknown Artist",
          album: null,
          release_date: null,
          cover: null,
          spotify: null,
          apple_music: null,
          score: data.confidence,
          timecode: null,
        });
        setStatus("✅ Song identified!");
      } else if (data.status === "not_found") {
        setError("No match found. The song may not be in the fingerprint database yet.");
        setStatus("");
      } else {
        setError(data.message || "Recognition failed. Please try again.");
        setStatus("");
      }
    } catch (e) {
      setError("Recognition failed: " + e.message);
      setStatus("");
    }
  }

  function reset() {
    setResult(null);
    setError(null);
    setStatus("");
  }

  var isActive  = listening || identifying;
  var btnBg     = listening
    ? "linear-gradient(135deg,#7c3aed,#a855f7)"
    : identifying
    ? "rgba(168,85,247,0.08)"
    : "rgba(168,85,247,0.12)";
  var btnBorder = isActive
    ? "2px solid #a855f7"
    : "2px solid rgba(168,85,247,0.3)";
  var btnShadow = listening
    ? "0 0 60px rgba(168,85,247,0.55)"
    : "0 0 30px rgba(168,85,247,0.15)";

  return (
    <div className="animate-in" style={{ maxWidth: 580, margin: "0 auto", textAlign: "center" }}>

      {/* Header */}
      <div className="page-header" style={{ marginBottom: 48 }}>
        <h2>
          <span className="gradient-text">Song</span> Recognition
        </h2>
        <p style={{ maxWidth: 420, margin: "0 auto" }}>
          Tap the mic, play music near your device, and we will identify the song in seconds.
        </p>
      </div>

      {/* Mic button area */}
      <div style={{ position: "relative", display: "inline-flex", alignItems: "center", justifyContent: "center", marginBottom: 36 }}>

        {/* Pulse ring 1 */}
        {listening
          ? <div style={{
              position: "absolute",
              width: 170, height: 170,
              borderRadius: "50%",
              border: "2px solid rgba(168,85,247,0.35)",
              top: -25, left: -25,
              animation: "ping 1.2s cubic-bezier(0,0,0.2,1) infinite",
            }} />
          : null
        }

        {/* Pulse ring 2 */}
        {listening
          ? <div style={{
              position: "absolute",
              width: 210, height: 210,
              borderRadius: "50%",
              border: "2px solid rgba(168,85,247,0.2)",
              top: -45, left: -45,
              animation: "ping 1.2s cubic-bezier(0,0,0.2,1) infinite",
              animationDelay: "0.4s",
            }} />
          : null
        }

        {/* Main button */}
        <button
          onClick={listening ? stopEarly : (identifying ? null : startListening)}
          disabled={identifying}
          style={{
            width: 120,
            height: 120,
            borderRadius: "50%",
            background: btnBg,
            border: btnBorder,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            cursor: identifying ? "not-allowed" : "pointer",
            transition: "all 0.3s ease",
            boxShadow: btnShadow,
            position: "relative",
            zIndex: 1,
          }}
        >
          {identifying
            ? <Loader size={38} color="#a855f7" style={{ animation: "spin 1s linear infinite" }} />
            : listening
            ? <MicOff size={38} color="#fff" />
            : <Mic size={38} color="#a855f7" />
          }
        </button>
      </div>

      {/* Timer bar when recording */}
      {listening
        ? <div style={{ marginBottom: 20 }}>
            <div style={{ width: 200, height: 4, borderRadius: 2, background: "rgba(255,255,255,0.06)", margin: "0 auto 10px", overflow: "hidden" }}>
              <div style={{
                height: "100%",
                width: ((timer / 10) * 100) + "%",
                background: "linear-gradient(90deg,#7c3aed,#a855f7)",
                borderRadius: 2,
                transition: "width 1s linear",
              }} />
            </div>
            <p style={{ fontSize: "0.78rem", color: "var(--text-dim)", fontFamily: "var(--font-mono)" }}>
              {timer}s / 10s — tap mic to stop early
            </p>
          </div>
        : null
      }

      {/* Status */}
      <div style={{ minHeight: 40, marginBottom: 28 }}>
        {status
          ? <p style={{ color: "#c084fc", fontWeight: 600, fontSize: "0.95rem" }}>{status}</p>
          : null
        }
        {!isActive && !status && !result && !error
          ? <p style={{ color: "var(--text-muted)", fontSize: "0.88rem" }}>
              Hold your device near the speaker and tap the mic
            </p>
          : null
        }
      </div>

      {/* Error */}
      {error
        ? <div style={{ marginBottom: 24 }}>
            <div className="status status-error animate-in" style={{ textAlign: "left", marginBottom: 12 }}>
              {error}
            </div>
            <button
              onClick={startListening}
              style={{ padding: "10px 24px", borderRadius: 10, background: "rgba(168,85,247,0.1)", border: "1px solid rgba(168,85,247,0.3)", color: "#c084fc", fontSize: "0.84rem", fontWeight: 600, cursor: "pointer", display: "inline-flex", alignItems: "center", gap: 6 }}
            >
              <Search size={13} /> Try Again
            </button>
          </div>
        : null
      }

      {/* Result card */}
      {result
        ? <div className="card animate-in" style={{ padding: 28, textAlign: "left" }}>

            {/* Song info row */}
            <div style={{ display: "flex", gap: 20, alignItems: "center", marginBottom: 22 }}>

              {/* Cover art */}
              {result.cover
                ? <img
                    src={result.cover}
                    alt="album cover"
                    style={{ width: 88, height: 88, borderRadius: 12, objectFit: "cover", flexShrink: 0, border: "1px solid rgba(255,255,255,0.08)", boxShadow: "0 8px 24px rgba(0,0,0,0.4)" }}
                  />
                : <div style={{ width: 88, height: 88, borderRadius: 12, background: "rgba(168,85,247,0.1)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, border: "1px solid rgba(168,85,247,0.2)" }}>
                    <Music size={32} color="#a855f7" />
                  </div>
              }

              {/* Text info */}
              <div style={{ flex: 1, minWidth: 0 }}>
                <p style={{ fontSize: "0.62rem", letterSpacing: "0.16em", textTransform: "uppercase", color: "rgba(192,132,252,0.7)", fontWeight: 700, marginBottom: 6, margin: "0 0 6px" }}>
                  Identified
                </p>
                <h3 style={{ fontSize: "1.15rem", fontWeight: 800, color: "var(--text-primary)", margin: "0 0 5px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {result.title}
                </h3>
                <p style={{ fontSize: "0.88rem", color: "#c084fc", fontWeight: 600, margin: "0 0 4px" }}>
                  {result.artist}
                </p>
                {result.album
                  ? <p style={{ fontSize: "0.74rem", color: "var(--text-dim)", margin: 0 }}>
                      {result.album}
                      {result.release_date ? " · " + result.release_date.slice(0, 4) : ""}
                    </p>
                  : null
                }
                {result.timecode
                  ? <p style={{ fontSize: "0.7rem", color: "var(--text-dim)", margin: "4px 0 0", fontFamily: "var(--font-mono)" }}>
                      Matched at {result.timecode}
                    </p>
                  : null
                }
              </div>
            </div>

            {/* Divider */}
            <div style={{ height: 1, background: "rgba(255,255,255,0.06)", marginBottom: 18 }} />

            {/* Action buttons */}
            <div style={{ display: "flex", gap: 10 }}>
              {result.spotify
                ? <a
                    href={result.spotify}
                    target="_blank"
                    rel="noreferrer"
                    style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", gap: 6, padding: "11px 0", borderRadius: 10, background: "rgba(30,215,96,0.08)", border: "1px solid rgba(30,215,96,0.22)", color: "#1ed760", textDecoration: "none", fontSize: "0.82rem", fontWeight: 700 }}
                  >
                    <ExternalLink size={13} /> Spotify
                  </a>
                : null
              }
              {result.apple_music
                ? <a
                    href={result.apple_music}
                    target="_blank"
                    rel="noreferrer"
                    style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", gap: 6, padding: "11px 0", borderRadius: 10, background: "rgba(252,60,60,0.08)", border: "1px solid rgba(252,60,60,0.22)", color: "#fc3c3c", textDecoration: "none", fontSize: "0.82rem", fontWeight: 700 }}
                  >
                    <ExternalLink size={13} /> Apple Music
                  </a>
                : null
              }
              <button
                onClick={reset}
                style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", gap: 6, padding: "11px 0", borderRadius: 10, background: "rgba(168,85,247,0.08)", border: "1px solid rgba(168,85,247,0.22)", color: "#c084fc", fontSize: "0.82rem", fontWeight: 700, cursor: "pointer" }}
              >
                <Search size={13} /> Identify Another
              </button>
            </div>
          </div>
        : null
      }

      {/* Empty state */}
      {!result && !error && !isActive
        ? <div style={{ marginTop: 40, display: "flex", gap: 8, justifyContent: "center", flexWrap: "wrap" }}>
            <span className="tag tag-sm">Dejavu Fingerprinting</span>
            <span className="tag tag-sm">Local Database</span>
            <span className="tag tag-sm">Results in ~3s</span>
          </div>
        : null
      }

      {/* Animations */}
      <style>{`
        @keyframes ping {
          0%   { transform: scale(1);   opacity: 0.7; }
          100% { transform: scale(1.9); opacity: 0;   }
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to   { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
