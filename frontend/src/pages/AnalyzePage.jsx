import { useState, useEffect, useRef } from "react";
import { Music, Play, Pause } from "lucide-react";
import { api } from "../api";
import WaveSurfer from "wavesurfer.js";

// ── Genre colour map ──────────────────────────────────────────────────
const GENRE_COLORS = {
  hiphop:    "#a855f7", trap:      "#ef4444", edm:       "#06b6d4",
  rock:      "#f59e0b", metal:     "#dc2626", jazz:      "#10b981",
  reggae:    "#84cc16", dnb:       "#f97316", ambient:   "#6366f1",
  afrobeats: "#ec4899", funk:      "#eab308", latin:     "#14b8a6",
  blues:     "#3b82f6", classical: "#6d28d9", country:   "#d97706",
  disco:     "#ec4899", pop:       "#f43f5e",
};

function formatGenre(g) {
  return g
    .replace("hiphop", "Hip-Hop")
    .replace("dnb", "Drum & Bass")
    .replace(/\b\w/g, c => c.toUpperCase());
}

// ── Genre confidence row ──────────────────────────────────────────────
function GenreRow({ genre, confidence, primary }) {
  const color = GENRE_COLORS[genre] || "var(--accent-primary)";
  const pct   = Math.round(confidence * 100);
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 12,
      padding: "10px 0",
      borderBottom: "1px solid var(--border-color)",
    }}>
      <div style={{
        width: 7, height: 7, borderRadius: "50%", flexShrink: 0,
        background: color,
        boxShadow: primary ? `0 0 5px ${color}` : "none",
      }} />
      <span style={{
        width: 112, flexShrink: 0,
        fontSize: "0.82rem", letterSpacing: "0.01em",
        fontWeight: primary ? 600 : 400,
        color: primary ? "var(--text-primary)" : "var(--text-secondary)",
      }}>
        {formatGenre(genre)}
      </span>
      <div style={{
        flex: 1, height: 3, borderRadius: 2,
        background: "rgba(255,255,255,0.06)",
      }}>
        <div style={{
          width: `${pct}%`, height: "100%", borderRadius: 2,
          background: primary ? color : `${color}70`,
          transition: "width 0.5s ease",
        }} />
      </div>
      <span style={{
        width: 34, textAlign: "right", flexShrink: 0,
        fontSize: "0.7rem", fontWeight: 600,
        fontFamily: "var(--font-mono)",
        color: primary ? color : "var(--text-muted)",
      }}>
        {pct}%
      </span>
    </div>
  );
}

// ── Waveform player ───────────────────────────────────────────────────
function WaveformPlayer({ file, beatTimes, duration }) {
  const containerRef = useRef(null);
  const wsRef        = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [ready,     setReady]     = useState(false);

  useEffect(() => {
    if (!file || !containerRef.current) return;
    const objectUrl = URL.createObjectURL(file);
    const ws = WaveSurfer.create({
      container:     containerRef.current,
      waveColor:     "rgba(124, 58, 237, 0.4)",
      progressColor: "rgba(168, 85, 247, 0.85)",
      cursorColor:   "rgba(241,245,249,0.4)",
      barWidth: 2, barGap: 1, barRadius: 2,
      height: 80, normalize: true,
    });
    ws.load(objectUrl);
    ws.on("ready",  () => setReady(true));
    ws.on("play",   () => setIsPlaying(true));
    ws.on("pause",  () => setIsPlaying(false));
    ws.on("finish", () => setIsPlaying(false));
    wsRef.current = ws;
    return () => {
      ws.destroy();
      URL.revokeObjectURL(objectUrl);
      wsRef.current = null;
      setIsPlaying(false);
      setReady(false);
    };
  }, [file]);

  return (
    <div className="card" style={{ padding: 20 }}>
      <div style={{
        display: "flex", justifyContent: "space-between",
        alignItems: "center", marginBottom: 14,
      }}>
        <p className="section-label">
          Waveform — {beatTimes?.length ?? 0} beats detected
        </p>
        <button
          className="btn btn-secondary btn-sm"
          onClick={() => wsRef.current?.playPause()}
          disabled={!ready}
          style={{ minWidth: 72 }}
        >
          {isPlaying
            ? <><Pause size={12} /> Pause</>
            : <><Play  size={12} /> Play</>}
        </button>
      </div>
      <div style={{ position: "relative" }}>
        <div ref={containerRef} />
        {beatTimes && duration && beatTimes.map((t, i) => (
          <div key={i} style={{
            position: "absolute",
            left: `${(t / duration) * 100}%`,
            top: 0, bottom: 0, width: 1,
            background: "rgba(236,72,153,0.4)",
            pointerEvents: "none",
          }} />
        ))}
      </div>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────
export default function AnalyzePage({ tracks, setTracks }) {
  const [selected, setSelected] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState(null);

  const handleAnalyze = async (track) => {
    if (loading) return;
    setSelected(track);
    setLoading(true);
    setError(null);
    setAnalysis(null);
    try {
      const result = await api.analyzeFile(track.file_id);
      setAnalysis(result);
      setTracks(prev => prev.map(t =>
        t.file_id === track.file_id
          ? { ...t, analyzed: true, bpm: result.bpm, beat_times: result.beat_times, genre: result.genre }
          : t
      ));
    } catch (err) {
      setError(err.message);
    }
    setLoading(false);
  };

  return (
    <div className="animate-in">
      <div className="page-header">
        <h2><span className="gradient-text">Analyze</span> Audio</h2>
        <p>Select a track to detect beats, estimate BPM, and classify genre.</p>
      </div>

      {tracks.length === 0 ? (
        <div className="card" style={{ textAlign: "center", padding: "48px 24px" }}>
          <p style={{ color: "var(--text-muted)", fontSize: "0.88rem" }}>
            No tracks uploaded yet —{" "}
            <strong style={{ color: "var(--text-secondary)" }}>Upload</strong> a file first.
          </p>
        </div>
      ) : (
        <div style={{ display: "grid", gridTemplateColumns: "252px 1fr", gap: 20 }}>

          {/* ── Track selector ─────────────────────────────────── */}
          <div>
            <p className="section-label">Tracks</p>
            <div className="track-list">
              {tracks.map((t, i) => (
                <div
                  key={i}
                  className={`track-item ${selected?.file_id === t.file_id ? "selected" : ""}`}
                  onClick={() => handleAnalyze(t)}
                >
                  <div className="track-icon">
                    <Music size={17} color="white" />
                  </div>
                  <div className="track-info">
                    <div className="track-name">{t.filename}</div>
                    <div className="track-meta">
                      {t.duration}s{t.genre ? ` · ${formatGenre(t.genre)}` : ""}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* ── Results panel ──────────────────────────────────── */}
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

            {/* Loading */}
            {loading && (
              <div className="card" style={{ textAlign: "center", padding: "52px 24px" }}>
                <div className="spinner" style={{ marginBottom: 14 }} />
                <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>
                  Analyzing audio
                </p>
                <p style={{ color: "var(--text-muted)", fontSize: "0.76rem", marginTop: 5 }}>
                  beat detection · BPM estimation · genre classification
                </p>
              </div>
            )}

            {/* Error */}
            {error && (
              <div className="status status-error">{error}</div>
            )}

            {/* Results */}
            {analysis && !loading && (
              <div className="animate-in" style={{ display: "flex", flexDirection: "column", gap: 16 }}>

                {/* Stats */}
                <div className="analysis-grid">
                  {[
                    { v: analysis.bpm.toFixed(1),                         l: "BPM",         cl: "gradient"                  },
                    { v: analysis.beat_times.length,                      l: "Beats",       cl: "var(--accent-secondary)"   },
                    { v: `${analysis.duration.toFixed(1)}s`,              l: "Duration",    cl: "var(--success)"            },
                    { v: `${(analysis.sample_rate / 1000).toFixed(1)}k`,  l: "Sample Rate", cl: "var(--warning)"            },
                  ].map(({ v, l, cl }) => (
                    <div className="stat-card" key={l}>
                      <div
                        className={`stat-value${cl === "gradient" ? " gradient-text" : ""}`}
                        style={cl !== "gradient" ? { color: cl } : {}}
                      >
                        {v}
                      </div>
                      <div className="stat-label">{l}</div>
                    </div>
                  ))}
                </div>

                {/* Genre */}
                {analysis.genre && analysis.genre !== "unknown" && (
                  <div className="card" style={{ padding: 20 }}>
                    <p className="section-label" style={{ marginBottom: 6 }}>Genre Classification</p>
                    {(analysis.genre_top3?.length > 0
                      ? analysis.genre_top3
                      : [{ genre: analysis.genre, confidence: analysis.genre_confidence }]
                    ).map((g, i) => (
                      <GenreRow
                        key={g.genre}
                        genre={g.genre}
                        confidence={g.confidence}
                        primary={i === 0}
                      />
                    ))}
                  </div>
                )}

                {/* Waveform */}
                <WaveformPlayer
                  file={selected?.originalFile}
                  beatTimes={analysis.beat_times}
                  duration={analysis.duration}
                />

                {/* Beat timestamps */}
                <div className="card" style={{ padding: 20, maxHeight: 172, overflow: "auto" }}>
                  <p className="section-label" style={{ marginBottom: 10 }}>Beat Timestamps (s)</p>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
                    {analysis.beat_times.map((t, i) => (
                      <span key={i} style={{
                        background: "var(--bg-secondary)",
                        border: "1px solid var(--border-color)",
                        padding: "3px 7px", borderRadius: 3,
                        fontFamily: "var(--font-mono)", fontSize: "0.68rem",
                        color: "var(--text-muted)",
                      }}>
                        {t.toFixed(3)}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Empty */}
            {!loading && !analysis && !error && (
              <div className="card" style={{ textAlign: "center", padding: "52px 24px", opacity: 0.45 }}>
                <p style={{ fontSize: "0.88rem" }}>Select a track to begin analysis</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
