import { useState, useEffect, useRef } from "react";
import {
  Music, Play, Pause, Activity, Mic, MicOff, Zap, Piano,
  TrendingUp, Clock, Hash, Layers
} from "lucide-react";
import { api } from "../api";
import WaveSurfer from "wavesurfer.js";

// Genre color map
const GENRE_COLORS = {
  hiphop: "#a855f7", trap: "#ef4444", edm: "#06b6d4",
  rock: "#f59e0b", metal: "#dc2626", jazz: "#10b981",
  reggae: "#84cc16", dnb: "#f97316", ambient: "#6366f1",
  afrobeats: "#ec4899", funk: "#eab308", latin: "#14b8a6",
  blues: "#3b82f6", classical: "#6d28d9", country: "#d97706",
  disco: "#ec4899", pop: "#f43f5e",
};

const MOOD_CONFIG = {
  energetic: { color: "#f59e0b", bg: "rgba(245, 158, 11, 0.1)", icon: Zap },
  calm: { color: "#10b981", bg: "rgba(16, 185, 129, 0.1)", icon: Activity },
  melancholic: { color: "#a855f7", bg: "rgba(168, 85, 247, 0.1)", icon: Activity },
  intense: { color: "#ef4444", bg: "rgba(239, 68, 68, 0.1)", icon: TrendingUp },
  neutral: { color: "#64748b", bg: "rgba(100, 116, 139, 0.1)", icon: Activity },
};

function formatGenre(g) {
  return g
    .replace("hiphop", "Hip-Hop")
    .replace("dnb", "Drum & Bass")
    .replace(/\b\w/g, c => c.toUpperCase());
}

// Genre confidence row
function GenreRow({ genre, confidence, primary, index }) {
  const color = GENRE_COLORS[genre] || "var(--accent-primary)";
  const pct = Math.round(confidence * 100);

  return (
    <div
      className="animate-in"
      style={{
        display: "flex",
        alignItems: "center",
        gap: 14,
        padding: "12px 0",
        borderBottom: "1px solid var(--border-color)",
        animationDelay: `${index * 100}ms`,
      }}
    >
      <div style={{
        width: 10,
        height: 10,
        borderRadius: "50%",
        flexShrink: 0,
        background: color,
        boxShadow: primary ? `0 0 12px ${color}` : "none",
      }} />
      <span style={{
        width: 120,
        flexShrink: 0,
        fontSize: "0.88rem",
        letterSpacing: "0.01em",
        fontWeight: primary ? 700 : 500,
        color: primary ? "var(--text-primary)" : "var(--text-secondary)",
      }}>
        {formatGenre(genre)}
      </span>
      <div style={{
        flex: 1,
        height: 6,
        borderRadius: 3,
        background: "rgba(255,255,255,0.06)",
        overflow: "hidden",
      }}>
        <div style={{
          width: `${pct}%`,
          height: "100%",
          borderRadius: 3,
          background: primary ? `linear-gradient(90deg, ${color}, ${color}aa)` : `${color}50`,
          transition: "width 0.8s ease",
          boxShadow: primary ? `0 0 8px ${color}80` : "none",
        }} />
      </div>
      <span style={{
        width: 48,
        textAlign: "right",
        flexShrink: 0,
        fontSize: "0.78rem",
        fontWeight: 700,
        fontFamily: "var(--font-mono)",
        color: primary ? color : "var(--text-muted)",
      }}>
        {pct}%
      </span>
    </div>
  );
}

// Waveform player
function WaveformPlayer({ file, beatTimes, duration }) {
  const containerRef = useRef(null);
  const wsRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [ready, setReady] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);

  useEffect(() => {
    if (!file || !containerRef.current) return;
    const objectUrl = URL.createObjectURL(file);
    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: "rgba(139, 92, 246, 0.35)",
      progressColor: "rgba(168, 85, 247, 0.9)",
      cursorColor: "rgba(255, 255, 255, 0.5)",
      barWidth: 2,
      barGap: 2,
      barRadius: 3,
      height: 90,
      normalize: true,
    });
    ws.load(objectUrl);
    ws.on("ready", () => setReady(true));
    ws.on("play", () => setIsPlaying(true));
    ws.on("pause", () => setIsPlaying(false));
    ws.on("finish", () => setIsPlaying(false));
    ws.on("timeupdate", (time) => setCurrentTime(time));
    wsRef.current = ws;
    return () => {
      ws.destroy();
      URL.revokeObjectURL(objectUrl);
      wsRef.current = null;
      setIsPlaying(false);
      setReady(false);
    };
  }, [file]);

  const formatTime = (s) => {
    const mins = Math.floor(s / 60);
    const secs = Math.floor(s % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="card" style={{ padding: 24 }}>
      <div style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        marginBottom: 16,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <p className="section-label">Waveform</p>
          <span style={{
            padding: "3px 10px",
            background: "var(--bg-secondary)",
            border: "1px solid var(--border-color)",
            borderRadius: "var(--radius-full)",
            fontSize: "0.7rem",
            fontFamily: "var(--font-mono)",
            color: "var(--text-muted)",
          }}>
            {beatTimes?.length ?? 0} beats
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{
            fontFamily: "var(--font-mono)",
            fontSize: "0.78rem",
            color: "var(--text-muted)",
          }}>
            {formatTime(currentTime)} / {formatTime(duration || 0)}
          </span>
          <button
            className="btn btn-secondary btn-sm"
            onClick={() => wsRef.current?.playPause()}
            disabled={!ready}
            style={{ minWidth: 80 }}
          >
            {isPlaying
              ? <><Pause size={13} /> Pause</>
              : <><Play size={13} /> Play</>
            }
          </button>
        </div>
      </div>

      <div style={{ position: "relative", borderRadius: "var(--radius-md)", overflow: "hidden" }}>
        <div ref={containerRef} />
        {/* Beat markers */}
        {beatTimes && duration && beatTimes.map((t, i) => (
          <div
            key={i}
            style={{
              position: "absolute",
              left: `${(t / duration) * 100}%`,
              top: 0,
              bottom: 0,
              width: 1,
              background: i % 4 === 0
                ? "rgba(236, 72, 153, 0.6)"
                : "rgba(236, 72, 153, 0.25)",
              pointerEvents: "none",
            }}
          />
        ))}
      </div>
    </div>
  );
}

// Stat card component
function StatCard({ value, label, color, icon: Icon, delay = 0 }) {
  return (
    <div
      className="stat-card animate-in"
      style={{ animationDelay: `${delay}ms` }}
    >
      {Icon && (
        <div style={{
          position: "absolute",
          top: 12,
          right: 12,
          opacity: 0.15,
        }}>
          <Icon size={24} color={color === "gradient" ? "var(--accent-primary)" : color} />
        </div>
      )}
      <div
        className={`stat-value ${color === "gradient" ? "gradient-text" : ""}`}
        style={color !== "gradient" ? { color } : {}}
      >
        {value}
      </div>
      <div className="stat-label">{label}</div>
    </div>
  );
}

// Main page
export default function AnalyzePage({ tracks, setTracks }) {
  const [selected, setSelected] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

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
        <p>
          Select a track to run AI-powered analysis: beat detection, BPM estimation,
          genre classification, instrument recognition, and mood detection.
        </p>
      </div>

      {tracks.length === 0 ? (
        <div className="card empty-state">
          <div className="empty-state-icon">
            <Activity size={28} color="var(--accent-primary)" />
          </div>
          <h3>No tracks to analyze</h3>
          <p>Upload audio files first to begin analysis</p>
        </div>
      ) : (
        <div style={{ display: "grid", gridTemplateColumns: "280px 1fr", gap: 24 }}>
          {/* Track selector */}
          <div>
            <p className="section-label" style={{ marginBottom: 12 }}>Select Track</p>
            <div className="track-list">
              {tracks.map((t, i) => (
                <div
                  key={t.file_id}
                  className={`track-item ${selected?.file_id === t.file_id ? "selected" : ""}`}
                  onClick={() => handleAnalyze(t)}
                  style={{ animationDelay: `${i * 50}ms` }}
                >
                  <div className="track-icon">
                    <Music size={18} color="white" />
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

          {/* Results panel */}
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            {/* Loading */}
            {loading && (
              <div className="card" style={{ textAlign: "center", padding: "60px 24px" }}>
                <div className="spinner spinner-lg glow-pulse" style={{ margin: "0 auto 20px" }} />
                <p style={{ color: "var(--text-primary)", fontSize: "1rem", fontWeight: 600 }}>
                  Analyzing audio...
                </p>
                <p style={{ color: "var(--text-muted)", fontSize: "0.82rem", marginTop: 8 }}>
                  Beat detection · Genre · Instruments · Tags · Mood
                </p>
              </div>
            )}

            {/* Error */}
            {error && (
              <div className="status status-error animate-in">{error}</div>
            )}

            {/* Results */}
            {analysis && !loading && (
              <div className="animate-in" style={{ display: "flex", flexDirection: "column", gap: 20 }}>
                {/* Stats grid */}
                <div className="analysis-grid">
                  <StatCard value={analysis.bpm.toFixed(1)} label="BPM" color="gradient" icon={Activity} delay={0} />
                  <StatCard value={analysis.beat_times.length} label="Beats" color="var(--accent-secondary)" icon={Hash} delay={50} />
                  <StatCard value={`${analysis.duration.toFixed(1)}s`} label="Duration" color="var(--success)" icon={Clock} delay={100} />
                  <StatCard value={analysis.energy || "medium"} label="Energy" color="var(--warning)" icon={Zap} delay={150} />
                </div>

                {/* Mood & Vocals row */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                  {/* Mood */}
                  <div className="card animate-in" style={{ padding: "20px 24px", animationDelay: "200ms" }}>
                    <p className="section-label" style={{ marginBottom: 12 }}>Mood</p>
                    {(() => {
                      const mood = analysis.mood || "neutral";
                      const config = MOOD_CONFIG[mood] || MOOD_CONFIG.neutral;
                      const Icon = config.icon;
                      return (
                        <div style={{
                          display: "flex",
                          alignItems: "center",
                          gap: 12,
                        }}>
                          <div style={{
                            width: 40,
                            height: 40,
                            borderRadius: "var(--radius-md)",
                            background: config.bg,
                            border: `1px solid ${config.color}30`,
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                          }}>
                            <Icon size={20} color={config.color} />
                          </div>
                          <span style={{
                            fontSize: "1.2rem",
                            fontWeight: 700,
                            textTransform: "capitalize",
                            color: config.color,
                          }}>
                            {mood}
                          </span>
                        </div>
                      );
                    })()}
                  </div>

                  {/* Vocals */}
                  <div className="card animate-in" style={{ padding: "20px 24px", animationDelay: "250ms" }}>
                    <p className="section-label" style={{ marginBottom: 12 }}>Vocals</p>
                    <div style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 12,
                    }}>
                      <div style={{
                        width: 40,
                        height: 40,
                        borderRadius: "var(--radius-md)",
                        background: analysis.has_vocals
                          ? "rgba(139, 92, 246, 0.1)"
                          : "rgba(100, 116, 139, 0.1)",
                        border: `1px solid ${analysis.has_vocals ? "rgba(139, 92, 246, 0.3)" : "rgba(100, 116, 139, 0.2)"}`,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                      }}>
                        {analysis.has_vocals
                          ? <Mic size={20} color="var(--accent-primary)" />
                          : <MicOff size={20} color="var(--text-muted)" />
                        }
                      </div>
                      <span style={{
                        fontSize: "1.2rem",
                        fontWeight: 700,
                        color: analysis.has_vocals ? "var(--accent-primary)" : "var(--text-muted)",
                      }}>
                        {analysis.has_vocals ? "Detected" : "Instrumental"}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Instruments */}
                {analysis.dominant_instrument && analysis.dominant_instrument !== "unknown" && (
                  <div className="card animate-in" style={{ padding: 24, animationDelay: "300ms" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
                      <Piano size={16} color="var(--accent-secondary)" />
                      <p className="section-label" style={{ marginBottom: 0 }}>Dominant Instrument</p>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
                      <span style={{
                        fontSize: "1.4rem",
                        fontWeight: 800,
                        textTransform: "capitalize",
                        background: "var(--accent-gradient)",
                        WebkitBackgroundClip: "text",
                        WebkitTextFillColor: "transparent",
                      }}>
                        {analysis.dominant_instrument}
                      </span>
                      <span style={{
                        padding: "4px 12px",
                        background: "var(--bg-secondary)",
                        border: "1px solid var(--border-color)",
                        borderRadius: "var(--radius-full)",
                        fontSize: "0.78rem",
                        fontFamily: "var(--font-mono)",
                        color: "var(--text-muted)",
                      }}>
                        {Math.round((analysis.instrument_confidence || 0) * 100)}% confidence
                      </span>
                    </div>
                    {analysis.instruments_top3?.length > 1 && (
                      <div style={{ marginTop: 16, display: "flex", gap: 8, flexWrap: "wrap" }}>
                        {analysis.instruments_top3.slice(1).map((inst, i) => (
                          <span key={i} className="tag tag-sm">
                            {inst.instrument} ({Math.round(inst.confidence * 100)}%)
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* Genre */}
                {analysis.genre && analysis.genre !== "unknown" && (
                  <div className="card animate-in" style={{ padding: 24, animationDelay: "350ms" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                      <Layers size={16} color="var(--accent-secondary)" />
                      <p className="section-label" style={{ marginBottom: 0 }}>Genre Classification</p>
                    </div>
                    {(analysis.genre_top3?.length > 0
                      ? analysis.genre_top3
                      : [{ genre: analysis.genre, confidence: analysis.genre_confidence }]
                    ).map((g, i) => (
                      <GenreRow
                        key={g.genre}
                        genre={g.genre}
                        confidence={g.confidence}
                        primary={i === 0}
                        index={i}
                      />
                    ))}
                  </div>
                )}

                {/* Music Tags */}
                {analysis.tags?.length > 0 && (
                  <div className="card animate-in" style={{ padding: 24, animationDelay: "400ms" }}>
                    <p className="section-label" style={{ marginBottom: 16 }}>Music Tags</p>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
                      {analysis.tags.map((tag, i) => {
                        const score = analysis.tag_scores?.find(t => t.tag === tag)?.score || 0.5;
                        return (
                          <span
                            key={i}
                            className="tag animate-in"
                            style={{
                              animationDelay: `${450 + i * 30}ms`,
                              background: `rgba(139, 92, 246, ${0.1 + score * 0.2})`,
                              borderColor: `rgba(139, 92, 246, ${0.2 + score * 0.3})`,
                            }}
                          >
                            {tag}
                          </span>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Waveform */}
                <WaveformPlayer
                  file={selected?.originalFile}
                  beatTimes={analysis.beat_times}
                  duration={analysis.duration}
                />

                {/* Beat timestamps */}
                <div className="card animate-in" style={{
                  padding: 24,
                  maxHeight: 200,
                  overflow: "auto",
                  animationDelay: "500ms",
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
                    <Hash size={16} color="var(--accent-secondary)" />
                    <p className="section-label" style={{ marginBottom: 0 }}>
                      Beat Timestamps
                    </p>
                    <span style={{
                      marginLeft: "auto",
                      fontSize: "0.72rem",
                      color: "var(--text-dim)",
                    }}>
                      {analysis.beat_times.length} detected
                    </span>
                  </div>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                    {analysis.beat_times.map((t, i) => (
                      <span
                        key={i}
                        style={{
                          padding: "4px 8px",
                          background: i % 4 === 0 ? "rgba(139, 92, 246, 0.15)" : "var(--bg-secondary)",
                          border: `1px solid ${i % 4 === 0 ? "rgba(139, 92, 246, 0.25)" : "var(--border-color)"}`,
                          borderRadius: "var(--radius-xs)",
                          fontFamily: "var(--font-mono)",
                          fontSize: "0.7rem",
                          color: i % 4 === 0 ? "var(--accent-secondary)" : "var(--text-muted)",
                        }}
                      >
                        {t.toFixed(3)}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Empty state */}
            {!loading && !analysis && !error && (
              <div className="card empty-state">
                <div className="empty-state-icon">
                  <Activity size={28} color="var(--accent-primary)" />
                </div>
                <h3>Select a track</h3>
                <p>Choose a track from the list to begin AI analysis</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
