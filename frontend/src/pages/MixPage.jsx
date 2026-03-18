import { useState, useEffect, useRef } from "react";
import {
  Music, Headphones, Download, Play, Pause, Sliders,
  ArrowRight, Disc, Volume2, Clock, Activity
} from "lucide-react";
import { api } from "../api";
import WaveSurfer from "wavesurfer.js";

// Waveform output player
function WaveformOutput({ url }) {
  const containerRef = useRef(null);
  const wsRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [ready, setReady] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    if (!url || !containerRef.current) return;
    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: "rgba(139, 92, 246, 0.35)",
      progressColor: "rgba(236, 72, 153, 0.9)",
      cursorColor: "rgba(255, 255, 255, 0.6)",
      barWidth: 3,
      barGap: 2,
      barRadius: 3,
      height: 80,
      normalize: true,
      url,
    });
    ws.on("ready", () => {
      setReady(true);
      setDuration(ws.getDuration());
    });
    ws.on("play", () => setIsPlaying(true));
    ws.on("pause", () => setIsPlaying(false));
    ws.on("finish", () => setIsPlaying(false));
    ws.on("timeupdate", (time) => setCurrentTime(time));
    wsRef.current = ws;
    return () => {
      ws.destroy();
      wsRef.current = null;
      setIsPlaying(false);
      setReady(false);
    };
  }, [url]);

  const formatTime = (s) => {
    const mins = Math.floor(s / 60);
    const secs = Math.floor(s % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div>
      <div
        ref={containerRef}
        style={{
          borderRadius: "var(--radius-md)",
          overflow: "hidden",
        }}
      />
      <div style={{
        marginTop: 16,
        display: "flex",
        alignItems: "center",
        gap: 16,
      }}>
        <button
          className="btn btn-secondary"
          onClick={() => wsRef.current?.playPause()}
          disabled={!ready}
          style={{ minWidth: 100 }}
        >
          {isPlaying
            ? <><Pause size={14} /> Pause</>
            : <><Play size={14} /> Play</>
          }
        </button>
        <div style={{
          flex: 1,
          display: "flex",
          alignItems: "center",
          gap: 12,
        }}>
          <span style={{
            fontFamily: "var(--font-mono)",
            fontSize: "0.82rem",
            color: "var(--accent-secondary)",
            fontWeight: 600,
          }}>
            {formatTime(currentTime)}
          </span>
          <div style={{
            flex: 1,
            height: 4,
            background: "var(--bg-secondary)",
            borderRadius: 2,
          }}>
            <div style={{
              width: duration ? `${(currentTime / duration) * 100}%` : 0,
              height: "100%",
              background: "var(--accent-gradient)",
              borderRadius: 2,
              transition: "width 0.1s linear",
            }} />
          </div>
          <span style={{
            fontFamily: "var(--font-mono)",
            fontSize: "0.82rem",
            color: "var(--text-muted)",
          }}>
            {formatTime(duration)}
          </span>
        </div>
      </div>
    </div>
  );
}

// Deck card component
function DeckCard({ label, deckLetter, value, onChange, exclude, tracks }) {
  const gradientColors = deckLetter === "A"
    ? "linear-gradient(135deg, #7c3aed, #a855f7)"
    : "linear-gradient(135deg, #ec4899, #f43f5e)";

  return (
    <div className="mix-deck">
      {/* Header */}
      <div style={{
        display: "flex",
        alignItems: "center",
        gap: 12,
        marginBottom: 20,
      }}>
        <div style={{
          width: 36,
          height: 36,
          borderRadius: "var(--radius-md)",
          background: gradientColors,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontFamily: "var(--font-mono)",
          fontWeight: 800,
          fontSize: "1rem",
          color: "white",
          boxShadow: `0 4px 12px ${deckLetter === "A" ? "var(--accent-glow)" : "rgba(236, 72, 153, 0.35)"}`,
        }}>
          {deckLetter}
        </div>
        <div>
          <p className="section-label" style={{ marginBottom: 2 }}>{label}</p>
          <p style={{ fontSize: "0.72rem", color: "var(--text-dim)" }}>
            {deckLetter === "A" ? "Outgoing track" : "Incoming track"}
          </p>
        </div>
      </div>

      {value ? (
        /* Selected track */
        <div style={{
          background: `linear-gradient(135deg, ${deckLetter === "A" ? "rgba(124, 58, 237, 0.08)" : "rgba(236, 72, 153, 0.08)"}, transparent)`,
          border: `1px solid ${deckLetter === "A" ? "rgba(124, 58, 237, 0.25)" : "rgba(236, 72, 153, 0.25)"}`,
          borderRadius: "var(--radius-lg)",
          padding: 16,
        }}>
          <div style={{
            display: "flex",
            alignItems: "center",
            gap: 14,
          }}>
            <div style={{
              width: 48,
              height: 48,
              borderRadius: "var(--radius-md)",
              background: gradientColors,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexShrink: 0,
            }}>
              <Disc size={22} color="white" />
            </div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{
                fontWeight: 600,
                fontSize: "0.95rem",
                whiteSpace: "nowrap",
                overflow: "hidden",
                textOverflow: "ellipsis",
                marginBottom: 4,
              }}>
                {value.filename}
              </div>
              <div style={{
                display: "flex",
                gap: 12,
                fontSize: "0.78rem",
                color: "var(--text-muted)",
                fontFamily: "var(--font-mono)",
              }}>
                <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <Clock size={12} />
                  {value.duration}s
                </span>
                {value.bpm && (
                  <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                    <Activity size={12} />
                    {value.bpm.toFixed(1)} BPM
                  </span>
                )}
              </div>
            </div>
          </div>
          <button
            className="btn btn-ghost btn-sm"
            onClick={() => onChange(null)}
            style={{
              marginTop: 12,
              width: "100%",
              justifyContent: "center",
              color: "var(--text-muted)",
            }}
          >
            Clear Selection
          </button>
        </div>
      ) : (
        /* Track picker */
        tracks.filter(t => t.file_id !== exclude?.file_id).length === 0 ? (
          <div style={{
            textAlign: "center",
            padding: "40px 20px",
            color: "var(--text-muted)",
            fontSize: "0.88rem",
          }}>
            <Disc size={32} color="var(--text-dim)" style={{ marginBottom: 12 }} />
            <p>No tracks available</p>
          </div>
        ) : (
          <div className="track-list" style={{ maxHeight: 280, overflow: "auto" }}>
            {tracks
              .filter(t => t.file_id !== exclude?.file_id)
              .map((t, i) => (
                <div
                  key={t.file_id}
                  className="track-item"
                  onClick={() => onChange(t)}
                >
                  <div className="track-icon" style={{ background: gradientColors }}>
                    <Music size={16} color="white" />
                  </div>
                  <div className="track-info">
                    <div className="track-name">{t.filename}</div>
                    <div className="track-meta">
                      {t.duration}s{t.bpm ? ` · ${t.bpm.toFixed(1)} BPM` : ""}
                    </div>
                  </div>
                </div>
              ))}
          </div>
        )
      )}
    </div>
  );
}

// Main page
export default function MixPage({ tracks }) {
  const [trackA, setTrackA] = useState(null);
  const [trackB, setTrackB] = useState(null);
  const [crossfade, setCrossfade] = useState(5);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const canMix = trackA && trackB && !loading;

  const handleMix = async () => {
    if (!canMix) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await api.mixTracks(trackA.file_id, trackB.file_id, crossfade);
      setResult(res);
    } catch (err) {
      setError(err.message);
    }
    setLoading(false);
  };

  return (
    <div className="animate-in">
      <div className="page-header">
        <h2><span className="gradient-text">Mix</span> Tracks</h2>
        <p>
          Select two tracks and create a beat-synchronized DJ mix with automatic
          tempo matching and crossfade transitions.
        </p>
      </div>

      {tracks.length < 2 ? (
        <div className="card empty-state">
          <div className="empty-state-icon">
            <Sliders size={28} color="var(--accent-primary)" />
          </div>
          <h3>Not enough tracks</h3>
          <p>Upload at least 2 audio files to create a mix</p>
        </div>
      ) : (
        <>
          {/* Deck layout */}
          <div className="mix-layout">
            <DeckCard
              label="Deck A"
              deckLetter="A"
              value={trackA}
              onChange={setTrackA}
              exclude={trackB}
              tracks={tracks}
            />

            {/* Center controls */}
            <div className="mix-controls">
              {/* Connection indicator */}
              <div style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                marginBottom: 8,
              }}>
                <div style={{
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  background: trackA ? "var(--accent-primary)" : "var(--text-dim)",
                  boxShadow: trackA ? "0 0 8px var(--accent-primary)" : "none",
                }} />
                <ArrowRight size={16} color="var(--text-dim)" />
                <div style={{
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  background: trackB ? "#ec4899" : "var(--text-dim)",
                  boxShadow: trackB ? "0 0 8px #ec4899" : "none",
                }} />
              </div>

              <p className="crossfade-label">Crossfade</p>

              <input
                type="range"
                className="crossfade-slider"
                min="0.5"
                max="15"
                step="0.5"
                value={crossfade}
                onChange={e => setCrossfade(parseFloat(e.target.value))}
              />

              <span className="crossfade-value">{crossfade}s</span>

              {/* Volume icon */}
              <Volume2 size={18} color="var(--text-dim)" />

              <button
                className="btn btn-primary"
                disabled={!canMix}
                onClick={handleMix}
                style={{
                  marginTop: 12,
                  width: "100%",
                  padding: "14px 0",
                  fontSize: "0.9rem",
                }}
              >
                {loading
                  ? <><span className="spinner" style={{ width: 15, height: 15 }} /> Mixing...</>
                  : <><Sliders size={16} /> Create Mix</>
                }
              </button>
            </div>

            <DeckCard
              label="Deck B"
              deckLetter="B"
              value={trackB}
              onChange={setTrackB}
              exclude={trackA}
              tracks={tracks}
            />
          </div>

          {/* Error */}
          {error && (
            <div className="status status-error animate-in" style={{ marginTop: 24 }}>
              {error}
            </div>
          )}

          {/* Result */}
          {result && (
            <div className="animate-in" style={{ marginTop: 32 }}>
              <div className="card glow-border" style={{ padding: 28 }}>
                <div style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 12,
                  marginBottom: 24,
                }}>
                  <div style={{
                    width: 40,
                    height: 40,
                    borderRadius: "var(--radius-md)",
                    background: "var(--accent-gradient)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}>
                    <Headphones size={20} color="white" />
                  </div>
                  <div>
                    <p style={{ fontWeight: 700, fontSize: "1.1rem" }}>Mix Ready</p>
                    <p style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>
                      Beat-synchronized crossfade complete
                    </p>
                  </div>
                </div>

                {/* Stats */}
                <div className="analysis-grid" style={{ marginBottom: 24 }}>
                  <div className="stat-card">
                    <div className="stat-value gradient-text">{result.target_bpm.toFixed(1)}</div>
                    <div className="stat-label">Mix BPM</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-value" style={{ color: "var(--accent-secondary)" }}>
                      {result.duration.toFixed(1)}s
                    </div>
                    <div className="stat-label">Duration</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-value" style={{ color: "var(--success)" }}>
                      {result.bpm_a.toFixed(1)}
                    </div>
                    <div className="stat-label">Track A</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-value" style={{ color: "#ec4899" }}>
                      {result.bpm_b.toFixed(1)}
                    </div>
                    <div className="stat-label">Track B</div>
                  </div>
                </div>

                {/* Waveform */}
                <WaveformOutput url={api.getOutputUrl(result.output_file_id)} />

                {/* Download */}
                <div style={{ marginTop: 24 }}>
                  <a
                    className="btn btn-primary btn-lg"
                    href={api.getOutputUrl(result.output_file_id)}
                    download={`automix_${result.output_file_id}.wav`}
                    style={{ textDecoration: "none", width: "100%", justifyContent: "center" }}
                  >
                    <Download size={18} /> Download Mix (WAV)
                  </a>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
