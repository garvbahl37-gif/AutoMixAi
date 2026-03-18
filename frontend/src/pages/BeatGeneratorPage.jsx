import { useState, useEffect, useRef } from "react";
import {
  Wand2, Play, Pause, Download, Grid3x3, RefreshCw,
  Zap, Sparkles, Music, Hash, Clock, Layers
} from "lucide-react";
import WaveSurfer from "wavesurfer.js";
import { api } from "../api";

// Preset styles
const PRESETS = [
  { label: "Hip-Hop", prompt: "chill hip-hop beat with heavy bass and smooth hi-hats", color: "#a855f7" },
  { label: "Trap", prompt: "dark trap beat with rolling hi-hats and 808 kicks", color: "#ef4444" },
  { label: "House", prompt: "energetic house EDM beat at 128 BPM with big drops", color: "#06b6d4" },
  { label: "Rock", prompt: "driving rock beat with punchy snare and fast hi-hats", color: "#f59e0b" },
  { label: "Metal", prompt: "heavy metal beat with double kick and aggressive fills", color: "#dc2626" },
  { label: "Jazz", prompt: "laid-back jazz swing beat at 95 BPM", color: "#10b981" },
  { label: "Reggae", prompt: "one-drop reggae beat with off-beat rhythms at 80 BPM", color: "#84cc16" },
  { label: "Drum & Bass", prompt: "fast drum and bass beat at 174 BPM with complex breaks", color: "#f97316" },
  { label: "Ambient", prompt: "sparse ambient beat with slow atmospheric pulse", color: "#6366f1" },
  { label: "Afrobeats", prompt: "groovy afrobeats rhythm with syncopated patterns at 112 BPM", color: "#ec4899" },
  { label: "Funk", prompt: "funky groove beat with syncopated kick and tight snare", color: "#eab308" },
  { label: "Latin", prompt: "latin rhythm with clave-inspired pattern at 100 BPM", color: "#14b8a6" },
];

const BARS_OPTIONS = [2, 4, 8, 16];

const GENRE_COLORS = {
  hiphop: "#a855f7", trap: "#ef4444", edm: "#06b6d4",
  rock: "#f59e0b", metal: "#dc2626", jazz: "#10b981",
  reggae: "#84cc16", dnb: "#f97316", ambient: "#6366f1",
  afrobeats: "#ec4899", funk: "#eab308", latin: "#14b8a6",
  house: "#06b6d4",
};

// Waveform player
function BeatWaveform({ url }) {
  const ref = useRef(null);
  const wsRef = useRef(null);
  const [playing, setPlaying] = useState(false);
  const [ready, setReady] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    if (!url || !ref.current) return;
    const ws = WaveSurfer.create({
      container: ref.current,
      waveColor: "rgba(139, 92, 246, 0.4)",
      progressColor: "rgba(168, 85, 247, 0.9)",
      cursorColor: "rgba(255, 255, 255, 0.6)",
      barWidth: 3,
      barGap: 2,
      barRadius: 3,
      height: 80,
      normalize: true,
    });
    ws.load(url);
    ws.on("ready", () => {
      setReady(true);
      setDuration(ws.getDuration());
    });
    ws.on("play", () => setPlaying(true));
    ws.on("pause", () => setPlaying(false));
    ws.on("finish", () => setPlaying(false));
    ws.on("timeupdate", (time) => setCurrentTime(time));
    wsRef.current = ws;
    return () => {
      ws.destroy();
      wsRef.current = null;
      setPlaying(false);
      setReady(false);
    };
  }, [url]);

  const formatTime = (s) => {
    const secs = Math.floor(s);
    const ms = Math.floor((s - secs) * 100);
    return `${secs}.${ms.toString().padStart(2, '0')}`;
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div
        ref={ref}
        style={{ borderRadius: "var(--radius-md)", overflow: "hidden" }}
      />
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <button
          className="btn btn-secondary"
          onClick={() => wsRef.current?.playPause()}
          disabled={!ready}
          style={{ minWidth: 100 }}
        >
          {playing ? <><Pause size={14} /> Pause</> : <><Play size={14} /> Play</>}
        </button>
        <div style={{
          flex: 1,
          display: "flex",
          alignItems: "center",
          gap: 8,
        }}>
          <span style={{
            fontFamily: "var(--font-mono)",
            fontSize: "0.78rem",
            color: "var(--accent-secondary)",
            fontWeight: 600,
          }}>
            {formatTime(currentTime)}s
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
            fontSize: "0.78rem",
            color: "var(--text-muted)",
          }}>
            {formatTime(duration)}s
          </span>
        </div>
      </div>
    </div>
  );
}

// Drum pattern grid
const INSTRUMENTS = [
  { key: "kick", label: "Kick", color: "#7c3aed" },
  { key: "snare", label: "Snare", color: "#ec4899" },
  { key: "hihat_c", label: "HH–C", color: "#06b6d4" },
  { key: "hihat_o", label: "HH–O", color: "#0e7490" },
  { key: "clap", label: "Clap", color: "#f59e0b" },
];

function DrumPatternGrid({ pattern }) {
  if (!pattern) return null;

  return (
    <div style={{ overflowX: "auto" }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 6, minWidth: 520 }}>
        {/* Beat index row */}
        <div style={{ display: "flex", paddingLeft: 64 }}>
          {Array.from({ length: 16 }, (_, i) => (
            <div
              key={i}
              style={{
                flex: 1,
                textAlign: "center",
                fontSize: "0.68rem",
                color: i % 4 === 0 ? "var(--accent-secondary)" : "var(--text-dim)",
                fontFamily: "var(--font-mono)",
                fontWeight: i % 4 === 0 ? 700 : 400,
                paddingBottom: 6,
              }}
            >
              {i % 4 === 0 ? i / 4 + 1 : "·"}
            </div>
          ))}
        </div>
        {/* Instrument rows */}
        {INSTRUMENTS.map(({ key, label, color }) => {
          const steps = pattern[key] || Array(16).fill(0);
          return (
            <div key={key} style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{
                width: 58,
                flexShrink: 0,
                fontSize: "0.72rem",
                letterSpacing: "0.04em",
                color: "var(--text-muted)",
                textAlign: "right",
                paddingRight: 8,
                fontFamily: "var(--font-mono)",
                textTransform: "uppercase",
                fontWeight: 500,
              }}>
                {label}
              </span>
              {steps.map((hit, i) => (
                <div
                  key={i}
                  style={{
                    flex: 1,
                    height: 24,
                    borderRadius: 4,
                    background: hit
                      ? `linear-gradient(135deg, ${color}, ${color}cc)`
                      : i % 4 === 0
                        ? "rgba(255,255,255,0.05)"
                        : "rgba(255,255,255,0.02)",
                    border: `1px solid ${hit ? `${color}aa` : i % 4 === 0 ? "rgba(255,255,255,0.08)" : "rgba(255,255,255,0.03)"}`,
                    boxShadow: hit ? `0 0 10px ${color}50, inset 0 1px 0 rgba(255,255,255,0.2)` : "none",
                    transition: "all 0.15s ease",
                  }}
                />
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// Genre chip
function GenreChip({ genre, confidence }) {
  const color = GENRE_COLORS[genre] || "var(--accent-primary)";
  const pct = Math.round((confidence ?? 1) * 100);
  const name = genre
    .replace("hiphop", "Hip-Hop")
    .replace("dnb", "Drum & Bass")
    .replace(/\b\w/g, c => c.toUpperCase());

  return (
    <div style={{
      display: "inline-flex",
      alignItems: "center",
      gap: 10,
      background: `${color}15`,
      border: `1px solid ${color}40`,
      borderRadius: "var(--radius-md)",
      padding: "8px 16px",
    }}>
      <span style={{
        color,
        fontSize: "0.88rem",
        fontWeight: 700,
        letterSpacing: "0.03em",
      }}>
        {name}
      </span>
      <div style={{
        width: 60,
        height: 5,
        borderRadius: 3,
        background: "rgba(255,255,255,0.08)",
        overflow: "hidden",
      }}>
        <div style={{
          width: `${pct}%`,
          height: "100%",
          background: color,
          borderRadius: 3,
        }} />
      </div>
      <span style={{
        color: "var(--text-muted)",
        fontSize: "0.75rem",
        fontFamily: "var(--font-mono)",
        fontWeight: 600,
      }}>
        {pct}%
      </span>
    </div>
  );
}

// Main page
export default function BeatGeneratorPage() {
  const [prompt, setPrompt] = useState("");
  const [bars, setBars] = useState(4);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [outputUrl, setOutputUrl] = useState(null);
  const [activePreset, setActivePreset] = useState(null);

  const canGenerate = prompt.trim().length >= 3 && !loading;

  const applyPreset = (p) => {
    setPrompt(p.prompt);
    setActivePreset(p.label);
  };

  const handleGenerate = async () => {
    if (!canGenerate) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setOutputUrl(null);
    try {
      const data = await api.generateBeat(prompt, bars);
      setResult(data);
      setOutputUrl(api.getOutputUrl(data.output_file_id));
    } catch (err) {
      setError(err.message);
    }
    setLoading(false);
  };

  return (
    <div className="animate-in">
      {/* Header */}
      <div className="page-header">
        <h2><span className="gradient-text">Beat</span> Generator</h2>
        <p>
          Describe your beat in natural language — tempo, genre, style, mood —
          and let AI synthesize a custom drum pattern.
        </p>
      </div>

      <div style={{
        display: "grid",
        gridTemplateColumns: "1fr 360px",
        gap: 28,
        alignItems: "start",
      }}>
        {/* Left column: input */}
        <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
          {/* Prompt textarea */}
          <div className="card" style={{ padding: 24 }}>
            <div style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
              marginBottom: 16,
            }}>
              <Wand2 size={18} color="var(--accent-secondary)" />
              <p className="section-label" style={{ marginBottom: 0 }}>Prompt</p>
            </div>
            <textarea
              value={prompt}
              onChange={e => {
                setPrompt(e.target.value);
                setActivePreset(null);
              }}
              onKeyDown={e => e.key === "Enter" && (e.metaKey || e.ctrlKey) && handleGenerate()}
              placeholder={
                "e.g. chill hip-hop beat at 90 BPM with heavy bass\n" +
                "e.g. dark trap banger with 808s and fast hi-hat rolls\n\n" +
                "Tip: Press ⌘/Ctrl + Enter to generate"
              }
              rows={5}
              maxLength={500}
              className="textarea"
              style={{
                lineHeight: 1.8,
                fontSize: "0.92rem",
              }}
            />
            <div style={{
              marginTop: 8,
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}>
              <span style={{
                fontSize: "0.72rem",
                color: "var(--text-dim)",
              }}>
                Supports 20+ genres, mood, complexity, time signatures
              </span>
              <span style={{
                fontSize: "0.72rem",
                color: "var(--text-muted)",
                fontFamily: "var(--font-mono)",
              }}>
                {prompt.length}/500
              </span>
            </div>
          </div>

          {/* Style presets */}
          <div className="card" style={{ padding: 24 }}>
            <div style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
              marginBottom: 16,
            }}>
              <Sparkles size={18} color="var(--accent-secondary)" />
              <p className="section-label" style={{ marginBottom: 0 }}>Quick Styles</p>
            </div>
            <div style={{
              display: "grid",
              gridTemplateColumns: "repeat(4, 1fr)",
              gap: 10,
            }}>
              {PRESETS.map(p => {
                const active = activePreset === p.label;
                return (
                  <button
                    key={p.label}
                    onClick={() => applyPreset(p)}
                    style={{
                      background: active ? `${p.color}20` : "var(--bg-secondary)",
                      border: `1px solid ${active ? `${p.color}60` : "var(--border-color)"}`,
                      color: active ? p.color : "var(--text-secondary)",
                      borderRadius: "var(--radius-md)",
                      padding: "10px 12px",
                      fontSize: "0.82rem",
                      fontWeight: 600,
                      cursor: "pointer",
                      fontFamily: "var(--font-sans)",
                      transition: "all 0.15s ease",
                      letterSpacing: "0.01em",
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                    }}
                    onMouseEnter={e => {
                      if (!active) {
                        e.target.style.borderColor = `${p.color}40`;
                        e.target.style.color = p.color;
                      }
                    }}
                    onMouseLeave={e => {
                      if (!active) {
                        e.target.style.borderColor = "var(--border-color)";
                        e.target.style.color = "var(--text-secondary)";
                      }
                    }}
                  >
                    {p.label}
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right column: controls */}
        <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
          {/* Bars */}
          <div className="card" style={{ padding: 24 }}>
            <div style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
              marginBottom: 16,
            }}>
              <Hash size={18} color="var(--accent-secondary)" />
              <p className="section-label" style={{ marginBottom: 0 }}>Bars</p>
            </div>
            <div style={{
              display: "grid",
              gridTemplateColumns: "repeat(4, 1fr)",
              gap: 10,
            }}>
              {BARS_OPTIONS.map(b => (
                <button
                  key={b}
                  onClick={() => setBars(b)}
                  style={{
                    padding: "14px 0",
                    borderRadius: "var(--radius-md)",
                    border: `1px solid ${bars === b ? "rgba(139, 92, 246, 0.6)" : "var(--border-color)"}`,
                    background: bars === b ? "rgba(139, 92, 246, 0.15)" : "var(--bg-secondary)",
                    color: bars === b ? "var(--accent-secondary)" : "var(--text-secondary)",
                    cursor: "pointer",
                    fontWeight: 700,
                    fontSize: "1rem",
                    fontFamily: "var(--font-mono)",
                    transition: "all 0.15s ease",
                    boxShadow: bars === b ? "0 0 12px rgba(139, 92, 246, 0.2)" : "none",
                  }}
                >
                  {b}
                </button>
              ))}
            </div>
          </div>

          {/* Generate button */}
          <button
            className="btn btn-primary btn-lg glow-pulse"
            onClick={handleGenerate}
            disabled={!canGenerate}
            style={{
              width: "100%",
              padding: "16px 0",
              fontSize: "1rem",
              letterSpacing: "0.02em",
            }}
          >
            {loading
              ? <><div className="spinner" style={{ width: 18, height: 18 }} /> Generating...</>
              : <><Wand2 size={18} /> Generate Beat</>
            }
          </button>

          {/* Result meta */}
          {result && !loading && (
            <div className="card glow-border animate-in" style={{ padding: 24 }}>
              <GenreChip genre={result.genre} confidence={result.genre_confidence ?? 1} />

              <div style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 12,
                marginTop: 20,
              }}>
                {[
                  { label: "BPM", value: result.bpm, icon: Music, mono: true },
                  { label: "Bars", value: result.bars, icon: Hash, mono: true },
                  { label: "Duration", value: `${result.duration.toFixed(1)}s`, icon: Clock, mono: false },
                  { label: "Complexity", value: result.complexity, icon: Layers, mono: false },
                ].map(({ label, value, icon: Icon, mono }) => (
                  <div
                    key={label}
                    style={{
                      background: "var(--bg-secondary)",
                      borderRadius: "var(--radius-md)",
                      border: "1px solid var(--border-color)",
                      padding: "14px 16px",
                      position: "relative",
                    }}
                  >
                    <Icon
                      size={14}
                      color="var(--text-dim)"
                      style={{
                        position: "absolute",
                        top: 12,
                        right: 12,
                        opacity: 0.5,
                      }}
                    />
                    <div style={{
                      fontSize: "1.2rem",
                      fontWeight: 800,
                      fontFamily: mono ? "var(--font-mono)" : "var(--font-sans)",
                      color: "var(--text-primary)",
                      textTransform: "capitalize",
                    }}>
                      {value}
                    </div>
                    <div style={{
                      fontSize: "0.7rem",
                      letterSpacing: "0.1em",
                      textTransform: "uppercase",
                      color: "var(--text-muted)",
                      marginTop: 4,
                      fontWeight: 600,
                    }}>
                      {label}
                    </div>
                  </div>
                ))}
              </div>

              {outputUrl && (
                <a
                  href={outputUrl}
                  download={`automix_beat_${result.genre}_${result.bpm}bpm.wav`}
                  className="btn btn-secondary"
                  style={{
                    marginTop: 16,
                    width: "100%",
                    justifyContent: "center",
                    textDecoration: "none",
                  }}
                >
                  <Download size={14} /> Download WAV
                </a>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="status status-error animate-in" style={{ marginTop: 24 }}>
          {error}
        </div>
      )}

      {/* Player + Pattern */}
      {result && !loading && (
        <div className="animate-in" style={{
          marginTop: 32,
          display: "flex",
          flexDirection: "column",
          gap: 20,
        }}>
          {/* Waveform */}
          <div className="card" style={{ padding: 24 }}>
            <div style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: 20,
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <Music size={16} color="var(--accent-secondary)" />
                <p className="section-label" style={{ marginBottom: 0 }}>Waveform</p>
              </div>
              <button
                className="btn btn-ghost btn-sm"
                onClick={handleGenerate}
              >
                <RefreshCw size={13} /> Regenerate
              </button>
            </div>
            <BeatWaveform url={outputUrl} key={outputUrl} />
          </div>

          {/* Drum pattern */}
          <div className="card" style={{ padding: 24 }}>
            <div style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
              marginBottom: 20,
            }}>
              <Grid3x3 size={16} color="var(--accent-secondary)" />
              <p className="section-label" style={{ marginBottom: 0 }}>
                Drum Pattern — 16 Steps
              </p>
            </div>
            <DrumPatternGrid pattern={result.pattern} />
          </div>
        </div>
      )}

      {/* Empty state */}
      {!result && !loading && !error && (
        <div className="card empty-state" style={{ marginTop: 32 }}>
          <div className="empty-state-icon glow-pulse">
            <Zap size={32} color="var(--accent-primary)" />
          </div>
          <h3>Ready to Create</h3>
          <p>Describe your beat using natural language and click Generate</p>
          <div style={{
            marginTop: 16,
            display: "flex",
            gap: 8,
            justifyContent: "center",
            flexWrap: "wrap",
          }}>
            <span className="tag tag-sm">20+ genres</span>
            <span className="tag tag-sm">3 complexity levels</span>
            <span className="tag tag-sm">Pure synthesis</span>
          </div>
        </div>
      )}
    </div>
  );
}
