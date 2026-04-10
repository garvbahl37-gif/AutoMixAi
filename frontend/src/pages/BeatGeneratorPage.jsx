import { useState, useEffect, useRef } from "react";
import {
  Wand2, Play, Pause, Download, Grid3x3, RefreshCw,
  Zap, Sparkles, Music, Hash, Clock, Layers, Cpu, Brain,
  Sliders, Thermometer, Target
} from "lucide-react";
import WaveSurfer from "wavesurfer.js";
import { api } from "../api";

// ─── Preset styles ──────────────────────────────────────────────────────────

const AI_PRESETS = [
  { label: "Trap Beat", prompt: "hard-hitting trap beat with 808 bass, rolling hi-hats, and dark melody", color: "#ef4444" },
  { label: "Lo-Fi Chill", prompt: "smooth lo-fi hip hop beat with vinyl crackle, mellow piano, and soft drums", color: "#a855f7" },
  { label: "EDM Drop", prompt: "energetic EDM beat with massive synth drop, punchy kick, and sidechained bass", color: "#06b6d4" },
  { label: "Drill Beat", prompt: "UK drill beat with sliding 808 bass, fast hi-hats, and dark piano melody", color: "#dc2626" },
  { label: "Boom Bap", prompt: "classic boom bap hip hop beat with heavy kick, crispy snare, and jazz sample", color: "#f59e0b" },
  { label: "House Groove", prompt: "deep house beat with four on the floor kick, funky bassline, and vocal chops", color: "#10b981" },
  { label: "Afrobeat", prompt: "afrobeat rhythm with percussion, groovy bass, and bright melody at 110 BPM", color: "#ec4899" },
  { label: "Ambient Pad", prompt: "atmospheric ambient beat with ethereal pads, subtle percussion, and reverb", color: "#6366f1" },
  { label: "Rock Drums", prompt: "powerful rock drum beat with heavy kick, crashing cymbals, and driving rhythm", color: "#f97316" },
  { label: "R&B Smooth", prompt: "smooth R&B beat with soft 808s, snapping snare, and warm chords", color: "#14b8a6" },
  { label: "Reggaeton", prompt: "reggaeton dembow beat with bouncy rhythm, percussive hits at 95 BPM", color: "#eab308" },
  { label: "Phonk", prompt: "dark phonk beat with distorted 808, cowbell, and aggressive Memphis style", color: "#7c3aed" },
];

const SYNTH_PRESETS = [
  { label: "Hip-Hop", prompt: "chill hip-hop beat with heavy bass and smooth hi-hats", color: "#a855f7" },
  { label: "Trap", prompt: "dark trap beat with rolling hi-hats and 808 kicks", color: "#ef4444" },
  { label: "House", prompt: "energetic house EDM beat at 128 BPM with big drops", color: "#06b6d4" },
  { label: "Rock", prompt: "driving rock beat with punchy snare and fast hi-hats", color: "#f59e0b" },
  { label: "Metal", prompt: "heavy metal beat with double kick and aggressive fills", color: "#dc2626" },
  { label: "Jazz", prompt: "laid-back jazz swing beat at 95 BPM", color: "#10b981" },
  { label: "Reggae", prompt: "one-drop reggae beat with off-beat rhythms at 80 BPM", color: "#84cc16" },
  { label: "Drum & Bass", prompt: "fast drum and bass beat at 174 BPM with complex breaks", color: "#f97316" },
];

const BARS_OPTIONS = [2, 4, 8, 16];
const DURATION_OPTIONS = [5, 10, 15, 20, 30];

const GENRE_COLORS = {
  hiphop: "#a855f7", trap: "#ef4444", edm: "#06b6d4",
  rock: "#f59e0b", metal: "#dc2626", jazz: "#10b981",
  reggae: "#84cc16", dnb: "#f97316", ambient: "#6366f1",
  afrobeats: "#ec4899", funk: "#eab308", latin: "#14b8a6",
  house: "#06b6d4",
};

// ─── Waveform player ────────────────────────────────────────────────────────

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
        <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{
            fontFamily: "var(--font-mono)",
            fontSize: "0.78rem",
            color: "var(--accent-secondary)",
            fontWeight: 600,
          }}>
            {formatTime(currentTime)}s
          </span>
          <div style={{
            flex: 1, height: 4,
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

// ─── Drum pattern grid ──────────────────────────────────────────────────────

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
        <div style={{ display: "flex", paddingLeft: 64 }}>
          {Array.from({ length: 16 }, (_, i) => (
            <div key={i} style={{
              flex: 1, textAlign: "center",
              fontSize: "0.68rem",
              color: i % 4 === 0 ? "var(--accent-secondary)" : "var(--text-dim)",
              fontFamily: "var(--font-mono)",
              fontWeight: i % 4 === 0 ? 700 : 400,
              paddingBottom: 6,
            }}>
              {i % 4 === 0 ? i / 4 + 1 : "·"}
            </div>
          ))}
        </div>
        {INSTRUMENTS.map(({ key, label, color }) => {
          const steps = pattern[key] || Array(16).fill(0);
          return (
            <div key={key} style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{
                width: 58, flexShrink: 0, fontSize: "0.72rem",
                letterSpacing: "0.04em", color: "var(--text-muted)",
                textAlign: "right", paddingRight: 8,
                fontFamily: "var(--font-mono)", textTransform: "uppercase",
                fontWeight: 500,
              }}>{label}</span>
              {steps.map((hit, i) => (
                <div key={i} style={{
                  flex: 1, height: 24, borderRadius: 4,
                  background: hit
                    ? `linear-gradient(135deg, ${color}, ${color}cc)`
                    : i % 4 === 0 ? "rgba(255,255,255,0.05)" : "rgba(255,255,255,0.02)",
                  border: `1px solid ${hit ? `${color}aa` : i % 4 === 0 ? "rgba(255,255,255,0.08)" : "rgba(255,255,255,0.03)"}`,
                  boxShadow: hit ? `0 0 10px ${color}50, inset 0 1px 0 rgba(255,255,255,0.2)` : "none",
                  transition: "all 0.15s ease",
                }} />
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── Slider component ───────────────────────────────────────────────────────

function SliderControl({ label, value, onChange, min, max, step, icon: Icon, unit = "", color = "var(--accent-primary)" }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{
        display: "flex", justifyContent: "space-between",
        alignItems: "center", marginBottom: 8,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {Icon && <Icon size={14} color={color} />}
          <span style={{ fontSize: "0.78rem", color: "var(--text-secondary)", fontWeight: 600 }}>
            {label}
          </span>
        </div>
        <span style={{
          fontFamily: "var(--font-mono)", fontSize: "0.78rem",
          fontWeight: 700, color,
        }}>
          {value}{unit}
        </span>
      </div>
      <input
        type="range"
        min={min} max={max} step={step}
        value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        className="slider"
        style={{ width: "100%", accentColor: color }}
      />
    </div>
  );
}

// ─── Mode toggle ────────────────────────────────────────────────────────────

function ModeToggle({ mode, setMode }) {
  return (
    <div style={{
      display: "flex", gap: 4,
      background: "var(--bg-secondary)",
      borderRadius: "var(--radius-md)",
      padding: 4,
      border: "1px solid var(--border-color)",
    }}>
      {[
        { key: "ai", label: "AI MusicGen", icon: Brain, desc: "Deep learning" },
        { key: "synth", label: "Drum Synth", icon: Cpu, desc: "Procedural" },
      ].map(({ key, label, icon: Icon }) => (
        <button
          key={key}
          onClick={() => setMode(key)}
          style={{
            flex: 1, display: "flex", alignItems: "center",
            justifyContent: "center", gap: 8,
            padding: "12px 16px",
            borderRadius: "var(--radius-sm)",
            border: "none",
            background: mode === key
              ? "linear-gradient(135deg, rgba(139,92,246,0.2), rgba(168,85,247,0.15))"
              : "transparent",
            color: mode === key ? "var(--accent-secondary)" : "var(--text-muted)",
            cursor: "pointer",
            fontSize: "0.85rem",
            fontWeight: mode === key ? 700 : 500,
            fontFamily: "var(--font-sans)",
            transition: "all 0.2s ease",
            boxShadow: mode === key ? "0 0 12px rgba(139,92,246,0.15)" : "none",
          }}
        >
          <Icon size={16} />
          {label}
        </button>
      ))}
    </div>
  );
}

// ─── Main page ──────────────────────────────────────────────────────────────

export default function BeatGeneratorPage() {
  const [mode, setMode] = useState("ai"); // "ai" | "synth"
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [outputUrl, setOutputUrl] = useState(null);
  const [activePreset, setActivePreset] = useState(null);

  // AI mode controls
  const [duration, setDuration] = useState(10);
  const [temperature, setTemperature] = useState(1.0);
  const [guidanceScale, setGuidanceScale] = useState(3.0);

  // Synth mode controls
  const [bars, setBars] = useState(4);

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
      if (mode === "ai") {
        const data = await api.generateBeatAI(prompt, duration, temperature, guidanceScale);
        setResult({ ...data, mode: "ai" });
        setOutputUrl(api.getBeatOutputUrl(data.output_file_id));
      } else {
        const data = await api.generateBeat(prompt, bars);
        setResult({ ...data, mode: "synth" });
        setOutputUrl(api.getOutputUrl(data.output_file_id));
      }
    } catch (err) {
      setError(err.message);
    }
    setLoading(false);
  };

  const presets = mode === "ai" ? AI_PRESETS : SYNTH_PRESETS;

  return (
    <div className="animate-in">
      {/* Header */}
      <div className="page-header">
        <h2><span className="gradient-text">Beat</span> Generator</h2>
        <p>
          {mode === "ai"
            ? "Generate studio-quality beats and music using Meta's MusicGen AI model."
            : "Describe your drum pattern and let procedural synthesis create it instantly."}
        </p>
      </div>

      {/* Mode Toggle */}
      <div style={{ marginBottom: 24 }}>
        <ModeToggle mode={mode} setMode={(m) => {
          setMode(m);
          setResult(null);
          setOutputUrl(null);
          setError(null);
          setActivePreset(null);
          setPrompt("");
        }} />
      </div>

      {/* AI Mode Badge */}
      {mode === "ai" && (
        <div className="animate-in" style={{
          display: "flex", alignItems: "center", gap: 10,
          padding: "10px 16px", marginBottom: 20,
          background: "linear-gradient(90deg, rgba(139,92,246,0.08), rgba(236,72,153,0.06))",
          border: "1px solid rgba(139,92,246,0.2)",
          borderRadius: "var(--radius-md)",
        }}>
          <Brain size={16} color="var(--accent-primary)" />
          <span style={{ fontSize: "0.82rem", color: "var(--text-secondary)" }}>
            Powered by <strong style={{ color: "var(--accent-secondary)" }}>Meta MusicGen</strong> —
            generates real melodies, bass, drums, and full beats from text
          </span>
        </div>
      )}

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
              display: "flex", alignItems: "center", gap: 10, marginBottom: 16,
            }}>
              <Wand2 size={18} color="var(--accent-secondary)" />
              <p className="section-label" style={{ marginBottom: 0 }}>Prompt</p>
            </div>
            <textarea
              value={prompt}
              onChange={e => { setPrompt(e.target.value); setActivePreset(null); }}
              onKeyDown={e => e.key === "Enter" && (e.metaKey || e.ctrlKey) && handleGenerate()}
              placeholder={mode === "ai"
                ? "e.g. hard-hitting trap beat with 808 bass and dark melody\ne.g. smooth lo-fi beat with vinyl crackle and piano\n\nTip: Press ⌘/Ctrl + Enter to generate"
                : "e.g. chill hip-hop beat at 90 BPM with heavy bass\ne.g. dark trap banger with 808s and fast hi-hat rolls\n\nTip: Press ⌘/Ctrl + Enter to generate"
              }
              rows={5}
              maxLength={500}
              className="textarea"
              style={{ lineHeight: 1.8, fontSize: "0.92rem" }}
            />
            <div style={{
              marginTop: 8, display: "flex",
              justifyContent: "space-between", alignItems: "center",
            }}>
              <span style={{ fontSize: "0.72rem", color: "var(--text-dim)" }}>
                {mode === "ai"
                  ? "Describe any style: melody, bass, drums, instruments, mood"
                  : "Supports 20+ genres, mood, complexity, time signatures"
                }
              </span>
              <span style={{
                fontSize: "0.72rem", color: "var(--text-muted)",
                fontFamily: "var(--font-mono)",
              }}>
                {prompt.length}/500
              </span>
            </div>
          </div>

          {/* Style presets */}
          <div className="card" style={{ padding: 24 }}>
            <div style={{
              display: "flex", alignItems: "center", gap: 10, marginBottom: 16,
            }}>
              <Sparkles size={18} color="var(--accent-secondary)" />
              <p className="section-label" style={{ marginBottom: 0 }}>Quick Styles</p>
            </div>
            <div style={{
              display: "grid",
              gridTemplateColumns: "repeat(4, 1fr)",
              gap: 10,
            }}>
              {presets.map(p => {
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
          {mode === "ai" ? (
            <>
              {/* AI Controls */}
              <div className="card" style={{ padding: 24 }}>
                <div style={{
                  display: "flex", alignItems: "center", gap: 10, marginBottom: 20,
                }}>
                  <Sliders size={18} color="var(--accent-secondary)" />
                  <p className="section-label" style={{ marginBottom: 0 }}>AI Controls</p>
                </div>

                <SliderControl
                  label="Duration" value={duration} onChange={setDuration}
                  min={3} max={30} step={1} unit="s"
                  icon={Clock} color="#10b981"
                />
                <SliderControl
                  label="Temperature" value={temperature} onChange={setTemperature}
                  min={0.5} max={1.5} step={0.1}
                  icon={Thermometer} color="#f59e0b"
                />
                <SliderControl
                  label="Guidance" value={guidanceScale} onChange={setGuidanceScale}
                  min={1.0} max={10.0} step={0.5}
                  icon={Target} color="#8b5cf6"
                />

                <div style={{
                  marginTop: 8, padding: "10px 14px",
                  background: "rgba(255,255,255,0.02)",
                  borderRadius: "var(--radius-sm)",
                  border: "1px solid var(--border-color)",
                }}>
                  <p style={{ fontSize: "0.72rem", color: "var(--text-dim)", lineHeight: 1.6 }}>
                    <strong>Temperature:</strong> Low = predictable, High = creative<br />
                    <strong>Guidance:</strong> Low = free, High = strictly follows prompt
                  </p>
                </div>
              </div>
            </>
          ) : (
            <>
              {/* Synth Controls */}
              <div className="card" style={{ padding: 24 }}>
                <div style={{
                  display: "flex", alignItems: "center", gap: 10, marginBottom: 16,
                }}>
                  <Hash size={18} color="var(--accent-secondary)" />
                  <p className="section-label" style={{ marginBottom: 0 }}>Bars</p>
                </div>
                <div style={{
                  display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10,
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
            </>
          )}

          {/* Generate button */}
          <button
            className="btn btn-primary btn-lg glow-pulse"
            onClick={handleGenerate}
            disabled={!canGenerate}
            style={{
              width: "100%", padding: "16px 0",
              fontSize: "1rem", letterSpacing: "0.02em",
            }}
          >
            {loading
              ? <><div className="spinner" style={{ width: 18, height: 18 }} /> {mode === "ai" ? "Generating with AI..." : "Generating..."}</>
              : <><Wand2 size={18} /> {mode === "ai" ? "Generate with MusicGen" : "Generate Beat"}</>
            }
          </button>

          {/* Result meta */}
          {result && !loading && (
            <div className="card glow-border animate-in" style={{ padding: 24 }}>
              {/* Mode badge */}
              <div style={{
                display: "inline-flex", alignItems: "center", gap: 8,
                padding: "6px 14px", marginBottom: 16,
                background: result.mode === "ai"
                  ? "rgba(139,92,246,0.12)" : "rgba(6,182,212,0.12)",
                border: `1px solid ${result.mode === "ai"
                  ? "rgba(139,92,246,0.3)" : "rgba(6,182,212,0.3)"}`,
                borderRadius: "var(--radius-full)",
                fontSize: "0.78rem",
                fontWeight: 700,
                color: result.mode === "ai" ? "#a855f7" : "#06b6d4",
              }}>
                {result.mode === "ai" ? <Brain size={14} /> : <Cpu size={14} />}
                {result.mode === "ai" ? "MusicGen AI" : "Drum Synthesis"}
              </div>

              {/* Stats */}
              <div style={{
                display: "grid",
                gridTemplateColumns: result.mode === "ai" ? "1fr 1fr" : "1fr 1fr",
                gap: 12,
              }}>
                {result.mode === "ai" ? (
                  <>
                    {[
                      { label: "Duration", value: `${result.duration}s`, icon: Clock },
                      { label: "Model", value: result.model?.split("/").pop() || "musicgen", icon: Brain },
                      { label: "Sample Rate", value: `${(result.sample_rate / 1000).toFixed(0)}kHz`, icon: Music },
                    ].map(({ label, value, icon: Icon }) => (
                      <div key={label} style={{
                        background: "var(--bg-secondary)",
                        borderRadius: "var(--radius-md)",
                        border: "1px solid var(--border-color)",
                        padding: "14px 16px",
                        position: "relative",
                      }}>
                        <Icon size={14} color="var(--text-dim)" style={{
                          position: "absolute", top: 12, right: 12, opacity: 0.5,
                        }} />
                        <div style={{
                          fontSize: "1.1rem", fontWeight: 800,
                          fontFamily: "var(--font-mono)",
                          color: "var(--text-primary)",
                          textTransform: "capitalize",
                        }}>{value}</div>
                        <div style={{
                          fontSize: "0.7rem", letterSpacing: "0.1em",
                          textTransform: "uppercase", color: "var(--text-muted)",
                          marginTop: 4, fontWeight: 600,
                        }}>{label}</div>
                      </div>
                    ))}
                  </>
                ) : (
                  <>
                    {[
                      { label: "BPM", value: result.bpm, icon: Music },
                      { label: "Bars", value: result.bars, icon: Hash },
                      { label: "Duration", value: `${result.duration?.toFixed(1)}s`, icon: Clock },
                      { label: "Complexity", value: result.complexity, icon: Layers },
                    ].map(({ label, value, icon: Icon }) => (
                      <div key={label} style={{
                        background: "var(--bg-secondary)",
                        borderRadius: "var(--radius-md)",
                        border: "1px solid var(--border-color)",
                        padding: "14px 16px",
                        position: "relative",
                      }}>
                        <Icon size={14} color="var(--text-dim)" style={{
                          position: "absolute", top: 12, right: 12, opacity: 0.5,
                        }} />
                        <div style={{
                          fontSize: "1.2rem", fontWeight: 800,
                          fontFamily: "var(--font-mono)",
                          color: "var(--text-primary)",
                          textTransform: "capitalize",
                        }}>{value}</div>
                        <div style={{
                          fontSize: "0.7rem", letterSpacing: "0.1em",
                          textTransform: "uppercase", color: "var(--text-muted)",
                          marginTop: 4, fontWeight: 600,
                        }}>{label}</div>
                      </div>
                    ))}
                  </>
                )}
              </div>

              {outputUrl && (
                <a
                  href={outputUrl}
                  download={`automix_beat_${result.mode === "ai" ? "ai" : result.genre}.wav`}
                  className="btn btn-secondary"
                  style={{
                    marginTop: 16, width: "100%",
                    justifyContent: "center", textDecoration: "none",
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
          marginTop: 32, display: "flex", flexDirection: "column", gap: 20,
        }}>
          {/* Waveform */}
          <div className="card" style={{ padding: 24 }}>
            <div style={{
              display: "flex", justifyContent: "space-between",
              alignItems: "center", marginBottom: 20,
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <Music size={16} color="var(--accent-secondary)" />
                <p className="section-label" style={{ marginBottom: 0 }}>Waveform</p>
                {result.mode === "ai" && (
                  <span style={{
                    padding: "3px 10px",
                    background: "rgba(139,92,246,0.1)",
                    border: "1px solid rgba(139,92,246,0.25)",
                    borderRadius: "var(--radius-full)",
                    fontSize: "0.68rem",
                    fontWeight: 700,
                    color: "#a855f7",
                  }}>
                    AI Generated
                  </span>
                )}
              </div>
              <button className="btn btn-ghost btn-sm" onClick={handleGenerate}>
                <RefreshCw size={13} /> Regenerate
              </button>
            </div>
            <BeatWaveform url={outputUrl} key={outputUrl} />
          </div>

          {/* Drum pattern (synth mode only) */}
          {result.mode === "synth" && result.pattern && (
            <div className="card" style={{ padding: 24 }}>
              <div style={{
                display: "flex", alignItems: "center", gap: 10, marginBottom: 20,
              }}>
                <Grid3x3 size={16} color="var(--accent-secondary)" />
                <p className="section-label" style={{ marginBottom: 0 }}>
                  Drum Pattern — 16 Steps
                </p>
              </div>
              <DrumPatternGrid pattern={result.pattern} />
            </div>
          )}
        </div>
      )}

      {/* Empty state */}
      {!result && !loading && !error && (
        <div className="card empty-state" style={{ marginTop: 32 }}>
          <div className="empty-state-icon glow-pulse">
            {mode === "ai"
              ? <Brain size={32} color="var(--accent-primary)" />
              : <Zap size={32} color="var(--accent-primary)" />
            }
          </div>
          <h3>Ready to Create</h3>
          <p>
            {mode === "ai"
              ? "Describe any beat or music style and let MusicGen AI generate it"
              : "Describe your beat using natural language and click Generate"
            }
          </p>
          <div style={{
            marginTop: 16, display: "flex", gap: 8,
            justifyContent: "center", flexWrap: "wrap",
          }}>
            {mode === "ai" ? (
              <>
                <span className="tag tag-sm">Full beats & melodies</span>
                <span className="tag tag-sm">Any genre</span>
                <span className="tag tag-sm">Meta MusicGen</span>
                <span className="tag tag-sm">Up to 30s</span>
              </>
            ) : (
              <>
                <span className="tag tag-sm">20+ genres</span>
                <span className="tag tag-sm">3 complexity levels</span>
                <span className="tag tag-sm">Instant synthesis</span>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
