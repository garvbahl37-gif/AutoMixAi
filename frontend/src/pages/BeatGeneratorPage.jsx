import { useState, useEffect, useRef } from "react";
import { Wand2, Play, Pause, Download, Grid3x3, RefreshCw, Zap } from "lucide-react";
import WaveSurfer from "wavesurfer.js";
import { api } from "../api";

// ── Preset styles ─────────────────────────────────────────────────────
const PRESETS = [
  { label: "Hip-Hop",     prompt: "chill hip-hop beat with heavy bass and smooth hi-hats" },
  { label: "Trap",        prompt: "dark trap beat with rolling hi-hats and 808 kicks" },
  { label: "House",       prompt: "energetic house EDM beat at 128 BPM with big drops" },
  { label: "Rock",        prompt: "driving rock beat with punchy snare and fast hi-hats" },
  { label: "Metal",       prompt: "heavy metal beat with double kick and aggressive fills" },
  { label: "Jazz",        prompt: "laid-back jazz swing beat at 95 BPM" },
  { label: "Reggae",      prompt: "one-drop reggae beat with off-beat rhythms at 80 BPM" },
  { label: "Drum & Bass", prompt: "fast drum and bass beat at 174 BPM with complex breaks" },
  { label: "Ambient",     prompt: "sparse ambient beat with slow atmospheric pulse" },
  { label: "Afrobeats",   prompt: "groovy afrobeats rhythm with syncopated patterns at 112 BPM" },
  { label: "Funk",        prompt: "funky groove beat with syncopated kick and tight snare" },
  { label: "Latin",       prompt: "latin rhythm with clave-inspired pattern at 100 BPM" },
];

const BARS_OPTIONS = [2, 4, 8, 16];

const GENRE_COLORS = {
  hiphop: "#a855f7", trap: "#ef4444", edm: "#06b6d4",
  rock: "#f59e0b", metal: "#dc2626", jazz: "#10b981",
  reggae: "#84cc16", dnb: "#f97316", ambient: "#6366f1",
  afrobeats: "#ec4899", funk: "#eab308", latin: "#14b8a6",
};

// ── Waveform player ───────────────────────────────────────────────────
function BeatWaveform({ url }) {
  const ref = useRef(null);
  const wsRef = useRef(null);
  const [playing, setPlaying] = useState(false);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (!url || !ref.current) return;
    const ws = WaveSurfer.create({
      container: ref.current,
      waveColor: "rgba(124, 58, 237, 0.4)",
      progressColor: "rgba(168, 85, 247, 0.85)",
      cursorColor: "rgba(241, 245, 249, 0.6)",
      barWidth: 2,
      barGap: 2,
      barRadius: 2,
      height: 72,
      normalize: true,
    });
    ws.load(url);
    ws.on("ready",  () => setReady(true));
    ws.on("play",   () => setPlaying(true));
    ws.on("pause",  () => setPlaying(false));
    ws.on("finish", () => setPlaying(false));
    wsRef.current = ws;
    return () => { ws.destroy(); wsRef.current = null; setPlaying(false); setReady(false); };
  }, [url]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <div ref={ref} />
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <button
          className="btn btn-secondary btn-sm"
          onClick={() => wsRef.current?.playPause()}
          disabled={!ready}
          style={{ minWidth: 80 }}
        >
          {playing ? <><Pause size={13} /> Pause</> : <><Play size={13} /> Play</>}
        </button>
        <span style={{ color: "var(--text-muted)", fontSize: "0.75rem", letterSpacing: "0.02em" }}>
          {ready ? "Ready" : "Loading…"}
        </span>
      </div>
    </div>
  );
}

// ── Drum pattern grid ─────────────────────────────────────────────────
const INSTRUMENTS = [
  { key: "kick",    label: "Kick",    color: "#7c3aed" },
  { key: "snare",   label: "Snare",   color: "#ec4899" },
  { key: "hihat_c", label: "HH–C",   color: "#06b6d4" },
  { key: "hihat_o", label: "HH–O",   color: "#0e7490" },
  { key: "clap",    label: "Clap",    color: "#f59e0b" },
];

function DrumPatternGrid({ pattern }) {
  if (!pattern) return null;
  return (
    <div style={{ overflowX: "auto" }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 5, minWidth: 480 }}>
        {/* Beat index row */}
        <div style={{ display: "flex", paddingLeft: 56 }}>
          {Array.from({ length: 16 }, (_, i) => (
            <div key={i} style={{
              flex: 1,
              textAlign: "center",
              fontSize: "0.62rem",
              color: i % 4 === 0 ? "var(--accent-secondary)" : "var(--text-muted)",
              fontFamily: "var(--font-mono)",
              fontWeight: i % 4 === 0 ? 700 : 400,
              paddingBottom: 4,
            }}>
              {i % 4 === 0 ? i / 4 + 1 : ""}
            </div>
          ))}
        </div>
        {/* Instrument rows */}
        {INSTRUMENTS.map(({ key, label, color }) => {
          const steps = pattern[key] || Array(16).fill(0);
          return (
            <div key={key} style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <span style={{
                width: 52, flexShrink: 0,
                fontSize: "0.7rem", letterSpacing: "0.04em",
                color: "var(--text-muted)", textAlign: "right",
                paddingRight: 8, fontFamily: "var(--font-mono)",
                textTransform: "uppercase",
              }}>
                {label}
              </span>
              {steps.map((hit, i) => (
                <div key={i} style={{
                  flex: 1, height: 20, borderRadius: 3,
                  background: hit
                    ? color
                    : i % 4 === 0
                      ? "rgba(255,255,255,0.06)"
                      : "rgba(255,255,255,0.025)",
                  border: `1px solid ${hit ? color + "aa" : i % 4 === 0 ? "rgba(255,255,255,0.08)" : "rgba(255,255,255,0.04)"}`,
                  boxShadow: hit ? `0 0 5px ${color}44` : "none",
                }} />
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Genre label ───────────────────────────────────────────────────────
function GenreChip({ genre, confidence }) {
  const color = GENRE_COLORS[genre] || "var(--accent-primary)";
  const pct = Math.round((confidence ?? 1) * 100);
  const name = genre.replace("hiphop", "Hip-Hop").replace("dnb", "Drum & Bass");
  return (
    <div style={{
      display: "inline-flex", alignItems: "center", gap: 8,
      background: `${color}18`, border: `1px solid ${color}40`,
      borderRadius: 6, padding: "5px 12px",
    }}>
      <span style={{ color, fontSize: "0.82rem", fontWeight: 700, letterSpacing: "0.03em" }}>
        {name.toUpperCase()}
      </span>
      <div style={{
        width: 48, height: 4, borderRadius: 2,
        background: "rgba(255,255,255,0.08)",
      }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 2 }} />
      </div>
      <span style={{ color: "var(--text-muted)", fontSize: "0.72rem", fontFamily: "var(--font-mono)" }}>
        {pct}%
      </span>
    </div>
  );
}

// ── Section label ─────────────────────────────────────────────────────
function SectionLabel({ children }) {
  return (
    <p style={{
      fontSize: "0.7rem", fontWeight: 700, letterSpacing: "0.1em",
      textTransform: "uppercase", color: "var(--text-muted)",
      marginBottom: 10,
    }}>
      {children}
    </p>
  );
}

// ── Main page ─────────────────────────────────────────────────────────
export default function BeatGeneratorPage() {
  const [prompt, setPrompt] = useState("");
  const [bars, setBars] = useState(4);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [outputUrl, setOutputUrl] = useState(null);

  const canGenerate = prompt.trim().length >= 3 && !loading;

  const applyPreset = (p) => setPrompt(p.prompt);

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
      {/* ── Header ──────────────────────────────────────────────── */}
      <div className="page-header">
        <h2><span className="gradient-text">Beat</span> Generator</h2>
        <p>Describe your beat in plain language — tempo, genre, style, mood.</p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 380px", gap: 24, alignItems: "start" }}>
        {/* ── Left column: input ──────────────────────────────────── */}
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

          {/* Prompt textarea */}
          <div className="card" style={{ padding: 20 }}>
            <SectionLabel>Prompt</SectionLabel>
            <textarea
              value={prompt}
              onChange={e => setPrompt(e.target.value)}
              onKeyDown={e => e.key === "Enter" && (e.metaKey || e.ctrlKey) && handleGenerate()}
              placeholder={
                "e.g. chill hip-hop beat at 90 BPM with heavy bass\n" +
                "e.g. dark trap banger with 808s and fast hi-hat rolls\n\n" +
                "Tip: ⌘ / Ctrl + Enter to generate"
              }
              rows={5}
              maxLength={500}
              style={{
                width: "100%", resize: "vertical",
                background: "var(--bg-secondary)", color: "var(--text-primary)",
                border: "1px solid var(--border-color)",
                borderRadius: "var(--radius-sm)",
                padding: "12px 14px", fontSize: "0.88rem",
                fontFamily: "var(--font-sans)", lineHeight: 1.7,
                outline: "none", transition: "border-color 0.2s",
              }}
              onFocus={e => e.target.style.borderColor = "var(--accent-primary)"}
              onBlur={e => e.target.style.borderColor = "var(--border-color)"}
            />
            <div style={{
              marginTop: 6, textAlign: "right",
              fontSize: "0.7rem", color: "var(--text-muted)",
              fontFamily: "var(--font-mono)",
            }}>
              {prompt.length} / 500
            </div>
          </div>

          {/* Style presets */}
          <div className="card" style={{ padding: 20 }}>
            <SectionLabel>Quick Style</SectionLabel>
            <div style={{
              display: "grid",
              gridTemplateColumns: "repeat(4, 1fr)",
              gap: 7,
            }}>
              {PRESETS.map(p => {
                const active = prompt === p.prompt;
                return (
                  <button
                    key={p.label}
                    onClick={() => applyPreset(p)}
                    style={{
                      background: active ? "rgba(124,58,237,0.18)" : "var(--bg-secondary)",
                      border: `1px solid ${active ? "rgba(124,58,237,0.5)" : "var(--border-color)"}`,
                      color: active ? "var(--accent-secondary)" : "var(--text-secondary)",
                      borderRadius: "var(--radius-sm)",
                      padding: "8px 10px",
                      fontSize: "0.78rem", fontWeight: 500,
                      cursor: "pointer", fontFamily: "var(--font-sans)",
                      transition: "all 0.15s",
                      letterSpacing: "0.01em",
                      whiteSpace: "nowrap", overflow: "hidden",
                      textOverflow: "ellipsis",
                    }}
                    onMouseEnter={e => { if (!active) e.target.style.borderColor = "rgba(124,58,237,0.3)"; }}
                    onMouseLeave={e => { if (!active) e.target.style.borderColor = "var(--border-color)"; }}
                  >
                    {p.label}
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* ── Right column: controls ───────────────────────────────── */}
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

          {/* Bars */}
          <div className="card" style={{ padding: 20 }}>
            <SectionLabel>Bars</SectionLabel>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8 }}>
              {BARS_OPTIONS.map(b => (
                <button
                  key={b}
                  onClick={() => setBars(b)}
                  style={{
                    padding: "10px 0",
                    borderRadius: "var(--radius-sm)",
                    border: `1px solid ${bars === b ? "rgba(124,58,237,0.55)" : "var(--border-color)"}`,
                    background: bars === b ? "rgba(124,58,237,0.16)" : "var(--bg-secondary)",
                    color: bars === b ? "var(--accent-secondary)" : "var(--text-secondary)",
                    cursor: "pointer", fontWeight: 700,
                    fontSize: "0.9rem", fontFamily: "var(--font-mono)",
                    transition: "all 0.15s",
                  }}
                >
                  {b}
                </button>
              ))}
            </div>
          </div>

          {/* Generate */}
          <button
            className="btn btn-primary"
            onClick={handleGenerate}
            disabled={!canGenerate}
            style={{ width: "100%", padding: "14px 0", fontSize: "0.92rem", letterSpacing: "0.02em" }}
          >
            {loading
              ? <><div className="spinner" style={{ width: 15, height: 15 }} /> Generating…</>
              : <><Wand2 size={16} /> Generate Beat</>
            }
          </button>

          {/* Result meta */}
          {result && !loading && (
            <div className="card animate-in" style={{ padding: 20, display: "flex", flexDirection: "column", gap: 14 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <GenreChip genre={result.genre} confidence={result.genre_confidence ?? 1} />
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                {[
                  { label: "BPM",        value: result.bpm,                  mono: true  },
                  { label: "Bars",       value: result.bars,                 mono: true  },
                  { label: "Duration",   value: `${result.duration.toFixed(1)}s`, mono: false },
                  { label: "Complexity", value: result.complexity,           mono: false },
                ].map(({ label, value, mono }) => (
                  <div key={label} style={{
                    background: "var(--bg-secondary)",
                    borderRadius: "var(--radius-sm)",
                    border: "1px solid var(--border-color)",
                    padding: "12px 14px",
                  }}>
                    <div style={{
                      fontSize: "1.15rem", fontWeight: 700,
                      fontFamily: mono ? "var(--font-mono)" : "var(--font-sans)",
                      color: "var(--text-primary)", textTransform: "capitalize",
                    }}>
                      {value}
                    </div>
                    <div style={{
                      fontSize: "0.68rem", letterSpacing: "0.08em",
                      textTransform: "uppercase", color: "var(--text-muted)",
                      marginTop: 3,
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
                  className="btn btn-secondary btn-sm"
                  style={{
                    textDecoration: "none", justifyContent: "center",
                    letterSpacing: "0.02em",
                  }}
                >
                  <Download size={13} /> Download WAV
                </a>
              )}
            </div>
          )}
        </div>
      </div>

      {/* ── Error ───────────────────────────────────────────────── */}
      {error && (
        <div className="status status-error" style={{ marginTop: 16 }}>{error}</div>
      )}

      {/* ── Player + Pattern ────────────────────────────────────────*/}
      {result && !loading && (
        <div className="animate-in" style={{ marginTop: 24, display: "flex", flexDirection: "column", gap: 16 }}>

          {/* Waveform */}
          <div className="card" style={{ padding: 20 }}>
            <div style={{
              display: "flex", justifyContent: "space-between",
              alignItems: "center", marginBottom: 16,
            }}>
              <SectionLabel>Waveform</SectionLabel>
              <button
                className="btn btn-secondary btn-sm"
                onClick={handleGenerate}
                style={{ display: "flex", alignItems: "center", gap: 6 }}
              >
                <RefreshCw size={12} /> Regenerate
              </button>
            </div>
            <BeatWaveform url={outputUrl} key={outputUrl} />
          </div>

          {/* Drum pattern */}
          <div className="card" style={{ padding: 20 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 16 }}>
              <Grid3x3 size={14} color="var(--text-muted)" />
              <SectionLabel>Drum Pattern — 16 Steps</SectionLabel>
            </div>
            <DrumPatternGrid pattern={result.pattern} />
          </div>
        </div>
      )}

      {/* ── Empty state ──────────────────────────────────────────── */}
      {!result && !loading && !error && (
        <div className="card" style={{ marginTop: 24, textAlign: "center", padding: "52px 24px" }}>
          <div style={{
            width: 48, height: 48, borderRadius: "var(--radius-md)",
            background: "rgba(124,58,237,0.12)", border: "1px solid rgba(124,58,237,0.2)",
            display: "flex", alignItems: "center", justifyContent: "center",
            margin: "0 auto 16px",
          }}>
            <Zap size={22} color="var(--accent-primary)" />
          </div>
          <p style={{ fontSize: "0.95rem", color: "var(--text-secondary)", fontWeight: 500 }}>
            Describe your beat and click Generate
          </p>
          <p style={{ color: "var(--text-muted)", fontSize: "0.8rem", marginTop: 6 }}>
            Supports 12 genres · 3 complexity levels · pure drum synthesis, no model needed
          </p>
        </div>
      )}
    </div>
  );
}
