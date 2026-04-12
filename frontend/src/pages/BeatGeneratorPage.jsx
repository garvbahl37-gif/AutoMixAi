import { useState, useEffect, useRef } from "react";
import {
  Wand2, Play, Pause, Download, Grid3x3, RefreshCw,
  Zap, Sparkles, Music, Hash, Clock, Layers, Cpu, Brain,
  Sliders, Settings, Gauge, Dice5
} from "lucide-react";
import WaveSurfer from "wavesurfer.js";
import { api } from "../api";

// ─── AI Gen Colab Backend ───────────────────────────────────────────────────
const COLAB_API = "https://bustled-hertha-unprojective.ngrok-free.dev";

// ─── Preset styles ──────────────────────────────────────────────────────────

const AI_PRESETS = [
  { label: "Cloud Trap", prompt: "A dark and melancholic cloud trap beat, with nostalgic piano, plucked bass and synth bells, at 110 BPM.", color: "#a855f7" },
  { label: "Lo-Fi Jazz Rap", prompt: "A laid back lo-fi jazz rap at 85 BPM, featuring deep sub, plucked bass, and vocal chop, with chill and jazzy relaxed moods.", color: "#06b6d4" },
  { label: "Cinematic Trap", prompt: "Melancholic trap beat at 105 BPM with shimmering synth bells and deep sub bass, minor chord progressions on piano, and airy vocal pads, evoking a cinematic and emotional atmosphere.", color: "#ef4444" },
  { label: "Jazzy Chillhop", prompt: "A jazzy chillhop beat at 101 BPM featuring synth bells, vocal pad, and movie sample, evoking trap nostalgic and chill moods.", color: "#f59e0b" },
  { label: "Seductive Trap", prompt: "Smooth and seductive at 115 BPM trap beat with electric guitar riffs, plucked bass, vocal adlibs, and warm synth pads. Relaxed, romantic, and sexy mood.", color: "#14b8a6" },
  { label: "Moody Cloud", prompt: "A moody cloud trap beat, boomy bass, synth bells and melodic piano, evoking etherate mood at 100 BPM.", color: "#6366f1" },
  { label: "Smooth R&B", prompt: "A smooth neo-soul R&B instrumental at 90 BPM in D major, featuring live bass, soft Rhodes keys, and warm analog drum grooves.", color: "#ec4899" },
];


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
    return `${secs}.${ms.toString().padStart(2, "0")}`;
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

// ─── Generation status indicator ────────────────────────────────────────────

function GeneratingStatus({ statusMsg }) {
  if (!statusMsg) return null;
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 10,
      padding: "12px 16px",
      background: "rgba(139,92,246,0.08)",
      border: "1px solid rgba(139,92,246,0.2)",
      borderRadius: "var(--radius-md)",
      fontSize: "0.82rem",
      color: "var(--accent-secondary)",
    }}>
      <div className="spinner" style={{ width: 14, height: 14, flexShrink: 0 }} />
      {statusMsg}
    </div>
  );
}

// ─── Main page ──────────────────────────────────────────────────────────────

export default function BeatGeneratorPage() {
  const mode = "ai";
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [statusMsg, setStatusMsg] = useState("");
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [outputUrl, setOutputUrl] = useState(null);
  const [activePreset, setActivePreset] = useState(null);

  // AI mode controls
  const [duration, setDuration] = useState(30);
  const [steps, setSteps] = useState(100);
  const [cfgScale, setCfgScale] = useState(7.0);
  const [seed, setSeed] = useState(-1);


  const canGenerate = prompt.trim().length >= 3 && !loading;

  const applyPreset = (p) => {
    setPrompt(p.prompt);
    setActivePreset(p.label);
  };

  const randomizeSeed = () => {
    setSeed(Math.floor(Math.random() * 999999));
  };

  const handleGenerate = async () => {
    if (!canGenerate) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setOutputUrl(null);
    setStatusMsg("");

    try {
      if (mode === "ai") {
        // ── AI Gen Colab backend (async job + polling) ──────────────────
        // Step 1: Submit generation job
        setStatusMsg("Submitting generation job to AI model…");

        const payload = {
          prompt: prompt,
          steps: steps,
          cfg_scale: cfgScale,
          duration: duration,
        };

        const submitRes = await fetch(`${COLAB_API}/generate`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true",
          },
          body: JSON.stringify(payload),
        });

        if (!submitRes.ok) {
          const errData = await submitRes.json().catch(() => ({}));
          throw new Error(errData.detail || errData.error || `Job submission failed (HTTP ${submitRes.status})`);
        }

        const submitData = await submitRes.json();
        const jobId = submitData.job_id;

        if (!jobId) {
          throw new Error("Backend did not return a job_id");
        }

        // Step 2: Poll /result/{job_id} until status is 'done' or 'error'
        setStatusMsg(`Generating ${duration}s beat with AI (${steps} steps, CFG ${cfgScale})…`);

        let pollData = null;
        const POLL_INTERVAL = 3000; // 3 seconds between polls
        const MAX_POLL_TIME = 600000; // 10 minute max wait
        const startTime = Date.now();

        while (Date.now() - startTime < MAX_POLL_TIME) {
          await new Promise(r => setTimeout(r, POLL_INTERVAL));

          const elapsed = Math.floor((Date.now() - startTime) / 1000);
          setStatusMsg(`Generating beat… ${elapsed}s elapsed (${steps} steps, CFG ${cfgScale})`);

          const pollRes = await fetch(`${COLAB_API}/result/${jobId}`, {
            headers: { "ngrok-skip-browser-warning": "true" },
          });

          if (!pollRes.ok) {
            // Might be a transient error, keep polling
            console.warn("Poll request failed, retrying…", pollRes.status);
            continue;
          }

          pollData = await pollRes.json();

          if (pollData.status === "done") {
            break;
          } else if (pollData.status === "error") {
            throw new Error(pollData.error || "Generation failed on the server");
          }
          // status === "processing" / "pending" → keep polling
        }

        if (!pollData || pollData.status !== "done") {
          throw new Error("Generation timed out — the model may be overloaded. Try again.");
        }

        // Step 3: Extract audio from the result
        // Backend field is "audio" (base64-encoded WAV)
        const audioB64 = pollData.audio || pollData.audio_base64;
        if (audioB64) {
          // Base64-encoded WAV audio
          const byteChars = atob(audioB64);
          const byteArray = new Uint8Array(byteChars.length);
          for (let i = 0; i < byteChars.length; i++) {
            byteArray[i] = byteChars.charCodeAt(i);
          }
          const blob = new Blob([byteArray], { type: "audio/wav" });
          setOutputUrl(URL.createObjectURL(blob));
        } else if (pollData.audio_url) {
          // Direct URL to audio file
          setOutputUrl(pollData.audio_url);
        } else if (pollData.file_url) {
          setOutputUrl(pollData.file_url);
        } else if (pollData.file_id) {
          // Fetch the audio file from the backend
          const audioRes = await fetch(`${COLAB_API}/output/${pollData.file_id}`, {
            headers: { "ngrok-skip-browser-warning": "true" },
          });
          const blob = await audioRes.blob();
          setOutputUrl(URL.createObjectURL(blob));
        } else {
          throw new Error("Server returned 'done' but no audio data was found in the response");
        }

        setResult({
          mode: "ai",
          duration: pollData.duration || duration,
          model: "AI Audio",
          sample_rate: pollData.sample_rate || 44100,
          steps: pollData.steps || steps,
          cfg_scale: pollData.cfg_scale || cfgScale,
          seed: pollData.seed ?? seed,
        });
      }
    } catch (err) {
      console.error("Generation error:", err);
      setError(err.message || "An unexpected error occurred");
    }

    setLoading(false);
    setStatusMsg("");
  };

  const presets = AI_PRESETS;

  return (
    <div className="animate-in">
      {/* Header */}
      <div className="page-header">
        <h2><span className="gradient-text">Beat</span> Generator</h2>
        <p style={{ fontSize: "0.95rem", opacity: 0.9 }}>
          Generate studio-quality beats using our fine-tuned AI model.
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
              display: "flex", alignItems: "center", gap: 10, marginBottom: 16,
            }}>
              <Wand2 size={18} color="var(--accent-secondary)" />
              <p className="section-label" style={{ marginBottom: 0 }}>Prompt</p>
            </div>
              <textarea
              value={prompt}
              onChange={e => { setPrompt(e.target.value); setActivePreset(null); }}
              onKeyDown={e => e.key === "Enter" && (e.metaKey || e.ctrlKey) && handleGenerate()}
              placeholder={"e.g. A dark and melancholic cloud trap beat with piano, plucked bass at 110 BPM\ne.g. Smooth R&B beat at 90 BPM with Rhodes keys and warm drums\n\nTip: Press ⌘/Ctrl + Enter to generate"}
              rows={5}
              maxLength={500}
              className="textarea"
              style={{ lineHeight: 1.8, fontSize: "0.92rem", background: "rgba(255,255,255,0.015)" }}
            />
            <div style={{
              marginTop: 8, display: "flex",
              justifyContent: "space-between", alignItems: "center",
            }}>
              <span style={{ fontSize: "0.72rem", color: "var(--text-dim)" }}>
                Include instruments, mood, BPM & key for best results
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
                      background: active 
                        ? `linear-gradient(135deg, ${p.color}15, ${p.color}05)` 
                        : "rgba(255,255,255,0.02)",
                      border: `1px solid ${active ? `${p.color}50` : "rgba(255,255,255,0.06)"}`,
                      color: active ? p.color : "var(--text-secondary)",
                      borderRadius: "var(--radius-full)",
                      padding: "8px 14px",
                      fontSize: "0.78rem",
                      fontWeight: 600,
                      cursor: "pointer",
                      fontFamily: "var(--font-sans)",
                      transition: "all 0.2s ease",
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      boxShadow: active ? `0 4px 12px ${p.color}10` : "none",
                    }}
                    onMouseEnter={e => {
                      if (!active) {
                        e.target.style.borderColor = `${p.color}40`;
                        e.target.style.color = p.color;
                        e.target.style.background = `linear-gradient(135deg, ${p.color}10, transparent)`;
                      }
                    }}
                    onMouseLeave={e => {
                      if (!active) {
                        e.target.style.borderColor = "rgba(255,255,255,0.06)";
                        e.target.style.color = "var(--text-secondary)";
                        e.target.style.background = "rgba(255,255,255,0.02)";
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
          {true && (
            <>
              {/* AI Controls */}
              <div className="card" style={{ padding: 24 }}>
                <div style={{
                  display: "flex", alignItems: "center", gap: 10, marginBottom: 20,
                }}>
                  <Sliders size={18} color="var(--accent-secondary)" />
                  <p className="section-label" style={{ marginBottom: 0 }}>Generation Controls</p>
                </div>

                <SliderControl
                  label="Duration" value={duration} onChange={setDuration}
                  min={10} max={47} step={1} unit="s"
                  icon={Clock} color="#10b981"
                />

                <SliderControl
                  label="Steps" value={steps} onChange={setSteps}
                  min={50} max={250} step={10} unit=""
                  icon={Gauge} color="#f59e0b"
                />

                <SliderControl
                  label="CFG Scale" value={cfgScale} onChange={setCfgScale}
                  min={1} max={15} step={0.5} unit=""
                  icon={Settings} color="#a855f7"
                />


              </div>
            </>
          )}

          {/* Status message while loading */}
          {loading && <GeneratingStatus statusMsg={statusMsg} />}

          {/* Generate button */}
          <button
            className="btn btn-primary btn-lg glow-pulse"
            onClick={handleGenerate}
            disabled={!canGenerate}
            style={{
              width: "100%", padding: "16px 0",
              fontSize: "1.05rem", letterSpacing: "0.03em",
              fontWeight: 700,
              boxShadow: "0 8px 30px rgba(139, 92, 246, 0.3)",
              border: "1px solid rgba(168, 85, 247, 0.4)",
              transition: "all 0.2s ease"
            }}
          >
            {loading
              ? <><div className="spinner" style={{ width: 18, height: 18 }} /> Generating…</>
              : <><Wand2 size={18} /> Generate</>
            }
          </button>

          {/* Result meta */}
          {result && !loading && (
            <div className="card glow-border animate-in" style={{ padding: 24 }}>
              {/* Stats */}
              <div style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 12,
              }}>
                <>
                  {[
                    { label: "Duration",   value: `${result.duration}s`,  icon: Clock    },
                    { label: "Steps",      value: `${result.steps}`,      icon: Gauge    },
                    { label: "CFG Scale",  value: `${result.cfg_scale}`,  icon: Settings },
                    { label: "Model",      value: "AI Audio",             icon: Brain    },
                  ].map(({ label, value, icon: Icon }) => (
                    <div key={label} style={{
                      background: "rgba(255,255,255,0.02)",
                      borderRadius: "var(--radius-md)",
                      border: "1px solid rgba(255,255,255,0.05)",
                      padding: "14px 16px",
                      position: "relative",
                      backdropFilter: "blur(10px)",
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
              </div>

              {outputUrl && (
                <a
                  href={outputUrl}
                  download="beat_ai_gen.wav"
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
                {true && (
                  <span style={{
                    padding: "3px 10px",
                    background: "linear-gradient(135deg, rgba(139,92,246,0.15), rgba(168,85,247,0.1))",
                    border: "1px solid rgba(139,92,246,0.3)",
                    borderRadius: "var(--radius-full)",
                    fontSize: "0.65rem",
                    letterSpacing: "0.03em",
                    fontWeight: 800,
                    color: "#c084fc",
                  }}>
                    AI Gen · 44.1 kHz
                  </span>
                )}
              </div>
              <button className="btn btn-ghost btn-sm" onClick={handleGenerate}>
                <RefreshCw size={13} /> Regenerate
              </button>
            </div>
            <BeatWaveform url={outputUrl} key={outputUrl} />
          </div>
        </div>
      )}

      {/* Empty state */}
      {!result && !loading && !error && (
        <div className="card empty-state" style={{ marginTop: 32 }}>
          <div className="empty-state-icon glow-pulse">
            <Brain size={32} color="var(--accent-primary)" />
          </div>
          <h3 style={{ fontSize: "1.4rem", fontWeight: 800 }}>Ready to Create</h3>
          <p style={{ opacity: 0.8, maxWidth: 300, margin: "0 auto" }}>
            Describe a trap, hip-hop, or R&B beat and let our AI generate it.
          </p>
          <div style={{
            marginTop: 20, display: "flex", gap: 10,
            justifyContent: "center", flexWrap: "wrap",
          }}>
            <span className="tag tag-sm" style={{ background: "rgba(139,92,246,0.1)", border: "1px solid rgba(139,92,246,0.2)", color: "#c084fc" }}>Trap · Hip-Hop · R&B</span>
            <span className="tag tag-sm" style={{ background: "rgba(16,185,129,0.1)", border: "1px solid rgba(16,185,129,0.2)", color: "#34d399" }}>Cloud · Drill · Phonk</span>
            <span className="tag tag-sm" style={{ background: "rgba(6,182,212,0.1)", border: "1px solid rgba(6,182,212,0.2)", color: "#22d3ee" }}>44.1 kHz · Up to 47s</span>
          </div>
        </div>
      )}
    </div>
  );
}
