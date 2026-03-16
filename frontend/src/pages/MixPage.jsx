import { useState, useEffect, useRef } from "react";
import { Music, Headphones, Download, Play, Pause, Sliders } from "lucide-react";
import { api } from "../api";
import WaveSurfer from "wavesurfer.js";

// ── Mix output waveform ───────────────────────────────────────────────
function WaveformOutput({ url }) {
  const containerRef = useRef(null);
  const wsRef        = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [ready,     setReady]     = useState(false);

  useEffect(() => {
    if (!url || !containerRef.current) return;
    const ws = WaveSurfer.create({
      container:     containerRef.current,
      waveColor:     "rgba(124, 58, 237, 0.4)",
      progressColor: "rgba(236, 72, 153, 0.85)",
      cursorColor:   "rgba(241,245,249,0.4)",
      barWidth: 2, barGap: 1, barRadius: 2,
      height: 72, normalize: true, url,
    });
    ws.on("ready",  () => setReady(true));
    ws.on("play",   () => setIsPlaying(true));
    ws.on("pause",  () => setIsPlaying(false));
    ws.on("finish", () => setIsPlaying(false));
    wsRef.current = ws;
    return () => { ws.destroy(); wsRef.current = null; setIsPlaying(false); setReady(false); };
  }, [url]);

  return (
    <div>
      <div ref={containerRef} />
      <div style={{ marginTop: 12, display: "flex", gap: 10, alignItems: "center" }}>
        <button
          className="btn btn-secondary btn-sm"
          onClick={() => wsRef.current?.playPause()}
          disabled={!ready}
          style={{ minWidth: 72 }}
        >
          {isPlaying ? <><Pause size={12} /> Pause</> : <><Play size={12} /> Play</>}
        </button>
        <span style={{ color: "var(--text-muted)", fontSize: "0.74rem" }}>
          {ready ? "Ready" : "Loading waveform…"}
        </span>
      </div>
    </div>
  );
}

// ── Single deck card ──────────────────────────────────────────────────
function DeckCard({ label, value, onChange, exclude, tracks }) {
  return (
    <div className="mix-deck">
      <p className="section-label" style={{ marginBottom: 12 }}>{label}</p>

      {value ? (
        /* Selected state */
        <div style={{
          background: "rgba(124,58,237,0.08)",
          border: "1px solid rgba(124,58,237,0.25)",
          borderRadius: "var(--radius-md)",
          padding: "14px 16px",
          display: "flex", alignItems: "center", gap: 14,
        }}>
          <div className="track-icon" style={{ flexShrink: 0 }}>
            <Headphones size={17} color="white" />
          </div>
          <div className="track-info">
            <div className="track-name">{value.filename}</div>
            <div className="track-meta">
              {value.duration}s{value.bpm ? ` · ${value.bpm.toFixed(1)} BPM` : ""}
            </div>
          </div>
          <button
            className="btn btn-secondary btn-sm"
            onClick={() => onChange(null)}
            style={{ flexShrink: 0, padding: "5px 10px", fontSize: "0.75rem" }}
          >
            Clear
          </button>
        </div>
      ) : (
        /* Track picker */
        tracks.filter(t => t.file_id !== exclude?.file_id).length === 0 ? (
          <p style={{ color: "var(--text-muted)", fontSize: "0.82rem" }}>
            No other tracks available.
          </p>
        ) : (
          <div className="track-list">
            {tracks
              .filter(t => t.file_id !== exclude?.file_id)
              .map((t, i) => (
                <div key={i} className="track-item" onClick={() => onChange(t)}>
                  <div className="track-icon"><Music size={17} color="white" /></div>
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

// ── Main page ─────────────────────────────────────────────────────────
export default function MixPage({ tracks }) {
  const [trackA,   setTrackA]   = useState(null);
  const [trackB,   setTrackB]   = useState(null);
  const [crossfade, setCrossfade] = useState(5);
  const [loading,  setLoading]  = useState(false);
  const [result,   setResult]   = useState(null);
  const [error,    setError]    = useState(null);

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
        <p>Select two tracks, set crossfade duration, and generate a beat-synchronised DJ mix.</p>
      </div>

      {tracks.length < 2 ? (
        <div className="card" style={{ textAlign: "center", padding: "48px 24px" }}>
          <p style={{ color: "var(--text-muted)", fontSize: "0.88rem" }}>
            Upload at least{" "}
            <strong style={{ color: "var(--text-secondary)" }}>2 tracks</strong>{" "}
            before mixing.
          </p>
        </div>
      ) : (
        <>
          {/* ── Deck layout ───────────────────────────────────────── */}
          <div className="mix-layout">
            <DeckCard
              label="Deck A — Outgoing"
              value={trackA}
              onChange={setTrackA}
              exclude={trackB}
              tracks={tracks}
            />

            {/* Centre controls */}
            <div className="mix-controls">
              <p className="crossfade-label">Crossfade</p>
              <input
                type="range"
                className="crossfade-slider"
                min="0.5" max="15" step="0.5"
                value={crossfade}
                onChange={e => setCrossfade(parseFloat(e.target.value))}
              />
              <span style={{
                fontFamily: "var(--font-mono)",
                fontSize: "1.1rem", fontWeight: 700,
                color: "var(--accent-secondary)",
              }}>
                {crossfade}s
              </span>
              <button
                className="btn btn-primary"
                disabled={!canMix}
                onClick={handleMix}
                style={{ marginTop: 8, width: "100%", fontSize: "0.85rem" }}
              >
                {loading
                  ? <><span className="spinner" style={{ width: 14, height: 14 }} /> Mixing…</>
                  : <><Sliders size={15} /> Generate Mix</>
                }
              </button>
            </div>

            <DeckCard
              label="Deck B — Incoming"
              value={trackB}
              onChange={setTrackB}
              exclude={trackA}
              tracks={tracks}
            />
          </div>

          {/* Error */}
          {error && (
            <div className="status status-error" style={{ marginTop: 20 }}>
              {error}
            </div>
          )}

          {/* Result */}
          {result && (
            <div className="animate-in" style={{ marginTop: 24 }}>
              <div className="card" style={{ padding: 24 }}>
                <p className="section-label" style={{ marginBottom: 16 }}>Mix Output</p>

                {/* Stats */}
                <div className="analysis-grid" style={{ marginBottom: 20 }}>
                  {[
                    { v: result.target_bpm.toFixed(1),  l: "Mix BPM",    cl: "gradient"                 },
                    { v: `${result.duration.toFixed(1)}s`, l: "Duration", cl: "var(--accent-secondary)"  },
                    { v: result.bpm_a.toFixed(1),        l: "Track A",   cl: "var(--success)"            },
                    { v: result.bpm_b.toFixed(1),        l: "Track B",   cl: "var(--warning)"            },
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

                {/* Waveform */}
                <WaveformOutput url={api.getOutputUrl(result.output_file_id)} />

                {/* Download */}
                <div style={{ marginTop: 16 }}>
                  <a
                    className="btn btn-primary"
                    href={api.getOutputUrl(result.output_file_id)}
                    download={`automix_${result.output_file_id}.wav`}
                    style={{ textDecoration: "none" }}
                  >
                    <Download size={15} /> Download Mix
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
