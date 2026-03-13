import { useState, useEffect, useRef } from "react";
import { Music, Headphones, Download, PlayCircle, Pause, Settings2 } from "lucide-react";
import { api } from "../api";
import WaveSurfer from "wavesurfer.js";

const WaveformOutput = ({ url }) => {
    const containerRef = useRef(null);
    const [ws, setWs] = useState(null);
    const [isPlaying, setIsPlaying] = useState(false);

    useEffect(() => {
        if (!url || !containerRef.current) return;

        const wavesurfer = WaveSurfer.create({
            container: containerRef.current,
            waveColor: 'rgba(124, 58, 237, 0.4)',
            progressColor: 'rgba(236, 72, 153, 0.8)',
            cursorColor: '#f1f5f9',
            barWidth: 2,
            barGap: 1,
            barRadius: 2,
            height: 80,
            url: url
        });

        wavesurfer.on('play', () => setIsPlaying(true));
        wavesurfer.on('pause', () => setIsPlaying(false));

        setWs(wavesurfer);

        return () => wavesurfer.destroy();
    }, [url]);

    return (
        <div className="audio-player" style={{ padding: 24, marginTop: 16 }}>
            <h4 style={{ marginBottom: 16, fontSize: "0.85rem", color: "var(--text-secondary)" }}>
                <Headphones size={16} style={{ verticalAlign: 'middle', marginRight: 8 }} />
                Preview Mix
            </h4>
            <div ref={containerRef} style={{ marginBottom: 16 }} />
            <div style={{ display: 'flex', justifyContent: 'center' }}>
                <button className="btn btn-secondary btn-sm" onClick={() => ws?.playPause()}>
                    {isPlaying ? <><Pause size={14} /> Pause</> : <><PlayCircle size={14} /> Play</>}
                </button>
            </div>
        </div>
    );
};

export default function MixPage({ tracks }) {
    const [trackA, setTrackA] = useState(null);
    const [trackB, setTrackB] = useState(null);
    const [crossfade, setCrossfade] = useState(5);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    const handleMix = async () => {
        if (!trackA || !trackB) return;
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

    const TrackSelector = ({ label, emoji, value, onChange, exclude }) => (
        <div className="mix-deck">
            <h3>{label}</h3>
            {value ? (
                <div className="track-item selected" style={{ cursor: "default" }}>
                    <div className="track-icon"><Headphones size={20} color="white" /></div>
                    <div className="track-info">
                        <div className="track-name">{value.filename}</div>
                        <div className="track-meta">
                            {value.duration}s {value.bpm ? `• ${value.bpm.toFixed(1)} BPM` : ""}
                        </div>
                    </div>
                    <button
                        className="btn btn-sm btn-secondary"
                        onClick={() => onChange(null)}
                    >
                        ✕
                    </button>
                </div>
            ) : (
                <div className="track-list">
                    {tracks
                        .filter((t) => t.file_id !== exclude?.file_id)
                        .map((t, i) => (
                            <div
                                key={i}
                                className="track-item"
                                onClick={() => onChange(t)}
                            >
                                <div className="track-icon"><Music size={20} color="white" /></div>
                                <div className="track-info">
                                    <div className="track-name">{t.filename}</div>
                                    <div className="track-meta">
                                        {t.duration}s {t.bpm ? `• ${t.bpm.toFixed(1)} BPM` : ""}
                                    </div>
                                </div>
                            </div>
                        ))}
                </div>
            )}
        </div>
    );

    return (
        <div className="animate-in">
            <div className="page-header">
                <h2>
                    <span className="gradient-text">Mix</span> Tracks
                </h2>
                <p>Select two tracks, set crossfade duration, and generate a beat-synchronized DJ mix.</p>
            </div>

            {tracks.length < 2 ? (
                <div className="card" style={{ textAlign: "center", padding: 40 }}>
                    <p style={{ color: "var(--text-muted)" }}>
                        You need at least <strong>2 uploaded tracks</strong> to create a mix.
                    </p>
                </div>
            ) : (
                <>
                    <div className="mix-layout">
                        <TrackSelector
                            label="Deck A — Outgoing"
                            emoji="🅰️"
                            value={trackA}
                            onChange={setTrackA}
                            exclude={trackB}
                        />

                        {/* Center controls */}
                        <div className="mix-controls">
                            <span className="crossfade-label">Crossfade</span>
                            <input
                                type="range"
                                className="crossfade-slider"
                                min="0.5"
                                max="15"
                                step="0.5"
                                value={crossfade}
                                onChange={(e) => setCrossfade(parseFloat(e.target.value))}
                            />
                            <span style={{
                                fontFamily: "var(--font-mono)",
                                color: "var(--accent-secondary)",
                                fontWeight: 600,
                            }}>
                                {crossfade}s
                            </span>

                            <button
                                className="btn btn-primary"
                                disabled={!trackA || !trackB || loading}
                                onClick={handleMix}
                                style={{ marginTop: 8 }}
                            >
                                {loading ? (
                                    <>
                                        <span className="spinner" /> Mixing…
                                    </>
                                ) : (
                                    <><Settings2 size={16} /> Generate Mix</>
                                )}
                            </button>
                        </div>

                        <TrackSelector
                            label="Deck B — Incoming"
                            emoji="🅱️"
                            value={trackB}
                            onChange={setTrackB}
                            exclude={trackA}
                        />
                    </div>

                    {error && (
                        <div className="status status-error" style={{ marginTop: 24 }}>
                            ✗ {error}
                        </div>
                    )}

                    {result && (
                        <div className="animate-in" style={{ marginTop: 24 }}>
                            <div className="card glow-border">
                                <h3 style={{ marginBottom: 16 }}>
                                    Mix Generated Successfully
                                </h3>

                                <div className="analysis-grid" style={{ marginBottom: 16 }}>
                                    <div className="stat-card">
                                        <div className="stat-value gradient-text">
                                            {result.target_bpm.toFixed(1)}
                                        </div>
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
                                        <div className="stat-label">Track A BPM</div>
                                    </div>
                                    <div className="stat-card">
                                        <div className="stat-value" style={{ color: "var(--warning)" }}>
                                            {result.bpm_b.toFixed(1)}
                                        </div>
                                        <div className="stat-label">Track B BPM</div>
                                    </div>
                                </div>

                                <WaveformOutput url={api.getOutputUrl(result.output_file_id)} />

                                <div style={{ marginTop: 16, display: "flex", gap: 12 }}>
                                    <a
                                        className="btn btn-primary"
                                        href={api.getOutputUrl(result.output_file_id)}
                                        download={`automix_${result.output_file_id}.wav`}
                                    >
                                        <Download size={16} /> Download Mix
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
