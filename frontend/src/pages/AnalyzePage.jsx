import { useState, useEffect, useRef } from "react";
import { Music, Activity, Play, Pause } from "lucide-react";
import { api } from "../api";
import WaveSurfer from "wavesurfer.js";

const WaveformPlayer = ({ file, beatTimes, duration }) => {
    const containerRef = useRef(null);
    const [ws, setWs] = useState(null);
    const [isPlaying, setIsPlaying] = useState(false);

    useEffect(() => {
        if (!file || !containerRef.current) return;
        const objectUrl = URL.createObjectURL(file);
        const wavesurfer = WaveSurfer.create({
            container: containerRef.current,
            waveColor: 'rgba(124, 58, 237, 0.4)',
            progressColor: 'rgba(168, 85, 247, 0.8)',
            cursorColor: '#f1f5f9',
            barWidth: 2,
            barGap: 1,
            barRadius: 2,
            height: 100,
        });

        wavesurfer.load(objectUrl);
        wavesurfer.on('play', () => setIsPlaying(true));
        wavesurfer.on('pause', () => setIsPlaying(false));
        setWs(wavesurfer);

        return () => {
            wavesurfer.destroy();
            URL.revokeObjectURL(objectUrl);
        };
    }, [file]);

    return (
        <div style={{ marginTop: 24 }}>
            <div className="card" style={{ padding: 24, position: 'relative', overflow: 'hidden' }}>
                <h4 style={{ marginBottom: 16, fontSize: "0.9rem", color: "var(--text-secondary)" }}>
                    Interactive Waveform ({beatTimes?.length || 0} beats)
                </h4>
                <div style={{ position: 'relative' }}>
                    <div ref={containerRef} />
                    {beatTimes && duration && beatTimes.map((t, i) => (
                        <div key={i} style={{
                            position: 'absolute',
                            left: `${(t / duration) * 100}%`,
                            top: 0,
                            bottom: 0,
                            width: '1px',
                            backgroundColor: 'rgba(236, 72, 153, 0.5)',
                            pointerEvents: 'none'
                        }} title={`Beat at ${t.toFixed(2)}s`} />
                    ))}
                </div>
                <div style={{ marginTop: 16, display: 'flex', justifyContent: 'center' }}>
                    <button className="btn btn-secondary btn-sm" onClick={() => ws?.playPause()}>
                        {isPlaying ? <><Pause size={14} /> Pause</> : <><Play size={14} /> Play</>}
                    </button>
                </div>
            </div>
        </div>
    );
};

export default function AnalyzePage({ tracks, setTracks }) {
    const [selected, setSelected] = useState(null);
    const [analysis, setAnalysis] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleAnalyze = async (track) => {
        setSelected(track);
        setLoading(true);
        setError(null);
        setAnalysis(null);
        try {
            const result = await api.analyzeFile(track.file_id);
            setAnalysis(result);
            // Mark track as analyzed
            setTracks((prev) =>
                prev.map((t) =>
                    t.file_id === track.file_id
                        ? { ...t, analyzed: true, bpm: result.bpm, beat_times: result.beat_times }
                        : t
                )
            );
        } catch (err) {
            setError(err.message);
        }
        setLoading(false);
    };

    // Old renderBeatBars function removed

    return (
        <div className="animate-in">
            <div className="page-header">
                <h2>
                    <span className="gradient-text">Analyze</span> Audio
                </h2>
                <p>Select an uploaded track to detect beats and estimate BPM.</p>
            </div>

            {tracks.length === 0 ? (
                <div className="card" style={{ textAlign: "center", padding: 40 }}>
                    <p style={{ color: "var(--text-muted)" }}>
                        No tracks uploaded yet. Go to <strong>Upload</strong> first.
                    </p>
                </div>
            ) : (
                <div style={{ display: "grid", gridTemplateColumns: "300px 1fr", gap: 24 }}>
                    {/* Track selector */}
                    <div>
                        <h4 style={{ marginBottom: 12, color: "var(--text-secondary)", fontSize: "0.85rem" }}>
                            SELECT A TRACK
                        </h4>
                        <div className="track-list">
                            {tracks.map((t, i) => (
                                <div
                                    key={i}
                                    className={`track-item ${selected?.file_id === t.file_id ? "selected" : ""}`}
                                    onClick={() => handleAnalyze(t)}
                                >
                                    <div className="track-icon"><Music size={20} color="white" /></div>
                                    <div className="track-info">
                                        <div className="track-name">{t.filename}</div>
                                        <div className="track-meta">{t.duration}s</div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Analysis results */}
                    <div>
                        {loading && (
                            <div className="card" style={{ textAlign: "center", padding: 40 }}>
                                <div className="spinner" style={{ marginBottom: 12 }} />
                                <p style={{ color: "var(--text-secondary)" }}>Analyzing audio…</p>
                                <p style={{ color: "var(--text-muted)", fontSize: "0.8rem", marginTop: 4 }}>
                                    Extracting features, detecting beats, estimating BPM
                                </p>
                            </div>
                        )}

                        {error && (
                            <div className="status status-error">{error}</div>
                        )}

                        {analysis && !loading && (
                            <div className="animate-in">
                                {/* Stats */}
                                <div className="analysis-grid">
                                    <div className="stat-card">
                                        <div className="stat-value gradient-text">
                                            {analysis.bpm.toFixed(1)}
                                        </div>
                                        <div className="stat-label">BPM</div>
                                    </div>
                                    <div className="stat-card">
                                        <div className="stat-value" style={{ color: "var(--accent-secondary)" }}>
                                            {analysis.beat_times.length}
                                        </div>
                                        <div className="stat-label">Beats Detected</div>
                                    </div>
                                    <div className="stat-card">
                                        <div className="stat-value" style={{ color: "var(--success)" }}>
                                            {analysis.duration.toFixed(1)}s
                                        </div>
                                        <div className="stat-label">Duration</div>
                                    </div>
                                    <div className="stat-card">
                                        <div className="stat-value" style={{ color: "var(--warning)" }}>
                                            {analysis.sample_rate}
                                        </div>
                                        <div className="stat-label">Sample Rate</div>
                                    </div>
                                </div>

                                <WaveformPlayer
                                    file={selected?.originalFile}
                                    beatTimes={analysis.beat_times}
                                    duration={analysis.duration}
                                />

                                {/* Beat timestamps */}
                                <div className="card" style={{ marginTop: 16, maxHeight: 200, overflow: "auto" }}>
                                    <h4 style={{ marginBottom: 8, fontSize: "0.85rem", color: "var(--text-secondary)" }}>
                                        Beat Timestamps (seconds)
                                    </h4>
                                    <div style={{
                                        display: "flex", flexWrap: "wrap", gap: 6,
                                        fontFamily: "var(--font-mono)", fontSize: "0.75rem"
                                    }}>
                                        {analysis.beat_times.map((t, i) => (
                                            <span key={i} style={{
                                                background: "var(--bg-secondary)",
                                                padding: "4px 8px",
                                                borderRadius: 4,
                                                color: "var(--text-muted)"
                                            }}>
                                                {t.toFixed(3)}
                                            </span>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        )}

                        {!loading && !analysis && !error && (
                            <div className="card" style={{ textAlign: "center", padding: 40, opacity: 0.6 }}>
                                <p>← Select a track to analyze</p>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
