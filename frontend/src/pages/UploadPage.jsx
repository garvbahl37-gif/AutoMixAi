import { useState, useRef } from "react";
import { UploadCloud, Music, CheckCircle } from "lucide-react";
import { api } from "../api";

export default function UploadPage({ tracks, setTracks }) {
    const [dragging, setDragging] = useState(false);
    const [uploading, setUploading] = useState(false);
    const [status, setStatus] = useState(null);
    const fileRef = useRef(null);

    const handleFiles = async (files) => {
        for (const file of files) {
            setUploading(true);
            setStatus({ type: "loading", text: `Uploading ${file.name}…` });
            try {
                const result = await api.uploadFile(file);
                setTracks((prev) => [
                    ...prev,
                    { ...result, originalFile: file, analyzed: false },
                ]);
                setStatus({ type: "success", text: `✓ ${file.name} uploaded — ${result.duration}s` });
            } catch (err) {
                setStatus({ type: "error", text: `✗ ${err.message}` });
            }
            setUploading(false);
        }
    };

    const onDrop = (e) => {
        e.preventDefault();
        setDragging(false);
        if (e.dataTransfer.files.length) handleFiles([...e.dataTransfer.files]);
    };

    return (
        <div className="animate-in">
            <div className="page-header">
                <h2>
                    <span className="gradient-text">Upload</span> Audio
                </h2>
                <p>Drag & drop your audio files or click to browse. Supports WAV, MP3, FLAC, OGG, M4A.</p>
            </div>

            {/* Drop Zone */}
            <div
                className={`upload-zone ${dragging ? "dragging" : ""}`}
                onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
                onClick={() => fileRef.current?.click()}
            >
                <UploadCloud size={48} style={{ marginBottom: 16, color: "var(--accent-primary)" }} />
                <h3>{uploading ? "Uploading…" : "Drop audio files here"}</h3>
                <p>or click to browse your files</p>
                {uploading && <div className="spinner" style={{ marginTop: 16 }} />}
                <input
                    ref={fileRef}
                    type="file"
                    accept=".wav,.mp3,.flac,.ogg,.m4a,.aac"
                    multiple
                    hidden
                    onChange={(e) => e.target.files.length && handleFiles([...e.target.files])}
                />
            </div>

            {status && (
                <div className={`status status-${status.type}`}>
                    {status.type === "loading" && <span className="spinner" />}
                    {status.text}
                </div>
            )}

            {/* Track Library */}
            {tracks.length > 0 && (
                <div style={{ marginTop: 32 }}>
                    <h3 style={{ marginBottom: 16, fontSize: "1.1rem" }}>
                        Uploaded Tracks ({tracks.length})
                    </h3>
                    <div className="track-list">
                        {tracks.map((t, i) => (
                            <div key={i} className="track-item">
                                <div className="track-icon"><Music size={20} color="white" /></div>
                                <div className="track-info">
                                    <div className="track-name">{t.filename}</div>
                                    <div className="track-meta">
                                        {t.duration}s • ID: {t.file_id.slice(0, 8)}…
                                    </div>
                                </div>
                                {t.analyzed && (
                                    <span style={{ color: "var(--success)", fontSize: "0.8rem", display: "flex", alignItems: "center", gap: 4 }}>
                                        <CheckCircle size={14} /> Analyzed
                                    </span>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
