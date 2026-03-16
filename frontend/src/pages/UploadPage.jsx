import { useState, useRef } from "react";
import { UploadCloud, Music, CheckCircle } from "lucide-react";
import { api } from "../api";

export default function UploadPage({ tracks, setTracks }) {
  const [dragging,  setDragging]  = useState(false);
  const [uploading, setUploading] = useState(false);
  const [status,    setStatus]    = useState(null);
  const fileRef = useRef(null);

  const handleFiles = async (files) => {
    for (const file of files) {
      setUploading(true);
      setStatus({ type: "loading", text: `Uploading ${file.name}…` });
      try {
        const result = await api.uploadFile(file);
        setTracks(prev => [...prev, { ...result, originalFile: file, analyzed: false }]);
        setStatus({ type: "success", text: `${file.name} — ${result.duration}s uploaded` });
      } catch (err) {
        setStatus({ type: "error", text: err.message });
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
        <h2><span className="gradient-text">Upload</span> Audio</h2>
        <p>Drag and drop audio files or click to browse. WAV, MP3, FLAC, OGG, M4A supported.</p>
      </div>

      {/* Drop zone */}
      <div
        className={`upload-zone ${dragging ? "dragging" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => fileRef.current?.click()}
      >
        {uploading
          ? <div className="spinner" style={{ marginBottom: 18, width: 28, height: 28, borderWidth: 3 }} />
          : <UploadCloud size={40} style={{ marginBottom: 16, color: "var(--accent-primary)", opacity: 0.85 }} />
        }
        <h3 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: 6 }}>
          {uploading ? "Uploading…" : "Drop audio files here"}
        </h3>
        <p style={{ color: "var(--text-muted)", fontSize: "0.82rem" }}>
          or click to browse your files
        </p>
        <input
          ref={fileRef}
          type="file"
          accept=".wav,.mp3,.flac,.ogg,.m4a,.aac"
          multiple
          hidden
          onChange={(e) => e.target.files.length && handleFiles([...e.target.files])}
        />
      </div>

      {/* Status message */}
      {status && (
        <div
          className={`status status-${status.type}`}
          style={{ marginTop: 14 }}
        >
          {status.type === "loading" && <span className="spinner" />}
          {status.text}
        </div>
      )}

      {/* Track library */}
      {tracks.length > 0 && (
        <div style={{ marginTop: 32 }}>
          <div style={{
            display: "flex", justifyContent: "space-between",
            alignItems: "center", marginBottom: 12,
          }}>
            <p className="section-label">Library — {tracks.length} {tracks.length === 1 ? "file" : "files"}</p>
          </div>

          <div className="track-list">
            {tracks.map((t, i) => (
              <div key={i} className="track-item" style={{ cursor: "default" }}>
                <div className="track-icon">
                  <Music size={17} color="white" />
                </div>
                <div className="track-info">
                  <div className="track-name">{t.filename}</div>
                  <div className="track-meta">
                    {t.duration}s · {t.file_id.slice(0, 8)}
                  </div>
                </div>
                {t.analyzed && (
                  <div style={{
                    display: "flex", alignItems: "center", gap: 5,
                    color: "var(--success)", fontSize: "0.75rem", fontWeight: 500,
                    flexShrink: 0,
                  }}>
                    <CheckCircle size={13} />
                    Analyzed
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
