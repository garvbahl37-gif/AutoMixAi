import { useState, useRef } from "react";
import { UploadCloud, Music, CheckCircle, FileAudio, Trash2, Clock } from "lucide-react";
import { api } from "../api";

export default function UploadPage({ tracks, setTracks }) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState(null);
  const [progress, setProgress] = useState(0);
  const fileRef = useRef(null);

  const handleFiles = async (files) => {
    const audioFiles = [...files].filter(f =>
      /\.(wav|mp3|flac|ogg|m4a|aac)$/i.test(f.name)
    );

    if (audioFiles.length === 0) {
      setStatus({ type: "error", text: "Please select valid audio files (WAV, MP3, FLAC, OGG, M4A)" });
      return;
    }

    for (let i = 0; i < audioFiles.length; i++) {
      const file = audioFiles[i];
      setUploading(true);
      setProgress(Math.round((i / audioFiles.length) * 100));
      setStatus({
        type: "loading",
        text: `Uploading ${file.name}${audioFiles.length > 1 ? ` (${i + 1}/${audioFiles.length})` : ""}`
      });

      try {
        const result = await api.uploadFile(file);
        setTracks(prev => [...prev, {
          ...result,
          originalFile: file,
          analyzed: false,
          uploadedAt: new Date().toISOString(),
        }]);
        setStatus({
          type: "success",
          text: `${file.name} uploaded successfully`
        });
      } catch (err) {
        setStatus({ type: "error", text: err.message });
      }
    }

    setUploading(false);
    setProgress(100);
    setTimeout(() => setProgress(0), 1000);
  };

  const removeTrack = (fileId) => {
    setTracks(prev => prev.filter(t => t.file_id !== fileId));
  };

  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files);
  };

  return (
    <div className="animate-in">
      {/* Header */}
      <div className="page-header">
        <h2>
          <span className="gradient-text">Upload</span> Audio
        </h2>
        <p>
          Import your audio tracks for AI-powered analysis and mixing.
          Supports WAV, MP3, FLAC, OGG, and M4A formats.
        </p>
      </div>

      {/* Upload Zone */}
      <div
        className={`upload-zone ${dragging ? "dragging" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => fileRef.current?.click()}
      >
        <div className="upload-icon">
          {uploading ? (
            <div className="spinner spinner-lg" />
          ) : (
            <UploadCloud
              size={28}
              color="var(--accent-primary)"
              strokeWidth={1.5}
            />
          )}
        </div>

        <h3>{uploading ? "Processing..." : "Drop your audio files here"}</h3>
        <p>or click to browse your computer</p>

        {/* Progress bar */}
        {uploading && progress > 0 && (
          <div style={{
            marginTop: 24,
            width: "100%",
            maxWidth: 300,
          }}>
            <div style={{
              height: 4,
              background: "var(--bg-secondary)",
              borderRadius: 2,
              overflow: "hidden",
            }}>
              <div style={{
                width: `${progress}%`,
                height: "100%",
                background: "var(--accent-gradient)",
                transition: "width 0.3s ease",
                borderRadius: 2,
              }} />
            </div>
          </div>
        )}

        <input
          ref={fileRef}
          type="file"
          accept=".wav,.mp3,.flac,.ogg,.m4a,.aac"
          multiple
          hidden
          onChange={(e) => e.target.files.length && handleFiles(e.target.files)}
        />
      </div>

      {/* Supported formats */}
      <div style={{
        display: "flex",
        justifyContent: "center",
        gap: 12,
        marginTop: 16,
        flexWrap: "wrap",
      }}>
        {["WAV", "MP3", "FLAC", "OGG", "M4A"].map(fmt => (
          <span
            key={fmt}
            style={{
              padding: "4px 12px",
              background: "var(--bg-card)",
              border: "1px solid var(--border-color)",
              borderRadius: "var(--radius-full)",
              fontSize: "0.72rem",
              fontWeight: 600,
              fontFamily: "var(--font-mono)",
              color: "var(--text-muted)",
              letterSpacing: "0.05em",
            }}
          >
            {fmt}
          </span>
        ))}
      </div>

      {/* Status message */}
      {status && (
        <div
          className={`status status-${status.type} animate-in`}
          style={{ marginTop: 20 }}
        >
          {status.type === "loading" && <span className="spinner" />}
          {status.type === "success" && <CheckCircle size={16} />}
          {status.text}
        </div>
      )}

      {/* Track Library */}
      {tracks.length > 0 && (
        <div style={{ marginTop: 40 }} className="animate-in">
          {/* Library header */}
          <div style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: 16,
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <div style={{
                width: 32,
                height: 32,
                borderRadius: "var(--radius-md)",
                background: "var(--accent-gradient-subtle)",
                border: "1px solid rgba(139, 92, 246, 0.2)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}>
                <FileAudio size={16} color="var(--accent-primary)" />
              </div>
              <div>
                <p className="section-label" style={{ marginBottom: 2 }}>
                  Track Library
                </p>
                <p style={{
                  fontSize: "0.75rem",
                  color: "var(--text-dim)",
                }}>
                  {tracks.length} {tracks.length === 1 ? "file" : "files"} •{" "}
                  {formatDuration(tracks.reduce((acc, t) => acc + (t.duration || 0), 0))} total
                </p>
              </div>
            </div>

            {tracks.length > 1 && (
              <button
                className="btn btn-ghost btn-sm"
                onClick={() => setTracks([])}
                style={{ color: "var(--error)" }}
              >
                <Trash2 size={14} />
                Clear All
              </button>
            )}
          </div>

          {/* Track list */}
          <div className="track-list">
            {tracks.map((t, i) => (
              <div
                key={t.file_id}
                className="track-item animate-in"
                style={{
                  cursor: "default",
                  animationDelay: `${i * 50}ms`,
                }}
              >
                <div className="track-icon">
                  <Music size={18} color="white" />
                </div>

                <div className="track-info">
                  <div className="track-name">{t.filename}</div>
                  <div className="track-meta">
                    <Clock size={10} style={{ marginRight: 4, opacity: 0.7 }} />
                    {formatDuration(t.duration)} • {t.file_id.slice(0, 8)}
                  </div>
                </div>

                {/* Analyzed badge */}
                {t.analyzed && (
                  <div style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                    padding: "4px 10px",
                    background: "rgba(34, 197, 94, 0.1)",
                    border: "1px solid rgba(34, 197, 94, 0.2)",
                    borderRadius: "var(--radius-full)",
                    color: "var(--success)",
                    fontSize: "0.72rem",
                    fontWeight: 600,
                    flexShrink: 0,
                  }}>
                    <CheckCircle size={12} />
                    Analyzed
                  </div>
                )}

                {/* Genre tag if analyzed */}
                {t.genre && (
                  <span style={{
                    padding: "4px 10px",
                    background: "rgba(139, 92, 246, 0.1)",
                    border: "1px solid rgba(139, 92, 246, 0.2)",
                    borderRadius: "var(--radius-full)",
                    fontSize: "0.72rem",
                    fontWeight: 500,
                    color: "var(--accent-secondary)",
                    textTransform: "capitalize",
                    flexShrink: 0,
                  }}>
                    {t.genre}
                  </span>
                )}

                {/* Remove button */}
                <button
                  className="btn btn-ghost btn-sm"
                  onClick={(e) => {
                    e.stopPropagation();
                    removeTrack(t.file_id);
                  }}
                  style={{
                    padding: 6,
                    color: "var(--text-dim)",
                    opacity: 0.6,
                  }}
                  aria-label="Remove track"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Empty state */}
      {tracks.length === 0 && !uploading && (
        <div className="card empty-state" style={{ marginTop: 40 }}>
          <div className="empty-state-icon">
            <Music size={28} color="var(--accent-primary)" />
          </div>
          <h3>No tracks yet</h3>
          <p>
            Upload your first audio file to get started with AI-powered analysis
          </p>
        </div>
      )}
    </div>
  );
}
