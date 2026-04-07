import { useState } from "react";
import { UploadCloud, BarChart2, Sliders, Wand2, Headphones, Sparkles, Menu, X, Mic } from "lucide-react";
import UploadPage from "./pages/UploadPage";
import AnalyzePage from "./pages/AnalyzePage";
import MixPage from "./pages/MixPage";
import BeatGeneratorPage from "./pages/BeatGeneratorPage";
import ShazamPage from "./pages/ShazamPage";

const NAV = [
  { id: "upload",   label: "Upload",    Icon: UploadCloud, desc: "Import audio files" },
  { id: "analyze",  label: "Analyze",   Icon: BarChart2,   desc: "AI audio analysis" },
  { id: "mix",      label: "Mix",       Icon: Sliders,     desc: "DJ mixing engine" },
  { id: "generate", label: "Generator", Icon: Wand2,       desc: "Beat synthesis" },
  { id: "shazam",   label: "Shazam",    Icon: Mic,         desc: "Song recognition" },
];

export default function App() {
  const [page, setPage] = useState("upload");
  const [tracks, setTracks] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="app">
      {/* Mobile menu button */}
      <button
        className="btn btn-icon btn-ghost"
        onClick={() => setSidebarOpen(!sidebarOpen)}
        style={{
          position: "fixed",
          top: 16,
          left: 16,
          zIndex: 400,
          display: "none",
        }}
        aria-label="Toggle menu"
      >
        {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
      </button>

      {/* Sidebar */}
      <aside className={`sidebar ${sidebarOpen ? "open" : ""}`}>
        {/* Logo */}
        <div className="sidebar-logo">
          <div className="logo-icon">
            <Headphones size={24} color="white" strokeWidth={2.5} />
          </div>
          <div>
            <h1 className="gradient-text">AutoMixAI</h1>
            <p style={{
              fontSize: "0.65rem",
              color: "var(--text-dim)",
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              marginTop: 2,
            }}>
              AI-Powered Mixing
            </p>
          </div>
        </div>

        {/* Navigation */}
        <nav className="sidebar-nav">
          {NAV.map(({ id, label, Icon, desc }) => (
            <button
              key={id}
              className={`nav-item ${page === id ? "active" : ""}`}
              onClick={() => {
                setPage(id);
                setSidebarOpen(false);
              }}
            >
              <Icon size={18} strokeWidth={page === id ? 2.5 : 2} />
              <div style={{ flex: 1, textAlign: "left" }}>
                <span style={{ display: "block" }}>{label}</span>
                <span style={{
                  fontSize: "0.68rem",
                  color: "var(--text-dim)",
                  display: page === id ? "block" : "none",
                }}>
                  {desc}
                </span>
              </div>
              {id === "upload" && tracks.length > 0 && (
                <span className="nav-badge">{tracks.length}</span>
              )}
            </button>
          ))}
        </nav>

        {/* Footer */}
        <div className="sidebar-footer">
          <div style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            marginBottom: 8,
          }}>
            <Sparkles size={12} color="var(--accent-secondary)" />
            <span style={{ color: "var(--text-muted)", fontSize: "0.72rem" }}>
              Powered by AI
            </span>
          </div>
          <p>
            <span style={{ color: "var(--accent-secondary)" }}>TensorFlow</span> · librosa · FastAPI
          </p>
          <p style={{ marginTop: 4, opacity: 0.6 }}>
            v2.0 — Beat Detection + Genre + Tags
          </p>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        {page === "upload" && (
          <UploadPage tracks={tracks} setTracks={setTracks} />
        )}
        {page === "analyze" && (
          <AnalyzePage tracks={tracks} setTracks={setTracks} />
        )}
        {page === "mix" && (
          <MixPage tracks={tracks} />
        )}
        {page === "generate" && (
          <BeatGeneratorPage />
        )}
        {page === "shazam" && (
          <ShazamPage />
        )}
      </main>

      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0,0,0,0.5)",
            zIndex: 99,
            display: "none",
          }}
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <style>{`
        @media (max-width: 960px) {
          .app > button:first-child {
            display: flex !important;
          }
          .app > div:last-child {
            display: block !important;
          }
        }
      `}</style>
    </div>
  );
}
