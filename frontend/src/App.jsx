import { useState } from "react";
import { UploadCloud, BarChart2, Sliders, Headphones } from "lucide-react";
import UploadPage from "./pages/UploadPage";
import AnalyzePage from "./pages/AnalyzePage";
import MixPage from "./pages/MixPage";

const NAV = [
  { id: "upload", label: "Upload", icon: <UploadCloud size={18} /> },
  { id: "analyze", label: "Analyze", icon: <BarChart2 size={18} /> },
  { id: "mix", label: "Mix", icon: <Sliders size={18} /> },
];

export default function App() {
  const [page, setPage] = useState("upload");
  const [tracks, setTracks] = useState([]);

  return (
    <div className="app">
      {/* ── Sidebar ────────────────────────────────── */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <div className="logo-icon"><Headphones size={24} color="white" /></div>
          <h1 className="gradient-text">AutoMixAI</h1>
        </div>

        <nav className="sidebar-nav">
          {NAV.map((item) => (
            <button
              key={item.id}
              className={`nav-item ${page === item.id ? "active" : ""}`}
              onClick={() => setPage(item.id)}
            >
              <span className="icon">{item.icon}</span>
              {item.label}
              {item.id === "upload" && tracks.length > 0 && (
                <span style={{
                  marginLeft: "auto",
                  background: "var(--accent-primary)",
                  color: "white",
                  fontSize: "0.7rem",
                  padding: "2px 8px",
                  borderRadius: 10,
                  fontWeight: 600,
                }}>
                  {tracks.length}
                </span>
              )}
            </button>
          ))}
        </nav>

        {/* Footer */}
        <div style={{
          padding: "16px 8px",
          borderTop: "1px solid var(--border-color)",
          marginTop: "auto",
        }}>
          <p style={{
            color: "var(--text-muted)",
            fontSize: "0.7rem",
            textAlign: "center",
          }}>
            Powered by ANN Beat Detection
          </p>
          <p style={{
            color: "var(--text-muted)",
            fontSize: "0.65rem",
            textAlign: "center",
            marginTop: 4,
          }}>
            FastAPI • TensorFlow • librosa
          </p>
        </div>
      </aside>

      {/* ── Main Content ──────────────────────────── */}
      <main className="main-content">
        {page === "upload" && <UploadPage tracks={tracks} setTracks={setTracks} />}
        {page === "analyze" && <AnalyzePage tracks={tracks} setTracks={setTracks} />}
        {page === "mix" && <MixPage tracks={tracks} />}
      </main>
    </div>
  );
}
