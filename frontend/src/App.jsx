import { useState } from "react";
import { UploadCloud, BarChart2, Sliders, Headphones, Wand2 } from "lucide-react";
import UploadPage from "./pages/UploadPage";
import AnalyzePage from "./pages/AnalyzePage";
import MixPage from "./pages/MixPage";
import BeatGeneratorPage from "./pages/BeatGeneratorPage";

const NAV = [
  { id: "upload",   label: "Upload",    Icon: UploadCloud },
  { id: "analyze",  label: "Analyze",   Icon: BarChart2   },
  { id: "mix",      label: "Mix",       Icon: Sliders     },
  { id: "generate", label: "Generator", Icon: Wand2       },
];

export default function App() {
  const [page,   setPage]   = useState("upload");
  const [tracks, setTracks] = useState([]);

  return (
    <div className="app">
      {/* ── Sidebar ─────────────────────────────────────── */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <div className="logo-icon">
            <Headphones size={22} color="white" />
          </div>
          <h1 className="gradient-text">AutoMixAI</h1>
        </div>

        <nav className="sidebar-nav">
          {NAV.map(({ id, label, Icon }) => (
            <button
              key={id}
              className={`nav-item ${page === id ? "active" : ""}`}
              onClick={() => setPage(id)}
            >
              <Icon size={16} />
              <span>{label}</span>
              {id === "upload" && tracks.length > 0 && (
                <span className="nav-badge">{tracks.length}</span>
              )}
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <p>ANN Beat Detection</p>
          <p style={{ opacity: 0.45, marginTop: 3 }}>FastAPI · TensorFlow · librosa</p>
        </div>
      </aside>

      {/* ── Main ────────────────────────────────────────── */}
      <main className="main-content">
        {page === "upload"   && <UploadPage   tracks={tracks} setTracks={setTracks} />}
        {page === "analyze"  && <AnalyzePage  tracks={tracks} setTracks={setTracks} />}
        {page === "mix"      && <MixPage      tracks={tracks} />}
        {page === "generate" && <BeatGeneratorPage />}
      </main>
    </div>
  );
}
