"""
AutoMixAI - Beat Dataset Downloader
=====================================
Downloads freely available beat-annotated datasets for training AutoMixAI v2.

IMPORTANT: Run with Python 3.11 (venv311), NOT system Python 3.14!

    backend\\venv311\\Scripts\\python.exe download_hainsworth.py

What this downloads:
  [1] SMC Beat Corpus (217 tracks, REAL human beat annotations, ~600 MB)
      Source: http://smc.inesc-porto.pt/research/smc_mirex2012_dataset/
  [2] HJDB Hainsworth+Jonsson annotations (annotations only, no audio)
      Available from madmom annotation sets

Why not mirdata Hainsworth?
  The Hainsworth audio was hosted at QMUL and is no longer publicly
  downloadable. mirdata's DOWNLOAD_INFO confirms this and asks for
  manual download. The SMC Beat Corpus is a better alternative - it
  has 217 tracks with ground-truth annotations and is freely available.
"""

import argparse
import sys
import warnings
import zipfile
import shutil
from pathlib import Path

warnings.filterwarnings("ignore")

# UTF-8 output for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Python version check
if sys.version_info >= (3, 14):
    print("ERROR: Run with Python 3.11:")
    print("  backend\\venv311\\Scripts\\python.exe download_hainsworth.py")
    sys.exit(1)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Download beat-annotated datasets")
parser.add_argument(
    "--dest",
    type=str,
    default=str(Path(__file__).parent / "Datasets"),
    help="Root datasets directory (default: ./Datasets)",
)
args = parser.parse_args()

DEST = Path(args.dest)
DEST.mkdir(parents=True, exist_ok=True)
print(f"[AutoMixAI] Datasets root -> {DEST}\n")

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests")
    sys.exit(1)

# ── Helper ────────────────────────────────────────────────────────────────────
def download_file(url: str, dest_path: Path, desc: str = "") -> bool:
    """Download a file with progress. Returns True on success."""
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {desc}: {pct:.1f}% ({downloaded//1024//1024} MB)", end="", flush=True)
        print()
        return True
    except Exception as e:
        print(f"\n  ERROR downloading {url}: {e}")
        return False


# ── SMC Beat Corpus ───────────────────────────────────────────────────────────
print("=" * 60)
print("Downloading SMC Beat Corpus (217 tracks, real annotations)")
print("=" * 60)

SMC_DIR = DEST / "smc_beat_corpus"
SMC_DIR.mkdir(exist_ok=True)

# SMC dataset is available from INESC-Porto via direct link
SMC_URL = "https://zenodo.org/record/1250905/files/SMC_MIREX.zip"
SMC_ZIP = SMC_DIR / "SMC_MIREX.zip"

if not SMC_ZIP.exists():
    print(f"  Downloading from Zenodo...")
    ok = download_file(SMC_URL, SMC_ZIP, "SMC Beat Corpus")
    if ok:
        print(f"  Extracting...")
        try:
            with zipfile.ZipFile(SMC_ZIP) as z:
                z.extractall(SMC_DIR)
            print(f"  Extracted to {SMC_DIR}")
        except Exception as e:
            print(f"  Extraction error: {e}")
    else:
        print("\n  SMC download failed. Try manually:")
        print(f"  https://zenodo.org/record/1250905")
        print(f"  Place extracted files in: {SMC_DIR}")
else:
    print(f"  Already downloaded: {SMC_ZIP}")

# Count what we got
wav_files = list(SMC_DIR.rglob("*.wav"))
ann_files = list(SMC_DIR.rglob("*.txt")) + list(SMC_DIR.rglob("*.beats"))
print(f"\n  SMC result: {len(wav_files)} audio files, {len(ann_files)} annotation files")


# ── Hainsworth info ───────────────────────────────────────────────────────────
print()
print("=" * 60)
print("Hainsworth Dataset - Manual Download Required")
print("=" * 60)
print()
print("  The Hainsworth audio files are no longer freely available online.")
print("  If you can obtain the dataset, place it as:")
print()
print(f"  {DEST / 'hainsworth' / 'audio'}          (222 WAV files)")
print(f"  {DEST / 'hainsworth' / 'annotations'}    (222 .beats files)")
print()
print("  Alternative: The Kaggle v2 notebook works great with just")
print("               GTZAN + SMC Beat Corpus!")
print()

# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("Summary")
print("=" * 60)
print(f"  SMC Beat Corpus -> {SMC_DIR} ({len(wav_files)} WAV files)")
print()
print("Next steps:")
print("  1. Upload 'Datasets/smc_beat_corpus' as a Kaggle Dataset")
print("  2. GTZAN is at: andradaolteanu/gtzan-dataset-music-genre-classification")
print("  3. Run kaggle/AutoMixAI_Training_v2.ipynb")
print("     (Add an SMC section - see instructions below)")
print()
print("  To use SMC in the v2 notebook, add this section:")
print("""
  SMC_DIR = Path('/kaggle/input/smc-beat-corpus')
  audio_files = list(SMC_DIR.rglob('*.wav'))
  for path in audio_files:
      y, sr = load_audio(path)
      feats = extract_features(y, sr)
      # Load annotation: same name, .beats extension
      ann_path = path.with_suffix('.beats')
      if ann_path.exists():
          beat_times = np.loadtxt(ann_path)[:, 0]  # first column = timestamps
          labels = annotation_beat_labels(beat_times, sr, feats.shape[0])
      else:
          labels = pseudo_beat_labels(y, sr, feats.shape[0])
      all_X.append(feats); all_y.append(labels)
""")