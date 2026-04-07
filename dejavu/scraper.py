import sys
import os

# Add parent directory to path so we can import dejavu
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yt_dlp
import json
from dejavu import Dejavu

djv = Dejavu(dburl="sqlite:///fingerprints.db")

# ============================================================
# PLAYLISTS — songs will be scraped from these directly
# ============================================================
PLAYLISTS = [
    "https://www.youtube.com/playlist?list=PLr7xQC-cXWL9EZ3dqpu8E_Xf_4nhS6xEJ",
    "https://www.youtube.com/playlist?list=PLxA687tYuMWjrFhZTNBtk13YUL2TkwUnU",
]

# ============================================================
# ARTISTS — scraper will find top songs automatically
# ============================================================
ARTISTS = [
    "Arijit Singh",
    "Shreya Ghoshal",
    "Sonu Nigam",
    "Lata Mangeshkar",
    "Kishore Kumar",
    "Mohammed Rafi",
    "Udit Narayan",
    "Alka Yagnik",
    "Kumar Sanu",
    "Sunidhi Chauhan",
    "Neha Kakkar",
    "Atif Aslam",
    "Rahat Fateh Ali Khan",
    "Armaan Malik",
    "Jubin Nautiyal",
    "Darshan Raval",
    "Taylor Swift",
    "Ed Sheeran",
    "The Weeknd",
    "Drake",
]

SONGS_PER_ARTIST = 25
BATCH_SIZE = 50
PROGRESS_FILE = "scraper_progress.json"

# ============================================================

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"done": [], "queue": []}

def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)

def build_queue(progress):
    if progress["queue"]:
        print(f"📋 Resuming existing queue ({len(progress['queue'])} songs left)")
        return progress

    print("🔍 Building song queue from playlists and artist names...")
    queue = []
    done_set = set(progress["done"])

    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
    }

    # ── Playlists ──────────────────────────────────────────
    for playlist_url in PLAYLISTS:
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                result = ydl.extract_info(playlist_url, download=False)
                playlist_title = result.get('title', playlist_url)
                entries = result.get('entries', [])
                added = 0
                for entry in entries:
                    if not entry or not entry.get('id'):
                        continue
                    url = f"https://www.youtube.com/watch?v={entry['id']}"
                    title = entry.get('title', 'Unknown')
                    if url not in done_set:
                        queue.append({"url": url, "title": title, "artist": "playlist"})
                        added += 1
                print(f"  ✅ Playlist '{playlist_title}': {added} songs added")
        except Exception as e:
            print(f"  ❌ Failed to load playlist {playlist_url}: {e}")

    # ── Artists ────────────────────────────────────────────
    for artist in ARTISTS:
        search_query = f"ytsearch{SONGS_PER_ARTIST}:{artist} songs"
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                result = ydl.extract_info(search_query, download=False)
                entries = result.get('entries', [])
                added = 0
                for entry in entries:
                    if not entry or not entry.get('id'):
                        continue
                    url = f"https://www.youtube.com/watch?v={entry['id']}"
                    title = entry.get('title', artist)
                    if url not in done_set:
                        queue.append({"url": url, "title": title, "artist": artist})
                        added += 1
                print(f"  ✅ Artist '{artist}': {added} songs added")
        except Exception as e:
            print(f"  ❌ Failed to search {artist}: {e}")

    progress["queue"] = queue
    save_progress(progress)
    print(f"\n📋 Total queue built: {len(queue)} songs")
    return progress

def download_and_fingerprint(url, title):
    wav_path = '/tmp/song.wav'

    # Clean up any leftover file from previous run
    if os.path.exists(wav_path):
        os.remove(wav_path)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': '/tmp/song.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)

        if os.path.exists(wav_path):
            djv.fingerprint_file(wav_path, song_name=title)
            print(f"  ✅ Fingerprinted: {title}")
            os.remove(wav_path)
            return True
        else:
            print(f"  ❌ WAV not found after download: {title}")
            return False

    except Exception as e:
        print(f"  ❌ Failed: {title} → {e}")
        if os.path.exists(wav_path):
            os.remove(wav_path)
        return False

def run_batch():
    progress = load_progress()
    progress = build_queue(progress)

    queue = progress["queue"]
    done = progress["done"]

    if not queue:
        print("🎉 All songs have been fingerprinted!")
        return

    batch = queue[:BATCH_SIZE]
    remaining = queue[BATCH_SIZE:]

    print(f"\n🎵 Running batch: {len(batch)} songs")
    print(f"📊 Total done so far: {len(done)} | Remaining after this batch: {len(remaining)}\n")

    success = 0
    failed = 0

    for i, song in enumerate(batch, 1):
        print(f"[{i}/{len(batch)}] {song['artist']} — {song['title']}")
        result = download_and_fingerprint(song['url'], song['title'])
        if result:
            success += 1
            done.append(song['url'])
        else:
            failed += 1

    progress["queue"] = remaining
    progress["done"] = done
    save_progress(progress)

    print(f"\n📊 Batch complete!")
    print(f"  ✅ Success: {success}")
    print(f"  ❌ Failed:  {failed}")
    print(f"  📋 Songs remaining in queue: {len(remaining)}")
    print(f"  🎵 Total fingerprinted: {len(done)}")

if __name__ == "__main__":
    run_batch()
