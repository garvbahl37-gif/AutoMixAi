import sys
sys.path.insert(0, '.')
import yt_dlp
import os
import json
from dejavu import Dejavu

djv = Dejavu(dburl="sqlite:///fingerprints.db")

# ============================================================
# ADD ARTIST NAMES — scraper will find top songs automatically
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
    # Add more artists...
]

SONGS_PER_ARTIST = 25        # 25 songs x 200 artists = 5000 total
BATCH_SIZE = 500             # 500 songs per daily run
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
    """Search YouTube for each artist and build a queue of video URLs."""
    if progress["queue"]:
        print(f"📋 Resuming existing queue ({len(progress['queue'])} songs left)")
        return progress

    print("🔍 Building song queue from artist names...")
    queue = []
    done_set = set(progress["done"])

    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
    }

    for artist in ARTISTS:
        search_query = f"ytsearch{SONGS_PER_ARTIST}:{artist} songs"
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                result = ydl.extract_info(search_query, download=False)
                entries = result.get('entries', [])
                for entry in entries:
                    url = f"https://www.youtube.com/watch?v={entry['id']}"
                    title = entry.get('title', artist)
                    if url not in done_set:
                        queue.append({"url": url, "title": title, "artist": artist})
            print(f"  ✅ Found songs for: {artist}")
        except Exception as e:
            print(f"  ❌ Failed to search {artist}: {e}")

    progress["queue"] = queue
    save_progress(progress)
    print(f"📋 Total queue built: {len(queue)} songs")
    return progress

def download_and_fingerprint(url, title):
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

        wav_path = '/tmp/song.wav'
        if os.path.exists(wav_path):
            djv.fingerprint_file(wav_path, song_name=title)
            print(f"  ✅ Fingerprinted: {title}")
            os.remove(wav_path)
            return True
        else:
            print(f"  ❌ WAV not found: {title}")
            return False
    except Exception as e:
        print(f"  ❌ Failed: {title} → {e}")
        # Clean up if file exists
        if os.path.exists('/tmp/song.wav'):
            os.remove('/tmp/song.wav')
        return False

def run_batch():
    progress = load_progress()
    progress = build_queue(progress)

    queue = progress["queue"]
    done = progress["done"]

    if not queue:
        print("🎉 All songs have been fingerprinted!")
        return

    # Take next BATCH_SIZE from queue
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

    # Update progress
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
