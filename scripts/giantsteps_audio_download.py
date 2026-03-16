#!/usr/bin/env python3
"""
GiantstepsKey Audio Downloader

Downloads 604 audio preview files from Beatport URL pattern.
Each track ID maps to a Beatport LOFI preview.

Usage:
    python giantsteps_audio_download.py [--output-dir ./audio] [--max-workers 5]
    
    Options:
        --output-dir    Directory to save audio files (default: ./audio)
        --max-workers   Number of parallel downloads (default: 5)
        --skip-existing Skip files that already exist (default: True)
        --verbose       Print detailed download info
"""

import os
import sys
import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
import urllib.error
from datetime import datetime
import time

# Configuration
BEATPORT_BASE_URL = "http://geo-samples.beatport.com/lofi"
GIANTSTEPS_REPO_DIR = Path("./giantsteps-key-dataset")
ANNOTATIONS_DIR = GIANTSTEPS_REPO_DIR / "annotations" / "key"


def load_track_ids(annotations_dir=ANNOTATIONS_DIR):
    """Extract track IDs from .key annotation files."""
    track_ids = []
    for key_file in sorted(annotations_dir.glob("*.key")):
        track_id = key_file.stem  # Remove .key extension
        track_ids.append(track_id)
    return track_ids


def download_file(url, output_path, timeout=30, max_retries=3):
    """Download a single file with retries."""
    for attempt in range(max_retries):
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if already exists
            if output_path.exists():
                return f"EXISTS", output_path.stat().st_size
            
            # Download
            request = urllib.request.Request(
                url,
                headers={'User-Agent': 'AutoMixAI-DataDownloader/1.0'}
            )
            
            with urllib.request.urlopen(request, timeout=timeout) as response:
                with open(output_path, 'wb') as f:
                    file_size = 0
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        file_size += len(chunk)
            
            return f"OK", file_size
        
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return f"NOT_FOUND (404)", 0
            elif attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                return f"ERROR (HTTP {e.code})", 0
        
        except urllib.error.URLError as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                return f"ERROR (Network: {str(e)[:30]})", 0
        
        except Exception as e:
            return f"ERROR ({type(e).__name__}: {str(e)[:30]})", 0
    
    return "FAILED", 0


def main():
    parser = argparse.ArgumentParser(
        description="Download GiantstepsKey audio files from Beatport",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python giantsteps_audio_download.py --output-dir ./audio --max-workers 10
    
    Downloads 604 MP3 files with 10 parallel connections.
    Total size: ~850 MB, estimated time: 30-60 minutes (depends on bandwidth)
        """
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./audio",
        help="Output directory for audio files (default: ./audio)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Number of parallel download threads (default: 5)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip files that already exist (default: True)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed download information"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load track IDs
    if not ANNOTATIONS_DIR.exists():
        print(f"❌ Error: {ANNOTATIONS_DIR} not found")
        print(f"   Please run from the project root or clone the dataset first:")
        print(f"   git clone https://github.com/GiantSteps/giantsteps-key-dataset.git")
        sys.exit(1)
    
    track_ids = load_track_ids()
    print(f"📊 Found {len(track_ids)} tracks to download")
    print(f"🎯 Output directory: {output_dir.resolve()}")
    print(f"🔗 Base URL: {BEATPORT_BASE_URL}")
    print(f"⚙️  Parallel workers: {args.max_workers}")
    print()
    
    # Download statistics
    stats = {
        'ok': 0,
        'exists': 0,
        'not_found': 0,
        'error': 0,
        'total_size': 0,
        'errors': []
    }
    
    # Download files in parallel
    print(f"⏳ Starting downloads at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    print()
    
    downloaded = 0
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {}
        
        for i, track_id in enumerate(track_ids, 1):
            url = f"{BEATPORT_BASE_URL}/{track_id}.mp3"
            output_path = output_dir / f"{track_id}.mp3"
            
            future = executor.submit(download_file, url, output_path)
            futures[future] = (i, track_id, url)
        
        for future in as_completed(futures):
            idx, track_id, url = futures[future]
            status, size = future.result()
            
            if status == "OK":
                stats['ok'] += 1
                downloaded += 1
                stats['total_size'] += size
                if args.verbose:
                    print(f"  [{idx:3d}/{len(track_ids)}] ✅ {track_id:20s} ({size/1024/1024:.1f} MB)")
            
            elif status == "EXISTS":
                stats['exists'] += 1
                if args.verbose:
                    print(f"  [{idx:3d}/{len(track_ids)}] ⏭️  {track_id:20s} (already exists)")
            
            elif status == "NOT_FOUND (404)":
                stats['not_found'] += 1
                stats['errors'].append((track_id, "Not found on Beatport"))
                # if args.verbose:
                #     print(f"  [{idx:3d}/{len(track_ids)}] ❌ {track_id:20s} (404 Not found)")
            
            else:
                stats['error'] += 1
                stats['errors'].append((track_id, status))
                # if args.verbose:
                #     print(f"  [{idx:3d}/{len(track_ids)}] ❌ {track_id:20s} ({status})")
            
            # Print progress every 50 tracks
            if idx % 50 == 0:
                elapsed = time.time() - start_time
                rate = idx / elapsed
                remaining = (len(track_ids) - idx) / rate if rate > 0 else 0
                print(f"  Progress: {idx}/{len(track_ids)} ({100*idx//len(track_ids)}%) "
                      f"| {downloaded} downloaded | "
                      f"ETA: {int(remaining)} sec")
    
    elapsed = time.time() - start_time
    
    # Summary
    print()
    print("=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"✅ Successfully downloaded: {stats['ok']:3d}")
    print(f"⏭️  Already existed:        {stats['exists']:3d}")
    print(f"❌ Not found (404):        {stats['not_found']:3d}")
    print(f"⚠️  Errors:                {stats['error']:3d}")
    print(f"---")
    print(f"📊 Total files:            {len(track_ids)}")
    print(f"💾 Total size:             {stats['total_size']/1024/1024:.1f} MB")
    print(f"⏱️  Elapsed time:           {elapsed/60:.1f} minutes")
    
    if stats['ok'] > 0:
        avg_speed = (stats['total_size'] / elapsed) / 1024 / 1024
        print(f"🚀 Average speed:          {avg_speed:.1f} MB/s")
    
    if stats['errors']:
        print(f"\nFailed tracks ({len(stats['errors'])}):")
        for track_id, error in stats['errors'][:10]:  # Show first 10
            print(f"  • {track_id}: {error}")
        if len(stats['errors']) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more")
    
    print()
    print(f"Files saved to: {output_dir.resolve()}")
    print("=" * 70)
    
    # Return exit code based on success
    if stats['ok'] == len(track_ids):
        print("✅ All files downloaded successfully!")
        return 0
    elif stats['ok'] + stats['exists'] == len(track_ids):
        print("✅ All files ready (some already existed)")
        return 0
    else:
        missing = stats['not_found'] + stats['error']
        print(f"⚠️  {missing} files could not be downloaded")
        return 1


if __name__ == "__main__":
    sys.exit(main())
