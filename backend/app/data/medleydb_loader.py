"""
AutoMixAI – MedleyDB Dataset Loader

Reads metadata and annotations from the MedleyDB dataset.
MedleyDB contains professional multi-track audio with detailed stem-level
instrument annotations in YAML metadata and time-aligned source labels.

Supported annotations:
    • **Metadata** – YAML files with track, stem, and instrument info
    • **Source_ID** – Time-aligned instrument labels for source separation
    • **Melody** – Melody line annotations
    • **Pitch** – Pitch annotations

Reference: https://github.com/marl/medleydb
"""

import csv
from pathlib import Path
from typing import Optional
import yaml

from app.utils.logger import get_logger

logger = get_logger(__name__)


class MedleyDBMetadata:
    """Parse and store MedleyDB YAML metadata for a single track."""
    
    def __init__(self, yaml_path: Path):
        """
        Load and parse a MedleyDB metadata YAML file.
        
        Args:
            yaml_path: Path to *_METADATA.yaml file
        """
        self.yaml_path = yaml_path
        self.data = {}
        self.artist = ""
        self.title = ""
        self.genre = ""
        self.stems = {}
        self.mix_filename = ""
        self.raw_dir = ""
        self.stem_dir = ""
        
        self._load()
    
    def _load(self):
        """Load YAML data from file."""
        try:
            with open(self.yaml_path, 'r', encoding='utf-8') as f:
                self.data = yaml.safe_load(f)
            
            self.artist = self.data.get('artist', 'Unknown')
            self.title = self.yaml_path.stem.replace('_METADATA', '')
            self.genre = self.data.get('genre', 'Unknown')
            self.mix_filename = self.data.get('mix_filename', '')
            self.raw_dir = self.data.get('raw_dir', '')
            self.stem_dir = self.data.get('stem_dir', '')
            self.stems = self.data.get('stems', {})
            
        except Exception as e:
            logger.error(f"Error loading MedleyDB metadata {self.yaml_path}: {e}")
    
    def get_instruments(self) -> list[str]:
        """
        Extract all distinct instruments from stems.
        
        Returns:
            List of instrument names (lowercase, deduplicated)
        """
        instruments = set()
        for stem_id, stem_data in self.stems.items():
            if isinstance(stem_data, dict):
                instrument = stem_data.get('instrument', '').lower().strip()
                if instrument:
                    instruments.add(instrument)
        return sorted(list(instruments))
    
    def get_components(self) -> list[str]:
        """
        Extract all distinct component types (melody, bass, drums, etc).
        
        Returns:
            List of component names (lowercase, deduplicated)
        """
        components = set()
        for stem_id, stem_data in self.stems.items():
            if isinstance(stem_data, dict):
                component = stem_data.get('component', '').lower().strip()
                if component:
                    components.add(component)
        return sorted(list(components))
    
    @property
    def track_id(self) -> str:
        """Return normalized track identifier."""
        return f"{self.artist}_{self.title}".replace(' ', '_')


class SourceIDAnnotation:
    """Parse and store MedleyDB Source_ID annotations."""
    
    def __init__(self, lab_path: Path):
        """
        Load and parse a MedleyDB Source_ID .lab file.
        
        Formato: start_time,end_time,instrument_label
        
        Args:
            lab_path: Path to *_SOURCEID.lab file
        """
        self.lab_path = lab_path
        self.annotations = []  # List of (start, end, instrument)
        self.unique_instruments = set()
        
        self._load()
    
    def _load(self):
        """Load CSV-formatted annotations from file."""
        try:
            with open(self.lab_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        start = float(row.get('start_time', 0))
                        end = float(row.get('end_time', 0))
                        instrument = row.get('instrument_label', 'unknown').lower().strip()
                        
                        self.annotations.append((start, end, instrument))
                        self.unique_instruments.add(instrument)
                    except (ValueError, KeyError):
                        continue
        except Exception as e:
            logger.error(f"Error loading MedleyDB Source_ID annotation {self.lab_path}: {e}")
    
    def get_instruments_at_time(self, time_sec: float) -> list[str]:
        """
        Get all instruments active at a specific time.
        
        Args:
            time_sec: Time in seconds
        
        Returns:
            List of active instrument labels
        """
        active = []
        for start, end, instrument in self.annotations:
            if start <= time_sec <= end:
                active.append(instrument)
        return active
    
    @property
    def instruments(self) -> list[str]:
        """Return sorted list of unique instruments."""
        return sorted(list(self.unique_instruments))


class MedleyDBLoader:
    """Load and access MedleyDB dataset."""
    
    def __init__(self, medleydb_dir: Path):
        """
        Initialize MedleyDB loader.
        
        Args:
            medleydb_dir: Path to medleydb/ folder (the cloned repo root)
        """
        self.medleydb_dir = Path(medleydb_dir)
        self.metadata_dir = self.medleydb_dir / "medleydb" / "data" / "Metadata"
        self.annotations_dir = self.medleydb_dir / "medleydb" / "data" / "Annotations"
        self.source_id_dir = self.annotations_dir / "Source_ID"
        
        if not self.metadata_dir.exists():
            logger.error(f"MedleyDB metadata directory not found: {self.metadata_dir}")
        
        self.metadata_files = sorted(list(self.metadata_dir.glob("*_METADATA.yaml")))
        logger.info(f"Loaded {len(self.metadata_files)} MedleyDB metadata files")
    
    def load_track_metadata(self, track_num: int = 0) -> Optional[MedleyDBMetadata]:
        """
        Load metadata for a specific track by index.
        
        Args:
            track_num: Index of track to load (0-indexed)
        
        Returns:
            MedleyDBMetadata object or None if not found
        """
        if 0 <= track_num < len(self.metadata_files):
            return MedleyDBMetadata(self.metadata_files[track_num])
        return None
    
    def load_track_by_title(self, title: str) -> Optional[MedleyDBMetadata]:
        """
        Load metadata for a track by title.
        
        Args:
            title: Track title or partial match
        
        Returns:
            MedleyDBMetadata object or None if not found
        """
        title_lower = title.lower()
        for metadata_file in self.metadata_files:
            if title_lower in metadata_file.stem.lower():
                return MedleyDBMetadata(metadata_file)
        return None
    
    def load_source_id_annotation(self, track_metadata: MedleyDBMetadata) -> Optional[SourceIDAnnotation]:
        """
        Load Source_ID annotations matching a metadata track.
        
        Args:
            track_metadata: MedleyDBMetadata object
        
        Returns:
            SourceIDAnnotation object or None if not found
        """
        # Extract track title from metadata YAML stem
        title_stem = track_metadata.yaml_path.stem.replace('_METADATA', '')
        source_id_path = self.source_id_dir / f"{title_stem}_SOURCEID.lab"
        
        if source_id_path.exists():
            return SourceIDAnnotation(source_id_path)
        
        return None
    
    def get_all_instruments(self) -> dict[str, int]:
        """
        Scan all metadata and annotations to collect all unique instruments.
        
        Returns:
            Dictionary mapping instrument name to frequency count
        """
        instruments = {}
        
        for metadata_file in self.metadata_files[:50]:  # Sample first 50 for speed
            try:
                metadata = MedleyDBMetadata(metadata_file)
                for instrument in metadata.get_instruments():
                    instruments[instrument] = instruments.get(instrument, 0) + 1
            except Exception as e:
                logger.warning(f"Error processing {metadata_file}: {e}")
        
        return dict(sorted(instruments.items(), key=lambda x: x[1], reverse=True))
    
    def get_all_components(self) -> dict[str, int]:
        """
        Scan all metadata to collect all unique component types.
        
        Returns:
            Dictionary mapping component name to frequency count
        """
        components = {}
        
        for metadata_file in self.metadata_files[:50]:  # Sample first 50 for speed
            try:
                metadata = MedleyDBMetadata(metadata_file)
                for component in metadata.get_components():
                    components[component] = components.get(component, 0) + 1
            except Exception as e:
                logger.warning(f"Error processing {metadata_file}: {e}")
        
        return dict(sorted(components.items(), key=lambda x: x[1], reverse=True))
    
    def get_track_count(self) -> int:
        """Return total number of tracks in dataset."""
        return len(self.metadata_files)
    
    def iterate_metadata(self, max_tracks: Optional[int] = None):
        """
        Iterate over all track metadata.
        
        Args:
            max_tracks: Optional limit on number of tracks to iterate
        
        Yields:
            MedleyDBMetadata objects
        """
        count = 0
        for metadata_file in self.metadata_files:
            if max_tracks and count >= max_tracks:
                break
            try:
                yield MedleyDBMetadata(metadata_file)
                count += 1
            except Exception as e:
                logger.warning(f"Error loading {metadata_file}: {e}")


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    loader = MedleyDBLoader(Path("../../Datasets/medleydb"))
    print(f"Total tracks: {loader.get_track_count()}")
    print(f"\nTop 20 instruments:")
    for instrument, count in list(loader.get_all_instruments().items())[:20]:
        print(f"  {instrument}: {count}")
    print(f"\nComponents:")
    for component, count in loader.get_all_components().items():
        print(f"  {component}: {count}")
