"""
Disambiguation cache for storing and retrieving frequent term disambiguations.

This module provides file-based caching to reduce redundant LLM calls.
"""

import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime


class DisambiguationCache:
    """Cache for storing disambiguation results."""

    def __init__(self, cache_path: Path = Path("./data/cache/disambiguations.json")):
        """
        Initialize disambiguation cache.

        Args:
            cache_path: Path to cache file
        """
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, Dict] = {}
        self.hit_count = 0
        self.miss_count = 0

        # Load existing cache
        self._load()

    def get(self, term: str, context: str = "") -> Optional[str]:
        """
        Get cached disambiguation for a term.

        Args:
            term: Ambiguous term to disambiguate
            context: Context string (optional, for context-aware caching)

        Returns:
            Disambiguated term if cached, None otherwise
        """
        cache_key = self._make_key(term, context)

        if cache_key in self.cache:
            self.hit_count += 1
            entry = self.cache[cache_key]
            entry["access_count"] = entry.get("access_count", 0) + 1
            entry["last_accessed"] = datetime.utcnow().isoformat()
            return entry["disambiguation"]
        else:
            self.miss_count += 1
            return None

    def set(self, term: str, disambiguation: str, context: str = "", confidence: float = 1.0):
        """
        Store disambiguation in cache.

        Args:
            term: Ambiguous term
            disambiguation: Disambiguated term
            context: Context string (optional)
            confidence: Confidence score for this disambiguation
        """
        cache_key = self._make_key(term, context)

        self.cache[cache_key] = {
            "term": term,
            "disambiguation": disambiguation,
            "context": context,
            "confidence": confidence,
            "created_at": datetime.utcnow().isoformat(),
            "last_accessed": datetime.utcnow().isoformat(),
            "access_count": 1,
        }

        # Save to disk
        self._save()

    def get_hit_rate(self) -> float:
        """
        Calculate cache hit rate.

        Returns:
            Hit rate as percentage (0-1)
        """
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0

    def get_stats(self) -> Dict:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats
        """
        return {
            "total_entries": len(self.cache),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": self.get_hit_rate(),
        }

    def clear(self):
        """Clear all cache entries."""
        self.cache = {}
        self.hit_count = 0
        self.miss_count = 0
        self._save()

    def _make_key(self, term: str, context: str) -> str:
        """Create cache key from term and context."""
        # Simple key: term + context hash
        if context:
            return f"{term}:{hash(context)}"
        return term

    def _load(self):
        """Load cache from disk."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r") as f:
                    self.cache = json.load(f)
                print(f"✓ Loaded {len(self.cache)} cached disambiguations")
            except Exception as e:
                print(f"Warning: Failed to load cache: {e}")
                self.cache = {}

    def _save(self):
        """Save cache to disk."""
        try:
            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
