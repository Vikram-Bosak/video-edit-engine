"""
Memory Agent for Video Editing Engine.

Loads previous editing history, filters duplicates, recommends styling priorities,
and stores metrics of successfully completed edits.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger("memory_agent")

class MemoryAgent:
    """Manages project editing memory, style priorities, duplicate protection, and history updates."""

    def __init__(self, workspace_root: str):
        self.workspace_root = workspace_root
        self.memory_dir = os.path.join(workspace_root, "memory")
        self.memory_path = os.path.join(self.memory_dir, "project_memory.json")
        os.makedirs(self.memory_dir, exist_ok=True)
        self.memory_data = self._load_memory()

    def _load_memory(self) -> Dict[str, Any]:
        """Loads memory JSON or initializes default structure if missing."""
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load memory file: {e}")
        
        # Default base memory structure
        return {
            "video_history": [],
            "style_library": {
                "funny": {"name": "Funny Zoom", "views": 0, "priority": "High", "success_count": 0},
                "sports": {"name": "Sports Cinematic", "views": 0, "priority": "Medium", "success_count": 0},
                "meme": {"name": "Fast Meme", "views": 0, "priority": "High", "success_count": 0}
            },
            "duplicate_hashes": [],
            "settings": {"default_transition": "tv_static", "default_caption_style": "Pop"}
        }

    def _save_memory(self):
        """Saves current memory data back to the JSON file."""
        try:
            with open(self.memory_path, "w", encoding="utf-8") as f:
                json.dump(self.memory_data, f, indent=2)
            logger.info("Memory database successfully saved.")
        except Exception as e:
            logger.error(f"Failed to save memory data: {e}")

    def is_duplicate(self, video_url: str) -> bool:
        """Checks if a video URL has already been processed in the past."""
        return video_url in self.memory_data.get("duplicate_hashes", [])

    def add_duplicate_hash(self, video_url: str):
        """Adds a video URL to the duplicate registry."""
        if "duplicate_hashes" not in self.memory_data:
            self.memory_data["duplicate_hashes"] = []
        if video_url not in self.memory_data["duplicate_hashes"]:
            self.memory_data["duplicate_hashes"].append(video_url)
            self._save_memory()

    def get_best_style(self, topic: str) -> Dict[str, Any]:
        """
        Determines the best editing style for the given topic based on priority and history.
        """
        styles = self.memory_data.get("style_library", {})
        
        # Simple heuristic: prioritize High priority, then sort by success count
        ranked_styles = sorted(
            styles.items(),
            key=lambda x: (1 if x[1].get("priority") == "High" else 0, x[1].get("success_count", 0)),
            reverse=True
        )
        
        # Default style is funny
        chosen_key = "funny"
        if ranked_styles:
            chosen_key = ranked_styles[0][0]
            
        logger.info(f"Memory Agent: Recommending editing style '{chosen_key}' for topic '{topic}'")
        style_data = styles.get(chosen_key, {})
        return {
            "style_key": chosen_key,
            "style_name": style_data.get("name", "Funny Zoom"),
            "transition": self.memory_data.get("settings", {}).get("default_transition", "tv_static"),
            "caption_style": self.memory_data.get("settings", {}).get("default_caption_style", "Pop"),
            "average_pacing": style_data.get("average_pacing", None),
            "pre_cut_whoosh_delay": style_data.get("pre_cut_whoosh_delay", None)
        }

    def register_successful_run(self, topic: str, style_key: str, video_path: str, url: str = ""):
        """Logs details of a successfully finished and uploaded video run."""
        import datetime
        
        # 1. Update style library stats
        styles = self.memory_data.get("style_library", {})
        if style_key in styles:
            styles[style_key]["success_count"] = styles[style_key].get("success_count", 0) + 1
            
        # 2. Append history record
        record = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "topic": topic,
            "style": style_key,
            "output_file": os.path.basename(video_path),
            "size_mb": round(os.path.getsize(video_path) / (1024 * 1024), 2) if os.path.exists(video_path) else 0.0,
            "url": url
        }
        self.memory_data.setdefault("video_history", []).append(record)
        
        self._save_memory()
        logger.info(f"Memory Agent: Successfully logged run metrics for topic '{topic}'.")
