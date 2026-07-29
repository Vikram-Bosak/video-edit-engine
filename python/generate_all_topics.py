"""
Batch Topic Video Generator.

Automates the generation of reaction compilation videos for all target topics
(Gym Fails, Swimming Pool Fails, Football Fails, Animal Fails, Basketball Fails)
completely hands-free.
"""

from __future__ import annotations

import logging
import os
import sys
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("batch_generator")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    
    topics = {
        "gym_fails": ("funny gym fails", "output/gym_reaction_final.mp4"),
        "swimming_pool": ("funny swimming pool fails", "output/swimming_pool_reaction_final.mp4"),
        "football": ("funny football soccer fails", "output/football_reaction_final.mp4"),
        "animal_fails": ("funny animal dog cat fails", "output/animal_reaction_final.mp4"),
        "basketball": ("funny basketball moments fails", "output/basketball_reaction_final.mp4")
    }
    
    logger.info("=== STARTING COMPLETE BATCH RUN FOR ALL TOPICS ===")
    
    for key, (query, output_path) in topics.items():
        logger.info(f"Generating video for topic: {key} (Query: '{query}')")
        cmd = [
            "python", "python/reaction_workflow.py",
            "--query", query,
            "--output", output_path
        ]
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Successfully generated: {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to generate video for {key}: {e}")
            
    logger.info("=== BATCH RUN FOR ALL TOPICS COMPLETED ===")


if __name__ == "__main__":
    main()
