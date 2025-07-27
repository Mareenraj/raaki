"""
Configuration settings for Knowledge Graph project
"""

import os
from pathlib import Path


class Settings:
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    LOGS_DIR = PROJECT_ROOT / "logs"

    # Data subdirectories
    ONTOLOGY_DIR = DATA_DIR / "ontology"
    SCRAPED_CONTENT_DIR = DATA_DIR / "scraped_content"
    SAMPLE_TEXT_DIR = DATA_DIR / "sample_text"

    # Model configurations
    GEMMA_MODEL = "google/gemma-2-9b-it"
    REBEL_MODEL = "Babelscape/rebel-large"

    # Scraping configurations
    SCRAPING_DELAY = 1.0  # seconds between requests
    MAX_RETRIES = 3
    TIMEOUT = 30

    # Neo4j configurations (from environment variables)
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://6e7e8fbf.databases.neo4j.io")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "8s3xntRX-IYB2dG4qlOK2J5gDGhHMa4AFFXmrr8ks6U")

    # File paths
    INITIAL_ONTOLOGY_FILE = ONTOLOGY_DIR / "initial_ontology.json"
    UPDATED_ONTOLOGY_FILE = ONTOLOGY_DIR / "updated_ontology.json"
    SCRAPED_DATA_FILE = SCRAPED_CONTENT_DIR / "w3schools_java_tutorials.json"

    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.DATA_DIR,
            cls.ONTOLOGY_DIR,
            cls.SCRAPED_CONTENT_DIR,
            cls.SAMPLE_TEXT_DIR,
            cls.LOGS_DIR
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        print("Project directories created successfully!")


if __name__ == "__main__":
    Settings.ensure_directories()