#!/usr/bin/env python3
"""
Load React Technology Triples into Neo4j
Simple script to load only React technology data
"""

import os
import sys
from pathlib import Path

# Add the scripts directory to the path
sys.path.append(str(Path(__file__).parent))

from TechnologyNeo4jLoader import TechnologyNeo4jLoader


def load_react_only():
    """Load only React technology triples"""
    print("🎯 Loading React technology triples into Neo4j...")
    
    # Create loader
    loader = TechnologyNeo4jLoader()
    
    # Load only React
    technologies = ["react"]
    success = loader.load_all_technologies(
        clear_existing=False,  # Don't clear existing data
        technologies=technologies
    )
    
    if success:
        print(f"✅ Successfully loaded React technology")
        print("🔍 You can now explore React-specific data in Neo4j Browser")
        print("🌐 Open: http://localhost:7474")
    else:
        print("❌ Failed to load React technology")


if __name__ == "__main__":
    load_react_only() 