#!/usr/bin/env python3
"""
Technology Neo4j Loader Runner
Simple script to demonstrate loading technology-specific triples into Neo4j
"""

import os
import sys
from pathlib import Path

# Add the scripts directory to the path
sys.path.append(str(Path(__file__).parent))

from TechnologyNeo4jLoader import TechnologyNeo4jLoader


def load_specific_technologies():
    """Load only specific technologies"""
    print("🎯 Loading specific technologies...")
    
    # Create loader
    loader = TechnologyNeo4jLoader()
    
    # Load only Python and Java
    technologies = ["python", "java"]
    success = loader.load_all_technologies(
        clear_existing=True,
        technologies=technologies
    )
    
    if success:
        print(f"✅ Successfully loaded {technologies}")
    else:
        print("❌ Failed to load specific technologies")


def load_all_technologies():
    """Load all available technology files"""
    print("🎯 Loading all available technologies...")
    
    # Create loader
    loader = TechnologyNeo4jLoader()
    
    # Load all technologies
    success = loader.load_all_technologies(clear_existing=True)
    
    if success:
        print("✅ Successfully loaded all technologies")
    else:
        print("❌ Failed to load all technologies")


def load_with_custom_config():
    """Load with custom Neo4j configuration"""
    print("🎯 Loading with custom configuration...")
    
    # Custom configuration
    config = {
        "uri": "bolt://localhost:7687",
        "user": "neo4j",
        "password": "Ragav@2000",  # Change this to your password
        "database": "trainee-graph"
    }
    
    # Create loader with custom config
    loader = TechnologyNeo4jLoader(
        uri=config["uri"],
        user=config["user"],
        password=config["password"],
        database=config["database"]
    )
    
    # Load all technologies
    success = loader.load_all_technologies(clear_existing=True)
    
    if success:
        print("✅ Successfully loaded with custom configuration")
    else:
        print("❌ Failed to load with custom configuration")


def main():
    """Main function with menu options"""
    print("🚀 Technology Neo4j Loader Runner")
    print("=" * 40)
    print("Choose an option:")
    print("1. Load all available technologies")
    print("2. Load specific technologies (Python, Java)")
    print("3. Load with custom configuration")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                load_all_technologies()
                break
            elif choice == "2":
                load_specific_technologies()
                break
            elif choice == "3":
                load_with_custom_config()
                break
            elif choice == "4":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-4.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main() 