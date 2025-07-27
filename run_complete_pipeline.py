"""
Complete Knowledge Graph Pipeline Runner
Executes all 6 steps in sequence with proper error handling
"""

import sys
import os
from pathlib import Path
from datetime import datetime

from scripts.Neo4jLocalLoader import Neo4jLocalLoader
from scripts.extract_triples import ImprovedTripleExtractor

# Add scripts directory to Python path
sys.path.append('scripts')

def run_complete_pipeline(clear_neo4j=True, use_improved_extraction=True):
    """Run the complete 6-step knowledge graph pipeline"""

    print("🚀" * 25)
    print("🎯 KNOWLEDGE GRAPH PIPELINE - COMPLETE EXECUTION")
    print("🚀" * 25)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    pipeline_results = {}

    try:
        # ============================================================
        # STEP 1: Create Initial Ontology
        # ============================================================
        print("📍 STEP 1/6: Creating Initial Ontology")
        print("=" * 50)

        from create_ontology import OntologyCreator

        creator = OntologyCreator(use_fallback=True)  # Use enhanced fallback
        ontology = creator.create_ontology()

        if not ontology:
            print("❌ Step 1 FAILED! Cannot proceed without ontology.")
            return False

        pipeline_results['step1'] = {
            'status': 'SUCCESS',
            'classes': len(ontology['classes']),
            'relations': len(ontology['relations']),
            'output_file': 'data/ontology/initial_ontology.json'
        }

        print(f"✅ Step 1 COMPLETED: {len(ontology['classes'])} classes, {len(ontology['relations'])} relations")
        print()

        # ============================================================
        # STEP 2: Web Scraping
        # ============================================================
        print("📍 STEP 2/6: Web Scraping Java Tutorials")
        print("=" * 50)

        from web_scraper import WebScraper

        scraper = WebScraper()
        scraped_data = scraper.scrape_and_save()

        if not scraped_data:
            print("❌ Step 2 FAILED! Cannot proceed without scraped data.")
            return False

        successful_scrapes = sum(1 for data in scraped_data.values() if 'error' not in data)
        total_pages = len(scraped_data)

        pipeline_results['step2'] = {
            'status': 'SUCCESS',
            'total_pages': total_pages,
            'successful_pages': successful_scrapes,
            'success_rate': f"{(successful_scrapes/total_pages)*100:.1f}%",
            'output_file': 'data/scraped_content/w3schools_java_tutorials.json'
        }

        print(f"✅ Step 2 COMPLETED: {successful_scrapes}/{total_pages} pages scraped successfully")
        print()

        # ============================================================
        # STEP 3: Extract Triples (Improved Version)
        # ============================================================
        print("📍 STEP 3/6: Extracting Triples with Advanced Methods")
        print("=" * 50)

        if use_improved_extraction:
            extractor = ImprovedTripleExtractor(
                use_rebel=True,  # Try REBEL first
                use_improved_patterns=True
            )
            triples = extractor.extract_and_save()
        else:
            # Fallback to basic extraction if needed
            print("⚠️ Using basic extraction method...")
            from extract_triples import TripleExtractor
            extractor = TripleExtractor(use_fallback=True)
            triples = extractor.extract_and_save()

        if not triples:
            print("❌ Step 3 FAILED! Cannot proceed without extracted triples.")
            return False

        pipeline_results['step3'] = {
            'status': 'SUCCESS',
            'total_triples': len(triples),
            'high_confidence': len([t for t in triples if t.get('confidence', 0) > 0.8]),
            'extraction_method': 'improved' if use_improved_extraction else 'basic',
            'output_file': 'data/extracted_triples/extracted_triples_improved.json'
        }

        print(f"✅ Step 3 COMPLETED: {len(triples)} triples extracted")
        print()

        # ============================================================
        # STEP 4: Dynamic Ontology Update
        # ============================================================
        print("📍 STEP 4/6: Dynamic Ontology Update")
        print("=" * 50)

        from update_ontology import DynamicOntologyUpdater

        # Use the improved triples file if it exists
        triples_file = "data/extracted_triples/extracted_triples_improved.json"
        if not Path(triples_file).exists():
            triples_file = "data/extracted_triples/extracted_triples.json"

        updater = DynamicOntologyUpdater(
            triples_file=triples_file,
            use_llm=False  # Use rule-based approach
        )
        updated_ontology = updater.update_ontology_dynamically()

        if not updated_ontology:
            print("⚠️ Step 4 had issues, continuing with original ontology...")
            updated_ontology = ontology

        pipeline_results['step4'] = {
            'status': 'SUCCESS',
            'final_classes': len(updated_ontology['classes']),
            'final_relations': len(updated_ontology['relations']),
            'added_concepts': len(updated_ontology['classes']) - len(ontology['classes']),
            'output_file': 'data/ontology/updated_ontology.json'
        }

        print(f"✅ Step 4 COMPLETED: {len(updated_ontology['classes'])} classes, {len(updated_ontology['relations'])} relations")
        print()

        # ============================================================
        # STEP 5: Data Validation
        # ============================================================
        print("📍 STEP 5/6: Validating Data Quality")
        print("=" * 50)

        from validate_data import DataValidator

        # Use updated ontology and improved triples
        validator = DataValidator(
            ontology_file="data/ontology/updated_ontology.json",
            triples_file=triples_file
        )
        validated_data = validator.validate_all_data()

        if not validated_data:
            print("❌ Step 5 FAILED! Cannot proceed without validated data.")
            return False

        pipeline_results['step5'] = {
            'status': 'SUCCESS',
            'total_nodes': validated_data['metadata']['total_nodes'],
            'total_edges': validated_data['metadata']['total_edges'],
            'validation_stats': validated_data['metadata']['validation_stats'],
            'output_files': ['data/validated_data/validated_data.json',
                           'data/validated_data/nodes.json',
                           'data/validated_data/edges.json']
        }

        print(f"✅ Step 5 COMPLETED: {validated_data['metadata']['total_nodes']} nodes, {validated_data['metadata']['total_edges']} edges validated")
        print()

        # ============================================================
        # STEP 6: Load into Neo4j
        # ============================================================
        print("📍 STEP 6/6: Loading into Local Neo4j")
        print("=" * 50)

        # Get Neo4j credentials from environment or use defaults
        uri = os.getenv("NEO4J_LOCAL_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_LOCAL_USER", "neo4j")
        password = os.getenv("NEO4J_LOCAL_PASSWORD", "password123")
        database = os.getenv("NEO4J_LOCAL_DATABASE", "neo4j")

        loader = Neo4jLocalLoader(
            uri=uri,
            user=user,
            password=password,
            database=database
        )
        success = loader.load_data_to_neo4j(clear_existing=clear_neo4j)

        if not success:
            print("❌ Step 6 FAILED! Data not loaded into Neo4j.")
            pipeline_results['step6'] = {'status': 'FAILED'}
            # Don't return False here - pipeline data is still valid
        else:
            pipeline_results['step6'] = {
                'status': 'SUCCESS',
                'neo4j_uri': uri,
                'browser_url': 'http://localhost:7474'
            }
            print(f"✅ Step 6 COMPLETED: Data loaded into Neo4j at {uri}")

        print()

        # ============================================================
        # PIPELINE COMPLETION SUMMARY
        # ============================================================
        print("🎉" * 25)
        print("🏆 KNOWLEDGE GRAPH PIPELINE COMPLETED!")
        print("🎉" * 25)

        print(f"\n📊 EXECUTION SUMMARY")
        print("=" * 40)
        print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        for step, result in pipeline_results.items():
            status_emoji = "✅" if result['status'] == 'SUCCESS' else "❌"
            print(f"{status_emoji} {step.upper()}: {result['status']}")

        print(f"\n📈 FINAL KNOWLEDGE GRAPH METRICS")
        print("=" * 40)
        print(f"📋 Ontology Classes: {pipeline_results['step4']['final_classes']}")
        print(f"🔗 Ontology Relations: {pipeline_results['step4']['final_relations']}")
        print(f"📊 Scraped Pages: {pipeline_results['step2']['successful_pages']}")
        print(f"🔍 Extracted Triples: {pipeline_results['step3']['total_triples']}")
        print(f"🎯 Final Nodes: {pipeline_results['step5']['total_nodes']}")
        print(f"🌐 Final Relationships: {pipeline_results['step5']['total_edges']}")

        if pipeline_results['step6']['status'] == 'SUCCESS':
            print(f"\n🌐 ACCESS YOUR KNOWLEDGE GRAPH:")
            print("=" * 35)
            print(f"🔍 Neo4j Browser: http://localhost:7474")
            print(f"🔑 Login: {user} / {password}")
            print(f"📊 Database: {database}")

            print(f"\n🔍 SAMPLE CYPHER QUERIES TO TRY:")
            sample_queries = [
                "MATCH (n) RETURN n LIMIT 25",
                "MATCH ()-[r]-() RETURN DISTINCT type(r), count(r) ORDER BY count(r) DESC",
                "MATCH (n:Entity) OPTIONAL MATCH (n)-[r]-() RETURN n.name, count(r) ORDER BY count(r) DESC LIMIT 10"
            ]

            for i, query in enumerate(sample_queries, 1):
                print(f"   {i}. {query}")

        return True

    except Exception as e:
        print(f"❌ CRITICAL PIPELINE ERROR: {e}")
        print(f"\n🔍 ERROR DETAILS:")
        import traceback
        traceback.print_exc()
        return False


def run_individual_step(step_number):
    """Run a specific step of the pipeline"""

    print(f"🎯 Running Step {step_number} individually...")
    print("=" * 50)

    try:
        if step_number == 1:
            from create_ontology import OntologyCreator
            creator = OntologyCreator(use_fallback=True)
            result = creator.create_ontology()

        elif step_number == 2:
            from web_scraper import WebScraper
            scraper = WebScraper()
            result = scraper.scrape_and_save()

        elif step_number == 3:
            from extract_triples_improved import ImprovedTripleExtractor
            extractor = ImprovedTripleExtractor(use_rebel=True, use_improved_patterns=True)
            result = extractor.extract_and_save()

        elif step_number == 4:
            from update_ontology import DynamicOntologyUpdater
            updater = DynamicOntologyUpdater(use_llm=False)
            result = updater.update_ontology_dynamically()

        elif step_number == 5:
            from validate_data import DataValidator
            validator = DataValidator()
            result = validator.validate_all_data()

        elif step_number == 6:
            from neo4j_loader_local import Neo4jLocalLoader
            uri = os.getenv("NEO4J_LOCAL_URI", "bolt://localhost:7687")
            user = os.getenv("NEO4J_LOCAL_USER", "neo4j")
            password = os.getenv("NEO4J_LOCAL_PASSWORD", "password123")

            loader = Neo4jLocalLoader(uri=uri, user=user, password=password)
            result = loader.load_data_to_neo4j(clear_existing=True)

        else:
            print(f"❌ Invalid step number: {step_number}. Use 1-6.")
            return False

        if result:
            print(f"✅ Step {step_number} completed successfully!")
            return True
        else:
            print(f"❌ Step {step_number} failed!")
            return False

    except Exception as e:
        print(f"❌ Step {step_number} failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_quality_analysis():
    """Run knowledge graph quality analysis"""
    print("🔍 Running Knowledge Graph Quality Analysis...")
    print("=" * 50)

    try:
        from analyze_kg_quality import KGQualityAnalyzer
        analyzer = KGQualityAnalyzer()
        analyzer.analyze_current_kg()
        return True
    except Exception as e:
        print(f"❌ Quality analysis failed: {e}")
        return False


def main():
    """Main entry point with command line options"""

    print("🎯 Knowledge Graph Pipeline Runner")
    print("=" * 40)

    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()

        # Handle specific step execution
        if arg.isdigit():
            step = int(arg)
            if 1 <= step <= 6:
                success = run_individual_step(step)
                sys.exit(0 if success else 1)
            else:
                print("❌ Step number must be between 1 and 6")
                sys.exit(1)

        # Handle special commands
        elif arg in ['--help', '-h', 'help']:
            print_help()
            sys.exit(0)

        elif arg in ['--analysis', '-a', 'analysis']:
            success = run_quality_analysis()
            sys.exit(0 if success else 1)

        elif arg in ['--test', 'test']:
            print("🧪 Running pipeline test...")
            success = test_pipeline_prerequisites()
            sys.exit(0 if success else 1)

        else:
            print(f"❌ Unknown argument: {arg}")
            print("Use --help for usage information")
            sys.exit(1)

    # Run complete pipeline by default
    print("🚀 Starting complete pipeline execution...")
    print()

    success = run_complete_pipeline(
        clear_neo4j=True,
        use_improved_extraction=True
    )

    if success:
        print(f"\n🎉 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("Your knowledge graph is ready for exploration!")
    else:
        print(f"\n❌ PIPELINE EXECUTION FAILED!")
        print("Check the error messages above for troubleshooting.")

    sys.exit(0 if success else 1)


def print_help():
    """Print help information"""
    print("""
🎯 Knowledge Graph Pipeline Runner - Help

USAGE:
  python run_complete_pipeline.py [OPTION]

OPTIONS:
  (no args)     Run complete 6-step pipeline
  1-6          Run specific step only
  --analysis   Run quality analysis tool
  --test       Test pipeline prerequisites
  --help       Show this help message

PIPELINE STEPS:
  1. Create Initial Ontology
  2. Web Scraping (W3Schools Java)
  3. Extract Triples (REBEL + Patterns)
  4. Update Ontology Dynamically
  5. Validate Data Quality
  6. Load into Neo4j

ENVIRONMENT VARIABLES:
  NEO4J_LOCAL_URI      Default: bolt://localhost:7687
  NEO4J_LOCAL_USER     Default: neo4j
  NEO4J_LOCAL_PASSWORD Default: password123
  NEO4J_LOCAL_DATABASE Default: neo4j

EXAMPLES:
  python run_complete_pipeline.py           # Full pipeline
  python run_complete_pipeline.py 3         # Only step 3
  python run_complete_pipeline.py --analysis # Quality analysis
  python run_complete_pipeline.py --test    # Test setup

PREREQUISITES:
  • Python packages: see requirements.txt
  • Neo4j Desktop running (for step 6)
  • Internet connection (for step 2)
    """)


def test_pipeline_prerequisites():
    """Test if all prerequisites are met"""
    print("🧪 Testing Pipeline Prerequisites...")
    print("=" * 40)

    tests_passed = 0
    total_tests = 0

    # Test 1: Check required directories
    total_tests += 1
    required_dirs = ['data', 'scripts']
    missing_dirs = [d for d in required_dirs if not Path(d).exists()]

    if not missing_dirs:
        print("✅ Required directories exist")
        tests_passed += 1
    else:
        print(f"❌ Missing directories: {missing_dirs}")

    # Test 2: Check Python packages
    total_tests += 1
    required_packages = ['requests', 'beautifulsoup4', 'transformers', 'neo4j']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if not missing_packages:
        print("✅ Required Python packages available")
        tests_passed += 1
    else:
        print(f"❌ Missing packages: {missing_packages}")
        print("   Install with: pip install " + " ".join(missing_packages))

    # Test 3: Check Neo4j connection
    total_tests += 1
    try:
        from neo4j import GraphDatabase
        uri = os.getenv("NEO4J_LOCAL_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_LOCAL_USER", "neo4j")
        password = os.getenv("NEO4J_LOCAL_PASSWORD", "password123")

        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            session.run("RETURN 1")
        driver.close()

        print("✅ Neo4j connection successful")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        print("   Make sure Neo4j Desktop is running")

    # Test 4: Check internet connection
    total_tests += 1
    try:
        import requests
        response = requests.get("https://www.w3schools.com", timeout=10)
        if response.status_code == 200:
            print("✅ Internet connection available")
            tests_passed += 1
        else:
            print("❌ Internet connection issues")
    except Exception:
        print("❌ Internet connection failed")
        print("   Check your network connection")

    print(f"\n📊 TEST RESULTS: {tests_passed}/{total_tests} passed")

    if tests_passed == total_tests:
        print("✅ All prerequisites met! Ready to run pipeline.")
        return True
    else:
        print("❌ Some prerequisites missing. Fix issues above before running.")
        return False


if __name__ == "__main__":
    main()