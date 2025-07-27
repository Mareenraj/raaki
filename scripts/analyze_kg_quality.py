"""
Knowledge Graph Quality Analyzer and Improvement Tool
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
import re


class KGQualityAnalyzer:
    def __init__(self):
        self.data_dir = Path("data")
        self.scraped_file = self.data_dir / "scraped_content" / "w3schools_java_tutorials.json"
        self.triples_file = self.data_dir / "extracted_triples" / "extracted_triples.json"
        self.validated_file = self.data_dir / "validated_data" / "validated_data.json"

    def analyze_current_kg(self):
        """Analyze the current knowledge graph quality"""
        print("🔍 ANALYZING CURRENT KNOWLEDGE GRAPH QUALITY")
        print("=" * 55)

        # Load data
        scraped_data = self.load_scraped_data()
        triples_data = self.load_triples_data()
        validated_data = self.load_validated_data()

        if not scraped_data or not triples_data:
            print("❌ Missing data files. Run the pipeline first.")
            return

        # Analyze each stage
        self.analyze_scraping_quality(scraped_data)
        self.analyze_extraction_quality(triples_data)
        self.analyze_validation_quality(validated_data)
        self.suggest_improvements()

    def load_scraped_data(self):
        """Load scraped content"""
        try:
            with open(self.scraped_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading scraped data: {e}")
            return None

    def load_triples_data(self):
        """Load extracted triples"""
        try:
            with open(self.triples_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading triples data: {e}")
            return None

    def load_validated_data(self):
        """Load validated data"""
        try:
            with open(self.validated_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Validated data not found: {e}")
            return None

    def analyze_scraping_quality(self, scraped_data):
        """Analyze quality of scraped content"""
        print("\n📄 SCRAPING QUALITY ANALYSIS")
        print("-" * 35)

        total_pages = len(scraped_data)
        successful_pages = sum(1 for data in scraped_data.values() if 'error' not in data)

        # Content analysis
        total_text_sections = 0
        total_code_examples = 0
        java_concepts_found = set()

        java_patterns = [
            r'\bclass\b', r'\binterface\b', r'\bpublic\b', r'\bprivate\b',
            r'\bstatic\b', r'\bfinal\b', r'\bextends\b', r'\bimplements\b',
            r'\bString\b', r'\bint\b', r'\bdouble\b', r'\bboolean\b',
            r'\bArrayList\b', r'\bHashMap\b', r'\bException\b'
        ]

        for url, content in scraped_data.items():
            if 'error' not in content:
                text_content = content.get('text_content', [])
                code_examples = content.get('code_examples', [])

                total_text_sections += len(text_content)
                total_code_examples += len(code_examples)

                # Find Java concepts
                all_text = ' '.join(text_content + code_examples)
                for pattern in java_patterns:
                    if re.search(pattern, all_text, re.IGNORECASE):
                        java_concepts_found.add(pattern.replace('\\b', ''))

        print(f"📊 Pages scraped: {successful_pages}/{total_pages} ({successful_pages / total_pages * 100:.1f}%)")
        print(f"📝 Text sections: {total_text_sections:,}")
        print(f"💻 Code examples: {total_code_examples:,}")
        print(f"☕ Java concepts found: {len(java_concepts_found)}")
        print(f"   Concepts: {', '.join(sorted(java_concepts_found))}")

        # Quality score
        scraping_score = (successful_pages / total_pages * 40 +
                          min(total_text_sections / 1000, 1) * 30 +
                          min(total_code_examples / 500, 1) * 30)
        print(f"📈 Scraping Quality Score: {scraping_score:.1f}/100")

    def analyze_extraction_quality(self, triples_data):
        """Analyze quality of triple extraction"""
        print("\n🔍 EXTRACTION QUALITY ANALYSIS")
        print("-" * 38)

        if 'triples' in triples_data:
            all_triples = triples_data['triples'].get('all', [])
            high_conf = triples_data['triples'].get('high_confidence', [])
            medium_conf = triples_data['triples'].get('medium_confidence', [])
        else:
            all_triples = triples_data if isinstance(triples_data, list) else []
            high_conf = [t for t in all_triples if t.get('confidence', 0) > 0.7]
            medium_conf = [t for t in all_triples if 0.5 <= t.get('confidence', 0) <= 0.7]

        print(f"🔢 Total triples extracted: {len(all_triples):,}")
        print(f"🎯 High confidence (>0.7): {len(high_conf):,}")
        print(f"📊 Medium confidence (0.5-0.7): {len(medium_conf):,}")

        if all_triples:
            # Analyze relations
            relations = [t.get('relation', '') for t in all_triples]
            relation_counts = Counter(relations)

            print(f"\n🔗 RELATIONSHIP ANALYSIS:")
            for rel, count in relation_counts.most_common(10):
                print(f"   • {rel}: {count:,} occurrences")

            # Analyze entities
            subjects = [t.get('subject', '') for t in all_triples]
            objects = [t.get('object', '') for t in all_triples]
            all_entities = subjects + objects
            entity_counts = Counter(all_entities)

            print(f"\n🏷️  TOP ENTITIES:")
            for entity, count in entity_counts.most_common(10):
                print(f"   • {entity}: {count:,} mentions")

            # Quality assessment
            valid_relations = sum(1 for rel in relations if rel and len(rel) > 2)
            valid_entities = sum(1 for ent in all_entities if ent and len(ent) > 1)

            extraction_score = (valid_relations / len(relations) * 50 +
                                valid_entities / len(all_entities) * 30 +
                                len(high_conf) / len(all_triples) * 20) if all_triples else 0

            print(f"\n📈 Extraction Quality Score: {extraction_score:.1f}/100")

    def analyze_validation_quality(self, validated_data):
        """Analyze validation and final KG quality"""
        print("\n✅ VALIDATION & FINAL KG ANALYSIS")
        print("-" * 40)

        if not validated_data:
            print("⚠️ No validated data available")
            return

        nodes = validated_data.get('nodes', [])
        edges = validated_data.get('edges', [])
        metadata = validated_data.get('metadata', {})

        print(f"🎯 Final nodes: {len(nodes):,}")
        print(f"🔗 Final relationships: {len(edges):,}")

        if metadata.get('validation_stats'):
            stats = metadata['validation_stats']
            print(
                f"📊 Validation success rate: {stats.get('valid_triples', 0) / (stats.get('valid_triples', 0) + stats.get('invalid_triples', 1)) * 100:.1f}%")

        # Analyze node types
        if nodes:
            node_types = Counter(node.get('type', 'Unknown') for node in nodes)
            print(f"\n🏷️  NODE TYPE DISTRIBUTION:")
            for ntype, count in node_types.most_common(5):
                print(f"   • {ntype}: {count:,} nodes")

        # Analyze relationship types
        if edges:
            rel_types = Counter(edge.get('relation', 'Unknown') for edge in edges)
            print(f"\n🔗 RELATIONSHIP TYPE DISTRIBUTION:")
            for rtype, count in rel_types.most_common(5):
                print(f"   • {rtype}: {count:,} relationships")

    def suggest_improvements(self):
        """Suggest specific improvements"""
        print("\n🚀 IMPROVEMENT RECOMMENDATIONS")
        print("=" * 35)

        improvements = [
            "1. 🤖 ENABLE REBEL MODEL",
            "   • Set use_fallback=False in extract_triples.py",
            "   • REBEL will extract much better relationships",
            "   • Install: pip install torch transformers",
            "",
            "2. 🎯 IMPROVE PATTERN MATCHING",
            "   • Add more Java-specific patterns",
            "   • Include framework-specific patterns (Spring, Hibernate)",
            "   • Add method signature patterns",
            "",
            "3. 📝 ENHANCE TEXT PREPROCESSING",
            "   • Clean HTML entities and special characters",
            "   • Split long sentences for better extraction",
            "   • Filter out navigation and footer text",
            "",
            "4. 🔍 POST-PROCESSING FILTERS",
            "   • Remove generic entities like 'example', 'code'",
            "   • Merge similar entities (String vs string)",
            "   • Filter low-confidence relationships",
            "",
            "5. 🧠 ONTOLOGY IMPROVEMENTS",
            "   • Add more specific Java classes (Collection, Stream, etc.)",
            "   • Include framework-specific relations",
            "   • Add inheritance hierarchy relations"
        ]

        for improvement in improvements:
            print(improvement)

    def generate_improved_patterns(self):
        """Generate improved extraction patterns"""
        return {
            "java_class_patterns": [
                (r'\bclass\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([\w,\s]+))?', 'class_definition'),
                (r'\binterface\s+(\w+)(?:\s+extends\s+([\w,\s]+))?', 'interface_definition'),
                (r'(\w+)\s+extends\s+(\w+)', 'inheritance'),
                (r'(\w+)\s+implements\s+(\w+)', 'implementation')
            ],
            "java_method_patterns": [
                (r'\b(?:public|private|protected)?\s*(?:static)?\s*(\w+)\s+(\w+)\s*\([^)]*\)', 'method_declaration'),
                (r'(\w+)\.(\w+)\s*\(', 'method_call'),
                (r'new\s+(\w+)\s*\(', 'instantiation')
            ],
            "java_concept_patterns": [
                (r'\b(ArrayList|HashMap|HashSet|LinkedList|TreeMap|Vector)\b', 'collection_type'),
                (r'\b(String|int|double|float|boolean|char|long|byte|short)\b', 'primitive_type'),
                (r'\b(try|catch|finally|throw|throws)\b.*?\b(\w+Exception|\w+Error)\b', 'exception_handling'),
                (r'@(\w+)', 'annotation'),
                (r'\bimport\s+([\w.]+)', 'import_statement')
            ],
            "framework_patterns": [
                (r'\b(Spring|SpringBoot|Hibernate|JPA|Maven|Gradle)\b', 'framework'),
                (r'@(Controller|Service|Repository|Component|Entity|Table)', 'spring_annotation'),
                (r'\b(REST|HTTP|JSON|XML|API)\b', 'web_concept')
            ]
        }


def main():
    """Run KG quality analysis"""
    analyzer = KGQualityAnalyzer()
    analyzer.analyze_current_kg()

    print(f"\n📋 NEXT STEPS:")
    print("1. Review the analysis above")
    print("2. Implement suggested improvements")
    print("3. Re-run the pipeline with better extraction")
    print("4. Use REBEL model for better relationship extraction")


if __name__ == "__main__":
    main()