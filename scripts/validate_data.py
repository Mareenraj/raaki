"""
Step 5: Validate Nodes and Relations against ontology
"""

import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict


class DataValidator:
    def __init__(self,
                 ontology_file="data/ontology/updated_ontology.json",
                 triples_file="data/extracted_triples/extracted_triples.json",
                 output_dir="data/validated_data"):
        self.ontology_file = Path(ontology_file)
        self.triples_file = Path(triples_file)
        self.output_dir = Path(output_dir)
        self.ontology = None
        self.validation_stats = defaultdict(int)

    def load_ontology(self):
        """Load the ontology for validation"""
        try:
            # Try updated ontology first, fall back to initial
            if not self.ontology_file.exists():
                initial_ontology = Path("data/ontology/initial_ontology.json")
                if initial_ontology.exists():
                    self.ontology_file = initial_ontology
                else:
                    raise FileNotFoundError("No ontology file found")

            with open(self.ontology_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle both formats
            if 'ontology' in data:
                self.ontology = data['ontology']
            else:
                self.ontology = data

            print(f"📁 Loaded ontology from: {self.ontology_file}")
            print(
                f"📊 Classes: {len(self.ontology.get('classes', []))}, Relations: {len(self.ontology.get('relations', []))}")
            return True

        except Exception as e:
            print(f"❌ Error loading ontology: {e}")
            return False

    def load_triples(self):
        """Load extracted triples for validation"""
        try:
            if not self.triples_file.exists():
                raise FileNotFoundError(f"Triples file not found: {self.triples_file}")

            with open(self.triples_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Get all triples
            if 'triples' in data:
                all_triples = data['triples'].get('all', [])
            else:
                all_triples = data if isinstance(data, list) else []

            print(f"📊 Loaded {len(all_triples)} triples for validation")
            return all_triples

        except Exception as e:
            print(f"❌ Error loading triples: {e}")
            return []

    def normalize_entity(self, entity):
        """Normalize entity names for consistent validation"""
        if not entity:
            return ""

        entity = entity.strip()

        # Handle common variations
        mappings = {
            'string': 'String',
            'integer': 'Integer',
            'int': 'Integer',
            'double': 'Double',
            'float': 'Float',
            'boolean': 'Boolean',
            'char': 'Character',
            'arraylist': 'ArrayList',
            'hashmap': 'HashMap',
            'list': 'List',
            'set': 'Set',
            'map': 'Map'
        }

        lower_entity = entity.lower()
        if lower_entity in mappings:
            return mappings[lower_entity]

        return entity.capitalize()

    def validate_triple(self, triple):
        """Validate a single triple against the ontology"""
        validation_result = {
            'original_triple': triple,
            'is_valid': True,
            'issues': [],
            'normalized_triple': {}
        }

        try:
            subject = self.normalize_entity(triple.get('subject', ''))
            relation = triple.get('relation', '').strip()
            obj = self.normalize_entity(triple.get('object', ''))
            confidence = triple.get('confidence', 0)

            # Store normalized version
            validation_result['normalized_triple'] = {
                'subject': subject,
                'relation': relation,
                'object': obj,
                'confidence': confidence,
                'source_url': triple.get('source_url', ''),
                'source_type': triple.get('source_type', 'text')
            }

            # Check if relation exists in ontology
            valid_relations = self.ontology.get('relations', [])
            if relation not in valid_relations:
                validation_result['issues'].append(f"Unknown relation: {relation}")
                self.validation_stats['unknown_relations'] += 1

            # Check if entities could be valid classes
            valid_classes = self.ontology.get('classes', [])

            # For subjects and objects, we're more lenient as they could be instances
            # But we track if they're known class types
            if subject not in valid_classes:
                # Check if it's a reasonable programming concept
                if not self.is_reasonable_entity(subject):
                    validation_result['issues'].append(f"Questionable subject: {subject}")
                    self.validation_stats['questionable_subjects'] += 1
                else:
                    self.validation_stats['unknown_but_reasonable_subjects'] += 1

            if obj not in valid_classes:
                if not self.is_reasonable_entity(obj):
                    validation_result['issues'].append(f"Questionable object: {obj}")
                    self.validation_stats['questionable_objects'] += 1
                else:
                    self.validation_stats['unknown_but_reasonable_objects'] += 1

            # Check confidence threshold
            if confidence < 0.3:
                validation_result['issues'].append(f"Low confidence: {confidence}")
                self.validation_stats['low_confidence'] += 1

            # Mark as invalid if there are critical issues
            critical_issues = ['Unknown relation', 'Questionable subject', 'Questionable object']
            if any(issue.split(':')[0] in critical_issues for issue in validation_result['issues']):
                validation_result['is_valid'] = False
                self.validation_stats['invalid_triples'] += 1
            else:
                self.validation_stats['valid_triples'] += 1

        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['issues'].append(f"Validation error: {str(e)}")
            self.validation_stats['validation_errors'] += 1

        return validation_result

    def is_reasonable_entity(self, entity):
        """Check if an entity name seems reasonable for programming concepts"""
        if not entity or len(entity) < 2:
            return False

        # Common programming terms that might not be in ontology
        reasonable_patterns = [
            'method', 'function', 'variable', 'parameter', 'argument',
            'class', 'interface', 'enum', 'annotation', 'package',
            'exception', 'error', 'handler', 'listener', 'event',
            'thread', 'process', 'service', 'controller', 'model',
            'view', 'component', 'module', 'library', 'framework',
            'api', 'rest', 'http', 'json', 'xml', 'sql', 'database'
        ]

        entity_lower = entity.lower()

        # Check if entity contains reasonable programming terms
        for pattern in reasonable_patterns:
            if pattern in entity_lower:
                return True

        # Check if it's a reasonable identifier (starts with letter, contains only alphanumeric)
        if entity.replace('_', '').replace('-', '').isalnum() and entity[0].isalpha():
            return True

        return False

    def remove_duplicates(self, validated_triples):
        """Remove duplicate triples while preserving highest confidence ones"""
        seen_triples = {}

        for result in validated_triples:
            if not result['is_valid']:
                continue

            triple = result['normalized_triple']
            key = (triple['subject'], triple['relation'], triple['object'])

            if key not in seen_triples:
                seen_triples[key] = result
            else:
                # Keep the one with higher confidence
                existing_conf = seen_triples[key]['normalized_triple']['confidence']
                current_conf = triple['confidence']

                if current_conf > existing_conf:
                    seen_triples[key] = result

        unique_count = len(seen_triples)
        self.validation_stats['unique_valid_triples'] = unique_count

        return list(seen_triples.values())

    def create_nodes_and_edges(self, validated_triples):
        """Create separate node and edge lists for Neo4j"""
        nodes = {}
        edges = []

        for result in validated_triples:
            triple = result['normalized_triple']
            subject = triple['subject']
            obj = triple['object']
            relation = triple['relation']

            # Create nodes (avoid duplicates)
            if subject not in nodes:
                nodes[subject] = {
                    'name': subject,
                    'type': 'Entity',
                    'source_count': 0
                }
            nodes[subject]['source_count'] += 1

            if obj not in nodes:
                nodes[obj] = {
                    'name': obj,
                    'type': 'Entity',
                    'source_count': 0
                }
            nodes[obj]['source_count'] += 1

            # Create edge
            edges.append({
                'from': subject,
                'to': obj,
                'relation': relation,
                'confidence': triple['confidence'],
                'source_url': triple.get('source_url', ''),
                'source_type': triple.get('source_type', 'text')
            })

        return list(nodes.values()), edges

    def validate_all_data(self):
        """Main validation method"""
        print("=" * 50)
        print("✅ Step 5: Validating Nodes and Relations")
        print("=" * 50)

        try:
            # Load ontology and triples
            if not self.load_ontology():
                return None

            triples = self.load_triples()
            if not triples:
                print("❌ No triples to validate")
                return None

            print(f"🔍 Validating {len(triples)} triples...")

            # Validate each triple
            validated_results = []
            for i, triple in enumerate(triples):
                result = self.validate_triple(triple)
                validated_results.append(result)

                if (i + 1) % 100 == 0:
                    print(f"📈 Progress: {i + 1}/{len(triples)} triples validated")

            # Remove duplicates
            print("🧹 Removing duplicates...")
            unique_valid_results = self.remove_duplicates(validated_results)

            # Create nodes and edges
            print("🏗️ Creating nodes and edges...")
            nodes, edges = self.create_nodes_and_edges(unique_valid_results)

            # Save validated data
            output_data = {
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'ontology_source': str(self.ontology_file),
                    'triples_source': str(self.triples_file),
                    'validation_stats': dict(self.validation_stats),
                    'total_nodes': len(nodes),
                    'total_edges': len(edges)
                },
                'nodes': nodes,
                'edges': edges,
                'validation_details': {
                    'all_results': validated_results,
                    'valid_unique_results': unique_valid_results
                }
            }

            return self.save_validated_data(output_data)

        except Exception as e:
            print(f"❌ Error in validation process: {e}")
            return None

    def save_validated_data(self, data):
        """Save validated data to files"""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Save complete data
            main_file = self.output_dir / "validated_data.json"
            with open(main_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Save nodes separately for Neo4j
            nodes_file = self.output_dir / "nodes.json"
            with open(nodes_file, 'w', encoding='utf-8') as f:
                json.dump(data['nodes'], f, indent=2, ensure_ascii=False)

            # Save edges separately for Neo4j
            edges_file = self.output_dir / "edges.json"
            with open(edges_file, 'w', encoding='utf-8') as f:
                json.dump(data['edges'], f, indent=2, ensure_ascii=False)

            print(f"💾 Validated data saved to: {self.output_dir}")
            print(f"📁 Files: validated_data.json, nodes.json, edges.json")

            return data

        except Exception as e:
            print(f"❌ Error saving validated data: {e}")
            return None


def main():
    """Run data validation"""
    print("🎯 Knowledge Graph Data Validator")
    print("-" * 35)

    validator = DataValidator()
    validated_data = validator.validate_all_data()

    if validated_data:
        stats = validated_data['metadata']['validation_stats']

        print(f"\n📊 VALIDATION SUMMARY")
        print("-" * 25)
        print(f"Total triples processed: {stats.get('valid_triples', 0) + stats.get('invalid_triples', 0)}")
        print(f"Valid triples: {stats.get('valid_triples', 0)}")
        print(f"Invalid triples: {stats.get('invalid_triples', 0)}")
        print(f"Unique valid triples: {stats.get('unique_valid_triples', 0)}")
        print(f"Total nodes: {validated_data['metadata']['total_nodes']}")
        print(f"Total edges: {validated_data['metadata']['total_edges']}")
        print(f"Unknown relations: {stats.get('unknown_relations', 0)}")
        print(f"Low confidence: {stats.get('low_confidence', 0)}")

        success_rate = (stats.get('valid_triples', 0) / max(1,
                                                            stats.get('valid_triples', 0) + stats.get('invalid_triples',
                                                                                                      0))) * 100
        print(f"Success rate: {success_rate:.1f}%")

        print(f"\n✅ Ready for Step 6: Neo4j Loading!")
        return validated_data
    else:
        print("❌ Data validation failed!")
        return None


if __name__ == "__main__":
    main()