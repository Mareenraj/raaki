"""
Step 4: Dynamic Ontology Update using extracted triples
"""

import json
from datetime import datetime
from pathlib import Path
from collections import Counter


class DynamicOntologyUpdater:
    def __init__(self,
                 ontology_file="data/ontology/initial_ontology.json",
                 triples_file="data/extracted_triples/extracted_triples.json",
                 output_dir="data/ontology",
                 use_llm=False):
        self.ontology_file = Path(ontology_file)
        self.triples_file = Path(triples_file)
        self.output_dir = Path(output_dir)
        self.use_llm = use_llm
        self.model = None
        self.tokenizer = None

    def load_ontology(self):
        """Load existing ontology"""
        try:
            if not self.ontology_file.exists():
                raise FileNotFoundError(f"Ontology file not found: {self.ontology_file}")

            with open(self.ontology_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle both formats (with/without metadata)
            if 'ontology' in data:
                ontology = data['ontology']
                metadata = data.get('metadata', {})
            else:
                ontology = data
                metadata = {}

            print(
                f"📁 Loaded ontology with {len(ontology.get('classes', []))} classes, {len(ontology.get('relations', []))} relations")
            return ontology, metadata

        except Exception as e:
            print(f"❌ Error loading ontology: {e}")
            return None, None

    def load_triples(self):
        """Load extracted triples"""
        try:
            if not self.triples_file.exists():
                raise FileNotFoundError(f"Triples file not found: {self.triples_file}")

            with open(self.triples_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Get all triples
            if 'triples' in data:
                all_triples = data['triples'].get('all', [])
                high_conf_triples = data['triples'].get('high_confidence', [])
                medium_conf_triples = data['triples'].get('medium_confidence', [])
            else:
                all_triples = data if isinstance(data, list) else []
                high_conf_triples = [t for t in all_triples if t.get('confidence', 0) > 0.7]
                medium_conf_triples = [t for t in all_triples if 0.5 <= t.get('confidence', 0) <= 0.7]

            print(f"📊 Loaded {len(all_triples)} triples ({len(high_conf_triples)} high confidence)")
            return all_triples, high_conf_triples, medium_conf_triples

        except Exception as e:
            print(f"❌ Error loading triples: {e}")
            return [], [], []

    def analyze_new_concepts(self, triples, existing_ontology):
        """Analyze triples to find new classes and relations"""
        existing_classes = set(existing_ontology.get('classes', []))
        existing_relations = set(existing_ontology.get('relations', []))

        # Extract entities and relations from triples
        new_classes = set()
        new_relations = set()
        relation_frequency = Counter()
        class_frequency = Counter()

        for triple in triples:
            subject = triple.get('subject', '').strip()
            relation = triple.get('relation', '').strip()
            obj = triple.get('object', '').strip()
            confidence = triple.get('confidence', 0)

            # Only consider high confidence triples for new concepts
            if confidence > 0.6:
                # Normalize and clean entity names
                subject = self.normalize_entity_name(subject)
                obj = self.normalize_entity_name(obj)
                relation = self.normalize_relation_name(relation)

                if subject and subject not in existing_classes:
                    new_classes.add(subject)
                    class_frequency[subject] += confidence

                if obj and obj not in existing_classes:
                    new_classes.add(obj)
                    class_frequency[obj] += confidence

                if relation and relation not in existing_relations:
                    new_relations.add(relation)
                    relation_frequency[relation] += confidence

        # Filter by frequency/confidence threshold
        min_frequency = 1.0  # Minimum total confidence score
        filtered_classes = {cls for cls in new_classes if class_frequency[cls] >= min_frequency}
        filtered_relations = {rel for rel in new_relations if relation_frequency[rel] >= min_frequency}

        return filtered_classes, filtered_relations, class_frequency, relation_frequency

    def normalize_entity_name(self, name):
        """Normalize entity names for consistency"""
        if not name:
            return ""

        # Remove common prefixes/suffixes and normalize
        name = name.strip()

        # Convert to PascalCase for classes
        if name.lower() in ['string', 'int', 'integer', 'double', 'float', 'boolean', 'char']:
            return f"DataType_{name.capitalize()}"

        # Handle common programming terms
        replacements = {
            'arraylist': 'ArrayList',
            'hashmap': 'HashMap',
            'hashset': 'HashSet',
            'linkedlist': 'LinkedList',
            'exception': 'Exception',
            'interface': 'Interface',
            'class': 'Class',
            'method': 'Method'
        }

        lower_name = name.lower()
        if lower_name in replacements:
            return replacements[lower_name]

        # Capitalize first letter for class names
        return name.capitalize() if name else ""

    def normalize_relation_name(self, relation):
        """Normalize relation names for consistency"""
        if not relation:
            return ""

        # Convert to camelCase for relations
        relation = relation.strip().lower()

        # Handle common relation variations
        replacements = {
            'is_type_of': 'isTypeOf',
            'has_method': 'hasMethod',
            'has_property': 'hasProperty',
            'throws_exception': 'throwsException',
            'belongs_to': 'belongsTo',
            'part_of': 'partOf',
            'depends_on': 'dependsOn',
            'is_a': 'isA',
            'has_a': 'hasA'
        }

        if relation in replacements:
            return replacements[relation]

        # Convert snake_case to camelCase
        if '_' in relation:
            parts = relation.split('_')
            return parts[0] + ''.join(word.capitalize() for word in parts[1:])

        return relation

    def generate_llm_updates(self, new_classes, new_relations, existing_ontology):
        """Use LLM to suggest ontology updates (if available)"""
        if not self.use_llm or not self.model:
            return new_classes, new_relations

        try:
            prompt = f"""Given this existing ontology and new concepts found in code:

Existing Classes: {existing_ontology.get('classes', [])}
Existing Relations: {existing_ontology.get('relations', [])}

New Classes Found: {list(new_classes)}
New Relations Found: {list(new_relations)}

Suggest which new classes and relations should be added to create a better software engineering ontology. Return only valid additions.

Respond in JSON format:
{{
  "add_classes": ["Class1", "Class2"],
  "add_relations": ["relation1", "relation2"]
}}"""

            # This would use the LLM if available
            # For now, return the original suggestions
            return new_classes, new_relations

        except Exception as e:
            print(f"⚠️ LLM update failed, using rule-based approach: {e}")
            return new_classes, new_relations

    def update_ontology(self, existing_ontology, new_classes, new_relations):
        """Create updated ontology with new concepts"""
        updated_ontology = {
            "classes": list(set(existing_ontology.get('classes', []) + list(new_classes))),
            "relations": list(set(existing_ontology.get('relations', []) + list(new_relations)))
        }

        # Sort for consistency
        updated_ontology["classes"].sort()
        updated_ontology["relations"].sort()

        return updated_ontology

    def save_updated_ontology(self, updated_ontology, original_metadata, new_classes, new_relations):
        """Save the updated ontology"""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            filepath = self.output_dir / "updated_ontology.json"

            output_data = {
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "updated_from": str(self.ontology_file),
                    "triples_source": str(self.triples_file),
                    "original_classes": len(
                        original_metadata.get('classes', [])) if 'classes' in original_metadata else len(
                        updated_ontology['classes']) - len(new_classes),
                    "original_relations": len(
                        original_metadata.get('relations', [])) if 'relations' in original_metadata else len(
                        updated_ontology['relations']) - len(new_relations),
                    "added_classes": len(new_classes),
                    "added_relations": len(new_relations),
                    "total_classes": len(updated_ontology['classes']),
                    "total_relations": len(updated_ontology['relations']),
                    "version": "2.0"
                },
                "changes": {
                    "new_classes": list(new_classes),
                    "new_relations": list(new_relations)
                },
                "ontology": updated_ontology
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            print(f"💾 Updated ontology saved to: {filepath}")
            return filepath

        except Exception as e:
            print(f"❌ Error saving updated ontology: {e}")
            return None

    def update_ontology_dynamically(self):
        """Main method to dynamically update ontology"""
        print("=" * 50)
        print("🔄 Step 4: Dynamic Ontology Update")
        print("=" * 50)

        try:
            # Load existing data
            existing_ontology, metadata = self.load_ontology()
            if not existing_ontology:
                return None

            all_triples, high_conf_triples, medium_conf_triples = self.load_triples()
            if not all_triples:
                print("⚠️ No triples found, using original ontology")
                return existing_ontology

            # Analyze for new concepts (use high and medium confidence triples)
            analysis_triples = high_conf_triples + medium_conf_triples
            new_classes, new_relations, class_freq, relation_freq = self.analyze_new_concepts(
                analysis_triples, existing_ontology
            )

            print(f"🔍 Found {len(new_classes)} new classes, {len(new_relations)} new relations")

            if not new_classes and not new_relations:
                print("✅ No significant new concepts found. Ontology is comprehensive!")
                return existing_ontology

            # Apply LLM filtering if available
            if self.use_llm:
                new_classes, new_relations = self.generate_llm_updates(
                    new_classes, new_relations, existing_ontology
                )

            # Create updated ontology
            updated_ontology = self.update_ontology(existing_ontology, new_classes, new_relations)

            # Save updated ontology
            filepath = self.save_updated_ontology(updated_ontology, metadata, new_classes, new_relations)

            if filepath:
                print(f"\n📊 UPDATE SUMMARY")
                print("-" * 30)
                print(f"Original classes: {len(existing_ontology.get('classes', []))}")
                print(f"Original relations: {len(existing_ontology.get('relations', []))}")
                print(f"Added classes: {len(new_classes)}")
                print(f"Added relations: {len(new_relations)}")
                print(f"Final classes: {len(updated_ontology['classes'])}")
                print(f"Final relations: {len(updated_ontology['relations'])}")

                return updated_ontology

            return None

        except Exception as e:
            print(f"❌ Error in ontology update process: {e}")
            return None


def main():
    """Run dynamic ontology update"""
    print("🎯 Knowledge Graph Dynamic Ontology Updater")
    print("-" * 45)

    updater = DynamicOntologyUpdater(use_llm=False)  # Use rule-based by default
    updated_ontology = updater.update_ontology_dynamically()

    if updated_ontology:
        print(f"\n🔄 NEW CONCEPTS ADDED")
        print("-" * 25)

        # Load original for comparison
        original_ontology, _ = updater.load_ontology()
        if original_ontology:
            original_classes = set(original_ontology.get('classes', []))
            original_relations = set(original_ontology.get('relations', []))
            new_classes = set(updated_ontology['classes']) - original_classes
            new_relations = set(updated_ontology['relations']) - original_relations

            if new_classes:
                print(f"\n🏷️  New Classes:")
                for cls in sorted(new_classes):
                    print(f"   + {cls}")

            if new_relations:
                print(f"\n🔗 New Relations:")
                for rel in sorted(new_relations):
                    print(f"   + {rel}")

        print(f"\n✅ Ready for Step 5: Data Validation!")
        return updated_ontology
    else:
        print("❌ Ontology update failed!")
        return None


if __name__ == "__main__":
    main()