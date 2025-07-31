"""
Training Data Manager for REBEL Fine-tuning
Handles creation, validation, and management of training datasets
"""

import json
import csv
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import re


class TrainingDataManager:
    def __init__(self, data_dir="data/training"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.manual_data_file = self.data_dir / "manual_training_data.json"
        self.generated_data_file = self.data_dir / "generated_training_data.json"
        self.combined_data_file = self.data_dir / "combined_training_data.json"
        self.validation_file = self.data_dir / "validation_data.json"

        # CSV exports for easy editing
        self.manual_csv_file = self.data_dir / "manual_training_data.csv"
        self.validation_csv_file = self.data_dir / "validation_data.csv"

    def create_manual_training_data(self):
        """Create comprehensive manual training data for Java domain"""

        manual_training_examples = [
            # ========================================
            # CLASS INHERITANCE & IMPLEMENTATION
            # ========================================
            {
                "id": "java_001",
                "category": "inheritance",
                "text": "ArrayList class extends AbstractList and implements List interface",
                "relations": [
                    {"subject": "ArrayList", "relation": "extends", "object": "AbstractList"},
                    {"subject": "ArrayList", "relation": "implements", "object": "List"}
                ],
                "rebel_format": "<triplet> extends <subj> ArrayList <obj> AbstractList <triplet> implements <subj> ArrayList <obj> List",
                "confidence": 1.0,
                "source": "manual"
            },
            {
                "id": "java_002",
                "category": "inheritance",
                "text": "HashMap extends AbstractMap and implements Map interface",
                "relations": [
                    {"subject": "HashMap", "relation": "extends", "object": "AbstractMap"},
                    {"subject": "HashMap", "relation": "implements", "object": "Map"}
                ],
                "rebel_format": "<triplet> extends <subj> HashMap <obj> AbstractMap <triplet> implements <subj> HashMap <obj> Map",
                "confidence": 1.0,
                "source": "manual"
            },
            {
                "id": "java_003",
                "category": "inheritance",
                "text": "String class extends Object and implements Serializable",
                "relations": [
                    {"subject": "String", "relation": "extends", "object": "Object"},
                    {"subject": "String", "relation": "implements", "object": "Serializable"}
                ],
                "rebel_format": "<triplet> extends <subj> String <obj> Object <triplet> implements <subj> String <obj> Serializable",
                "confidence": 1.0,
                "source": "manual"
            },

            # ========================================
            # METHOD RELATIONSHIPS
            # ========================================
            {
                "id": "java_004",
                "category": "methods",
                "text": "String class has length method that returns int type",
                "relations": [
                    {"subject": "String", "relation": "hasMethod", "object": "length"},
                    {"subject": "length", "relation": "returns", "object": "int"}
                ],
                "rebel_format": "<triplet> hasMethod <subj> String <obj> length <triplet> returns <subj> length <obj> int",
                "confidence": 1.0,
                "source": "manual"
            },
            {
                "id": "java_005",
                "category": "methods",
                "text": "ArrayList has add method that accepts Object parameter",
                "relations": [
                    {"subject": "ArrayList", "relation": "hasMethod", "object": "add"},
                    {"subject": "add", "relation": "accepts", "object": "Object"}
                ],
                "rebel_format": "<triplet> hasMethod <subj> ArrayList <obj> add <triplet> accepts <subj> add <obj> Object",
                "confidence": 1.0,
                "source": "manual"
            },

            # ========================================
            # PACKAGE RELATIONSHIPS
            # ========================================
            {
                "id": "java_006",
                "category": "packages",
                "text": "ArrayList belongs to java.util package",
                "relations": [
                    {"subject": "ArrayList", "relation": "belongsTo", "object": "java.util"}
                ],
                "rebel_format": "<triplet> belongsTo <subj> ArrayList <obj> java.util",
                "confidence": 1.0,
                "source": "manual"
            },
            {
                "id": "java_007",
                "category": "packages",
                "text": "String class is part of java.lang package",
                "relations": [
                    {"subject": "String", "relation": "belongsTo", "object": "java.lang"}
                ],
                "rebel_format": "<triplet> belongsTo <subj> String <obj> java.lang",
                "confidence": 1.0,
                "source": "manual"
            },

            # ========================================
            # EXCEPTION HANDLING
            # ========================================
            {
                "id": "java_008",
                "category": "exceptions",
                "text": "IOException extends Exception and indicates input output error",
                "relations": [
                    {"subject": "IOException", "relation": "extends", "object": "Exception"},
                    {"subject": "IOException", "relation": "indicates", "object": "input output error"}
                ],
                "rebel_format": "<triplet> extends <subj> IOException <obj> Exception <triplet> indicates <subj> IOException <obj> input output error",
                "confidence": 1.0,
                "source": "manual"
            },

            # ========================================
            # SPRING FRAMEWORK
            # ========================================
            {
                "id": "spring_001",
                "category": "spring",
                "text": "Controller annotation marks Spring MVC controller classes",
                "relations": [
                    {"subject": "Controller", "relation": "isTypeOf", "object": "Spring annotation"},
                    {"subject": "Controller", "relation": "marks", "object": "MVC controller"}
                ],
                "rebel_format": "<triplet> isTypeOf <subj> Controller <obj> Spring annotation <triplet> marks <subj> Controller <obj> MVC controller",
                "confidence": 1.0,
                "source": "manual"
            },
            {
                "id": "spring_002",
                "category": "spring",
                "text": "Autowired annotation enables dependency injection in Spring",
                "relations": [
                    {"subject": "Autowired", "relation": "enables", "object": "dependency injection"},
                    {"subject": "Autowired", "relation": "belongsTo", "object": "Spring"}
                ],
                "rebel_format": "<triplet> enables <subj> Autowired <obj> dependency injection <triplet> belongsTo <subj> Autowired <obj> Spring",
                "confidence": 1.0,
                "source": "manual"
            },

            # ========================================
            # DATA TYPES
            # ========================================
            {
                "id": "types_001",
                "category": "data_types",
                "text": "int is primitive data type that stores integer values",
                "relations": [
                    {"subject": "int", "relation": "isTypeOf", "object": "primitive type"},
                    {"subject": "int", "relation": "stores", "object": "integer values"}
                ],
                "rebel_format": "<triplet> isTypeOf <subj> int <obj> primitive type <triplet> stores <subj> int <obj> integer values",
                "confidence": 1.0,
                "source": "manual"
            },

            # ========================================
            # COLLECTIONS FRAMEWORK
            # ========================================
            {
                "id": "collections_001",
                "category": "collections",
                "text": "List interface extends Collection and provides ordered elements",
                "relations": [
                    {"subject": "List", "relation": "extends", "object": "Collection"},
                    {"subject": "List", "relation": "provides", "object": "ordered elements"}
                ],
                "rebel_format": "<triplet> extends <subj> List <obj> Collection <triplet> provides <subj> List <obj> ordered elements",
                "confidence": 1.0,
                "source": "manual"
            },

            # ========================================
            # DESIGN PATTERNS
            # ========================================
            {
                "id": "patterns_001",
                "category": "design_patterns",
                "text": "Singleton pattern ensures single instance of class",
                "relations": [
                    {"subject": "Singleton", "relation": "isTypeOf", "object": "design pattern"},
                    {"subject": "Singleton", "relation": "ensures", "object": "single instance"}
                ],
                "rebel_format": "<triplet> isTypeOf <subj> Singleton <obj> design pattern <triplet> ensures <subj> Singleton <obj> single instance",
                "confidence": 1.0,
                "source": "manual"
            },

            # ========================================
            # MULTITHREADING
            # ========================================
            {
                "id": "threading_001",
                "category": "threading",
                "text": "Thread class implements Runnable interface for concurrent execution",
                "relations": [
                    {"subject": "Thread", "relation": "implements", "object": "Runnable"},
                    {"subject": "Thread", "relation": "enables", "object": "concurrent execution"}
                ],
                "rebel_format": "<triplet> implements <subj> Thread <obj> Runnable <triplet> enables <subj> Thread <obj> concurrent execution",
                "confidence": 1.0,
                "source": "manual"
            },

            # ========================================
            # TESTING FRAMEWORKS
            # ========================================
            {
                "id": "testing_001",
                "category": "testing",
                "text": "JUnit framework provides Test annotation for unit testing",
                "relations": [
                    {"subject": "JUnit", "relation": "isTypeOf", "object": "testing framework"},
                    {"subject": "JUnit", "relation": "provides", "object": "Test annotation"}
                ],
                "rebel_format": "<triplet> isTypeOf <subj> JUnit <obj> testing framework <triplet> provides <subj> JUnit <obj> Test annotation",
                "confidence": 1.0,
                "source": "manual"
            }
        ]

        return manual_training_examples

    def create_validation_data(self):
        """Create validation dataset for testing model performance"""

        validation_examples = [
            {
                "id": "val_001",
                "text": "LinkedList implements List and Deque interfaces",
                "expected_relations": [
                    {"subject": "LinkedList", "relation": "implements", "object": "List"},
                    {"subject": "LinkedList", "relation": "implements", "object": "Deque"}
                ]
            },
            {
                "id": "val_002",
                "text": "StringBuilder class has append method returning StringBuilder",
                "expected_relations": [
                    {"subject": "StringBuilder", "relation": "hasMethod", "object": "append"},
                    {"subject": "append", "relation": "returns", "object": "StringBuilder"}
                ]
            },
            {
                "id": "val_003",
                "text": "Service annotation marks Spring service layer components",
                "expected_relations": [
                    {"subject": "Service", "relation": "isTypeOf", "object": "Spring annotation"},
                    {"subject": "Service", "relation": "marks", "object": "service layer"}
                ]
            },
            {
                "id": "val_004",
                "text": "FileInputStream extends InputStream for file reading operations",
                "expected_relations": [
                    {"subject": "FileInputStream", "relation": "extends", "object": "InputStream"},
                    {"subject": "FileInputStream", "relation": "usedFor", "object": "file reading"}
                ]
            },
            {
                "id": "val_005",
                "text": "Optional class prevents null pointer exceptions",
                "expected_relations": [
                    {"subject": "Optional", "relation": "prevents", "object": "null pointer exceptions"}
                ]
            }
        ]

        return validation_examples

    def save_manual_training_data(self):
        """Save manual training data to JSON and CSV files"""
        try:
            manual_data = self.create_manual_training_data()

            # Save JSON format
            training_dataset = {
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "total_examples": len(manual_data),
                    "categories": list(set(ex["category"] for ex in manual_data)),
                    "source": "manual_curation",
                    "version": "1.0"
                },
                "examples": manual_data
            }

            with open(self.manual_data_file, 'w', encoding='utf-8') as f:
                json.dump(training_dataset, f, indent=2, ensure_ascii=False)

            # Save CSV format for easy editing
            csv_data = []
            for example in manual_data:
                for relation in example["relations"]:
                    csv_data.append({
                        "id": example["id"],
                        "category": example["category"],
                        "text": example["text"],
                        "subject": relation["subject"],
                        "relation": relation["relation"],
                        "object": relation["object"],
                        "confidence": example["confidence"],
                        "source": example["source"]
                    })

            with open(self.manual_csv_file, 'w', newline='', encoding='utf-8') as f:
                if csv_data:
                    writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                    writer.writeheader()
                    writer.writerows(csv_data)

            print(f"💾 Manual training data saved:")
            print(f"   📄 JSON: {self.manual_data_file}")
            print(f"   📊 CSV: {self.manual_csv_file}")
            print(f"   📈 Examples: {len(manual_data)}")
            print(f"   🏷️ Categories: {len(training_dataset['metadata']['categories'])}")

            return training_dataset

        except Exception as e:
            print(f"❌ Error saving manual training data: {e}")
            return None

    def save_validation_data(self):
        """Save validation data"""
        try:
            validation_data = self.create_validation_data()

            validation_dataset = {
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "total_examples": len(validation_data),
                    "purpose": "model_validation",
                    "version": "1.0"
                },
                "examples": validation_data
            }

            with open(self.validation_file, 'w', encoding='utf-8') as f:
                json.dump(validation_dataset, f, indent=2, ensure_ascii=False)

            # CSV format
            csv_data = []
            for example in validation_data:
                for relation in example["expected_relations"]:
                    csv_data.append({
                        "id": example["id"],
                        "text": example["text"],
                        "subject": relation["subject"],
                        "relation": relation["relation"],
                        "object": relation["object"]
                    })

            with open(self.validation_csv_file, 'w', newline='', encoding='utf-8') as f:
                if csv_data:
                    writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                    writer.writeheader()
                    writer.writerows(csv_data)

            print(f"💾 Validation data saved:")
            print(f"   📄 JSON: {self.validation_file}")
            print(f"   📊 CSV: {self.validation_csv_file}")
            print(f"   🧪 Examples: {len(validation_data)}")

            return validation_dataset

        except Exception as e:
            print(f"❌ Error saving validation data: {e}")
            return None

    def generate_training_data_from_pipeline(self):
        """Generate training data from your existing pipeline results"""
        try:
            # Load high-confidence triples from your pipeline
            triples_file = Path("data/extracted_triples/extracted_triples_improved.json")
            if not triples_file.exists():
                print("⚠️ No pipeline triples found. Run main pipeline first.")
                return []

            with open(triples_file, 'r', encoding='utf-8') as f:
                triples_data = json.load(f)

            high_conf_triples = triples_data.get('triples', {}).get('high_confidence', [])

            generated_examples = []
            for i, triple in enumerate(high_conf_triples[:50]):  # Limit to 50 best examples
                if triple.get('confidence', 0) > 0.85:  # Very high confidence only

                    source_text = triple.get('source_text', '').strip()
                    if len(source_text) > 20 and len(source_text) < 200:  # Good length

                        # Clean up the text
                        clean_text = re.sub(r'\s+', ' ', source_text)
                        clean_text = clean_text.replace('...', '').strip()

                        rebel_format = f"<triplet> {triple['relation']} <subj> {triple['subject']} <obj> {triple['object']}"

                        generated_examples.append({
                            "id": f"gen_{i + 1:03d}",
                            "category": "generated",
                            "text": clean_text,
                            "relations": [{
                                "subject": triple['subject'],
                                "relation": triple['relation'],
                                "object": triple['object']
                            }],
                            "rebel_format": rebel_format,
                            "confidence": triple['confidence'],
                            "source": "pipeline_generated",
                            "original_source": triple.get('source_url', 'unknown')
                        })

            if generated_examples:
                generated_dataset = {
                    "metadata": {
                        "created_at": datetime.now().isoformat(),
                        "total_examples": len(generated_examples),
                        "source": "pipeline_extraction",
                        "min_confidence": 0.85,
                        "version": "1.0"
                    },
                    "examples": generated_examples
                }

                with open(self.generated_data_file, 'w', encoding='utf-8') as f:
                    json.dump(generated_dataset, f, indent=2, ensure_ascii=False)

                print(f"🔄 Generated training data from pipeline:")
                print(f"   📁 File: {self.generated_data_file}")
                print(f"   📊 Examples: {len(generated_examples)}")

                return generated_dataset
            else:
                print("⚠️ No suitable examples found in pipeline data")
                return None

        except Exception as e:
            print(f"❌ Error generating training data: {e}")
            return None

    def combine_training_datasets(self):
        """Combine manual and generated training data"""
        try:
            combined_examples = []

            # Load manual data
            if self.manual_data_file.exists():
                with open(self.manual_data_file, 'r', encoding='utf-8') as f:
                    manual_data = json.load(f)
                combined_examples.extend(manual_data['examples'])
                print(f"📄 Added {len(manual_data['examples'])} manual examples")

            # Load generated data
            if self.generated_data_file.exists():
                with open(self.generated_data_file, 'r', encoding='utf-8') as f:
                    generated_data = json.load(f)
                combined_examples.extend(generated_data['examples'])
                print(f"🔄 Added {len(generated_data['examples'])} generated examples")

            if combined_examples:
                combined_dataset = {
                    "metadata": {
                        "created_at": datetime.now().isoformat(),
                        "total_examples": len(combined_examples),
                        "manual_examples": len([ex for ex in combined_examples if ex['source'] == 'manual']),
                        "generated_examples": len(
                            [ex for ex in combined_examples if ex['source'] == 'pipeline_generated']),
                        "categories": list(set(ex['category'] for ex in combined_examples)),
                        "version": "1.0"
                    },
                    "examples": combined_examples
                }

                with open(self.combined_data_file, 'w', encoding='utf-8') as f:
                    json.dump(combined_dataset, f, indent=2, ensure_ascii=False)

                print(f"🔗 Combined training dataset created:")
                print(f"   📁 File: {self.combined_data_file}")
                print(f"   📊 Total examples: {len(combined_examples)}")
                print(f"   🏷️ Categories: {combined_dataset['metadata']['categories']}")

                return combined_dataset
            else:
                print("❌ No training data found to combine")
                return None

        except Exception as e:
            print(f"❌ Error combining datasets: {e}")
            return None

    def validate_training_data(self, dataset_file=None):
        """Validate training data quality"""
        try:
            if dataset_file is None:
                dataset_file = self.combined_data_file

            if not Path(dataset_file).exists():
                print(f"❌ Dataset file not found: {dataset_file}")
                return False

            with open(dataset_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            examples = data['examples']
            issues = []

            for example in examples:
                # Check required fields
                if not all(key in example for key in ['id', 'text', 'relations', 'rebel_format']):
                    issues.append(f"Missing required fields in {example.get('id', 'unknown')}")

                # Check text length
                if len(example.get('text', '')) < 10:
                    issues.append(f"Text too short in {example.get('id', 'unknown')}")

                # Check relations
                relations = example.get('relations', [])
                if not relations:
                    issues.append(f"No relations in {example.get('id', 'unknown')}")

                for relation in relations:
                    if not all(key in relation for key in ['subject', 'relation', 'object']):
                        issues.append(f"Invalid relation format in {example.get('id', 'unknown')}")

            print(f"🔍 TRAINING DATA VALIDATION REPORT")
            print("=" * 40)
            print(f"📊 Total examples: {len(examples)}")
            print(f"❌ Issues found: {len(issues)}")

            if issues:
                print("\n🚨 Issues:")
                for issue in issues[:10]:  # Show first 10 issues
                    print(f"   • {issue}")
                if len(issues) > 10:
                    print(f"   ... and {len(issues) - 10} more issues")
                return False
            else:
                print("✅ All validation checks passed!")
                return True

        except Exception as e:
            print(f"❌ Error validating training data: {e}")
            return False

    def export_for_editing(self):
        """Export training data in easily editable formats"""
        try:
            print("📤 Exporting training data for manual editing...")

            # Create template for adding new examples
            template = {
                "template_example": {
                    "id": "YOUR_ID_HERE",
                    "category": "YOUR_CATEGORY",
                    "text": "Your training sentence here",
                    "relations": [
                        {
                            "subject": "Subject",
                            "relation": "relationName",
                            "object": "Object"
                        }
                    ],
                    "rebel_format": "<triplet> relationName <subj> Subject <obj> Object",
                    "confidence": 1.0,
                    "source": "manual"
                }
            }

            template_file = self.data_dir / "training_template.json"
            with open(template_file, 'w', encoding='utf-8') as f:
                json.dump(template, f, indent=2, ensure_ascii=False)

            print(f"📋 Template created: {template_file}")
            print(f"📊 CSV files available for editing:")
            print(f"   • {self.manual_csv_file}")
            print(f"   • {self.validation_csv_file}")

            return True

        except Exception as e:
            print(f"❌ Error exporting: {e}")
            return False

    def create_all_training_files(self):
        """Create complete training data setup"""
        print("🎯 Creating Complete Training Data Setup")
        print("=" * 45)

        # Step 1: Create manual training data
        print("\n📝 Step 1: Creating manual training data...")
        manual_data = self.save_manual_training_data()

        # Step 2: Create validation data
        print("\n🧪 Step 2: Creating validation data...")
        validation_data = self.save_validation_data()

        # Step 3: Generate data from pipeline
        print("\n🔄 Step 3: Generating data from pipeline...")
        generated_data = self.generate_training_data_from_pipeline()

        # Step 4: Combine datasets
        print("\n🔗 Step 4: Combining datasets...")
        combined_data = self.combine_training_datasets()

        # Step 5: Validate data
        print("\n🔍 Step 5: Validating data quality...")
        is_valid = self.validate_training_data()

        # Step 6: Export for editing
        print("\n📤 Step 6: Exporting for manual editing...")
        self.export_for_editing()

        print(f"\n✅ TRAINING DATA SETUP COMPLETED!")
        print("=" * 35)
        print(f"📁 Files created in: {self.data_dir}")
        print(f"📊 Ready for REBEL fine-tuning: {'✅' if is_valid else '❌'}")

        return is_valid


def main():
    """Run training data management"""
    print("🎯 REBEL Training Data Manager")
    print("-" * 35)

    manager = TrainingDataManager()
    success = manager.create_all_training_files()

    if success:
        print(f"\n🚀 NEXT STEPS:")
        print("1. Review generated training data files")
        print("2. Edit CSV files to add more examples if needed")
        print("3. Run fine-tuning with: python scripts/finetune_rebel.py")
        print("4. The training data is ready for use!")
    else:
        print(f"\n⚠️ Some issues found. Review and fix before fine-tuning.")


if __name__ == "__main__":
    main()