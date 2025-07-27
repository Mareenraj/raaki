"""
Improved Triple Extraction with Better Error Handling and Bug Fixes
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovedTripleExtractor:
    def __init__(self,
                 scraped_data_file="data/scraped_content/w3schools_java_tutorials.json",
                 output_dir="data/extracted_triples",
                 use_rebel=True,
                 use_improved_patterns=True):
        self.scraped_data_file = Path(scraped_data_file)
        self.output_dir = Path(output_dir)
        self.use_rebel = use_rebel
        self.use_improved_patterns = use_improved_patterns
        self.rebel_model = None
        self.rebel_tokenizer = None

        # Load improved patterns
        self.extraction_patterns = self.get_improved_patterns()

    def load_rebel_model(self):
        """Load REBEL model for better relation extraction"""
        try:
            print("🤖 Loading REBEL model for advanced extraction...")
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch

            model_name = "Babelscape/rebel-large"
            print(f"📥 Downloading {model_name}...")

            self.rebel_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.rebel_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

            # Move to CPU explicitly to avoid device issues
            if torch.cuda.is_available():
                print("🎮 CUDA available, using GPU")
                self.rebel_model = self.rebel_model.cuda()
            else:
                print("💻 Using CPU for inference")
                self.rebel_model = self.rebel_model.cpu()

            print("✅ REBEL model loaded successfully!")
            return True

        except ImportError as e:
            print("⚠️ Transformers not available. Install with:")
            print("pip install torch transformers sentencepiece protobuf")
            print("🔄 Falling back to improved pattern matching...")
            return False
        except Exception as e:
            print(f"⚠️ Error loading REBEL: {e}")
            print("🔄 Using improved pattern matching...")
            return False

    def get_improved_patterns(self):
        """Define comprehensive extraction patterns for Java concepts"""
        return {
            # Class and Interface Definitions - Fixed lambda functions
            "class_definition": [
                (r'\bclass\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([\w,\s]+))?',
                 self._extract_class_relations)
            ],

            "interface_definition": [
                (r'\binterface\s+(\w+)(?:\s+extends\s+([\w,\s]+))?',
                 self._extract_interface_relations)
            ],

            # Method Definitions and Calls
            "method_definition": [
                (r'\b(?:public|private|protected)?\s*(?:static)?\s*(\w+)\s+(\w+)\s*\([^)]*\)',
                 self._extract_method_relations)
            ],

            "method_call": [
                (r'(\w+)\.(\w+)\s*\(',
                 self._extract_method_call_relations)
            ],

            # Object Creation and Instantiation
            "instantiation": [
                (r'new\s+(\w+)\s*\(',
                 self._extract_instantiation_relations)
            ],

            # Collections and Data Structures
            "collection_usage": [
                (r'\b(ArrayList|HashMap|HashSet|LinkedList|TreeMap|Vector|Stack|Queue)\b',
                 self._extract_collection_relations)
            ],

            # Data Types
            "data_types": [
                (r'\b(String|int|double|float|boolean|char|long|byte|short)\b',
                 self._extract_datatype_relations)
            ],

            # Exception Handling
            "exception_handling": [
                (r'\b(?:try|catch|finally|throw|throws)\b.*?\b(\w+(?:Exception|Error))\b',
                 self._extract_exception_relations)
            ],

            # Annotations
            "annotations": [
                (r'@(\w+)',
                 self._extract_annotation_relations)
            ],

            # Import Statements
            "imports": [
                (r'\bimport\s+([\w.]+)\.(\w+)',
                 self._extract_import_relations)
            ],

            # Framework Concepts
            "spring_concepts": [
                (r'@(Controller|Service|Repository|Component|Entity|RestController)',
                 self._extract_spring_relations)
            ],

            "web_concepts": [
                (r'\b(REST|HTTP|JSON|XML|API|GET|POST|PUT|DELETE)\b',
                 self._extract_web_relations)
            ],

            # Design Patterns
            "design_patterns": [
                (r'\b(Singleton|Factory|Observer|Strategy|Builder|Adapter)\b',
                 self._extract_pattern_relations)
            ]
        }

    # Fixed extraction methods - no more lambda functions
    def _extract_class_relations(self, match):
        """Extract class-related triples"""
        triples = [(match.group(1), "isTypeOf", "Class")]
        if match.group(2):
            triples.append((match.group(1), "extends", match.group(2)))
        if match.group(3):
            interfaces = [iface.strip() for iface in match.group(3).split(",") if iface.strip()]
            triples.extend([(match.group(1), "implements", iface) for iface in interfaces])
        return triples

    def _extract_interface_relations(self, match):
        """Extract interface-related triples"""
        triples = [(match.group(1), "isTypeOf", "Interface")]
        if match.group(2):
            parents = [parent.strip() for parent in match.group(2).split(",") if parent.strip()]
            triples.extend([(match.group(1), "extends", parent) for parent in parents])
        return triples

    def _extract_method_relations(self, match):
        """Extract method-related triples"""
        return [(match.group(2), "hasReturnType", match.group(1)),
                (match.group(2), "isTypeOf", "Method")]

    def _extract_method_call_relations(self, match):
        """Extract method call triples"""
        return [(match.group(1), "hasMethod", match.group(2))]

    def _extract_instantiation_relations(self, match):
        """Extract instantiation triples"""
        return [("Code", "instantiates", match.group(1))]

    def _extract_collection_relations(self, match):
        """Extract collection-related triples"""
        return [(match.group(1), "isTypeOf", "Collection"),
                (match.group(1), "belongsTo", "JavaCollections")]

    def _extract_datatype_relations(self, match):
        """Extract data type triples"""
        return [(match.group(1), "isTypeOf", "DataType"),
                (match.group(1), "belongsTo", "JavaPrimitives")]

    def _extract_exception_relations(self, match):
        """Extract exception handling triples"""
        return [(match.group(1), "isTypeOf", "Exception"),
                ("Code", "handles", match.group(1))]

    def _extract_annotation_relations(self, match):
        """Extract annotation triples"""
        return [(match.group(1), "isTypeOf", "Annotation"),
                ("Code", "uses", match.group(1))]

    def _extract_import_relations(self, match):
        """Extract import statement triples"""
        return [(match.group(2), "belongsTo", match.group(1)),
                ("Code", "imports", match.group(2))]

    def _extract_spring_relations(self, match):
        """Extract Spring framework triples"""
        return [(match.group(1), "isTypeOf", "SpringAnnotation"),
                (match.group(1), "belongsTo", "SpringFramework")]

    def _extract_web_relations(self, match):
        """Extract web concept triples"""
        return [(match.group(1), "isTypeOf", "WebConcept"),
                (match.group(1), "belongsTo", "WebDevelopment")]

    def _extract_pattern_relations(self, match):
        """Extract design pattern triples"""
        return [(match.group(1), "isTypeOf", "DesignPattern"),
                (match.group(1), "belongsTo", "SoftwareDesign")]

    def preprocess_text(self, text):
        """Clean and preprocess text for better extraction"""
        if not text:
            return ""

        try:
            # Remove HTML entities
            text = re.sub(r'&[a-zA-Z]+;', ' ', text)

            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)

            # Remove navigation and footer noise
            noise_patterns = [
                r'Try it Yourself.*?»',
                r'❮\s*Previous\s*Next\s*❯',
                r'W3Schools is optimized.*?',
                r'Spaces.*?Tutorials.*?References.*?',
            ]

            for pattern in noise_patterns:
                text = re.sub(pattern, ' ', text, flags=re.IGNORECASE | re.DOTALL)

            return text.strip()
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            return str(text) if text else ""

    def extract_with_rebel(self, text):
        """Extract triples using REBEL model with better error handling"""
        if not self.rebel_model or not self.rebel_tokenizer:
            return []

        try:
            # Truncate text to model limits
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]

            # Tokenize with proper error handling
            try:
                inputs = self.rebel_tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
            except Exception as e:
                logger.error(f"Tokenization error: {e}")
                return []

            # Move inputs to same device as model
            import torch
            if torch.cuda.is_available() and next(self.rebel_model.parameters()).is_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Generate with better parameters
            with torch.no_grad():
                outputs = self.rebel_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_length=256,
                    num_beams=3,  # Reduced for faster inference
                    early_stopping=True,
                    do_sample=False,
                    temperature=1.0
                )

            decoded = self.rebel_tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Parse REBEL output
            triples = self.parse_rebel_output(decoded)

            # Add metadata
            for triple in triples:
                triple['extraction_method'] = 'rebel'
                triple['confidence'] = 0.8

            return triples

        except Exception as e:
            logger.error(f"REBEL extraction error: {e}")
            return []

    def parse_rebel_output(self, rebel_text):
        """Parse REBEL model output into structured triples"""
        triples = []

        try:
            # REBEL uses specific format: <triplet> relation <subj> subject <obj> object
            pattern = r'<triplet>\s*(.*?)\s*<subj>\s*(.*?)\s*<obj>\s*(.*?)(?=<triplet>|$)'
            matches = re.findall(pattern, rebel_text, re.DOTALL)

            for match in matches:
                if len(match) == 3:
                    relation = match[0].strip()
                    subject = match[1].strip()
                    object_entity = match[2].strip()

                    # Validate extracted components
                    if relation and subject and object_entity and len(subject) > 1 and len(object_entity) > 1:
                        triples.append({
                            "subject": self.normalize_entity(subject),
                            "relation": self.normalize_relation(relation),
                            "object": self.normalize_entity(object_entity),
                            "confidence": 0.85
                        })

        except Exception as e:
            logger.error(f"Error parsing REBEL output: {e}")

        return triples

    def extract_with_improved_patterns(self, text):
        """Extract triples using improved pattern matching"""
        triples = []
        text = self.preprocess_text(text)

        if not text:
            return triples

        for category, patterns in self.extraction_patterns.items():
            for pattern, extractor_func in patterns:
                try:
                    matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)

                    for match in matches:
                        try:
                            extracted_triples = extractor_func(match)

                            if extracted_triples:
                                for triple_data in extracted_triples:
                                    if triple_data and len(triple_data) == 3:
                                        subject, relation, obj = triple_data
                                        if subject and relation and obj:
                                            triples.append({
                                                "subject": self.normalize_entity(subject),
                                                "relation": self.normalize_relation(relation),
                                                "object": self.normalize_entity(obj),
                                                "confidence": self.get_pattern_confidence(category),
                                                "extraction_method": "improved_patterns",
                                                "pattern_category": category
                                            })
                        except Exception as e:
                            logger.warning(f"Error processing match in {category}: {e}")
                            continue

                except Exception as e:
                    logger.error(f"Pattern extraction error in {category}: {e}")
                    continue

        return triples

    def normalize_entity(self, entity):
        """Normalize entity names with better validation"""
        if not entity:
            return ""

        try:
            entity = str(entity).strip()

            if not entity:
                return ""

            # Handle common variations
            normalizations = {
                'string': 'String',
                'integer': 'Integer',
                'arraylist': 'ArrayList',
                'hashmap': 'HashMap',
                'list': 'List',
                'map': 'Map',
                'set': 'Set'
            }

            lower_entity = entity.lower()
            if lower_entity in normalizations:
                return normalizations[lower_entity]

            # Remove special characters but keep alphanumeric and common programming chars
            entity = re.sub(r'[^\w.]', '', entity)

            # Capitalize first letter if it's a valid identifier
            if entity and entity[0].isalpha():
                return entity[0].upper() + entity[1:] if len(entity) > 1 else entity.upper()

            return entity if entity else ""

        except Exception as e:
            logger.error(f"Error normalizing entity '{entity}': {e}")
            return str(entity) if entity else ""

    def normalize_relation(self, relation):
        """Normalize relation names with better validation"""
        if not relation:
            return "relatedTo"

        try:
            relation = str(relation).strip().lower()

            if not relation:
                return "relatedTo"

            # Handle common variations
            normalizations = {
                'is_type_of': 'isTypeOf',
                'has_method': 'hasMethod',
                'belongs_to': 'belongsTo',
                'is_a': 'isA',
                'has_a': 'hasA',
                'instance_of': 'instanceOf',
                'subclass_of': 'subclassOf'
            }

            if relation in normalizations:
                return normalizations[relation]

            # Convert snake_case to camelCase
            if '_' in relation:
                parts = relation.split('_')
                if len(parts) > 1:
                    return parts[0] + ''.join(word.capitalize() for word in parts[1:])

            # Remove special characters
            relation = re.sub(r'[^\w]', '', relation)

            return relation if relation else "relatedTo"

        except Exception as e:
            logger.error(f"Error normalizing relation '{relation}': {e}")
            return "relatedTo"

    def get_pattern_confidence(self, category):
        """Get confidence score based on pattern category"""
        confidence_map = {
            'class_definition': 0.9,
            'interface_definition': 0.9,
            'method_definition': 0.8,
            'collection_usage': 0.85,
            'data_types': 0.9,
            'exception_handling': 0.8,
            'annotations': 0.85,
            'imports': 0.75,
            'spring_concepts': 0.8,
            'web_concepts': 0.7,
            'design_patterns': 0.75,
            'method_call': 0.7,
            'instantiation': 0.75
        }
        return confidence_map.get(category, 0.6)

    def load_scraped_data(self):
        """Load scraped content with better error handling"""
        try:
            if not self.scraped_data_file.exists():
                # Try alternative paths
                alternative_paths = [
                    Path("w3schools_java_tutorials.json"),
                    Path("data/w3schools_java_tutorials.json"),
                    Path("scraped_content/w3schools_java_tutorials.json")
                ]

                for alt_path in alternative_paths:
                    if alt_path.exists():
                        self.scraped_data_file = alt_path
                        break
                else:
                    raise FileNotFoundError(f"Scraped data file not found: {self.scraped_data_file}")

            with open(self.scraped_data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, dict):
                raise ValueError("Expected dictionary format in scraped data")

            print(f"📁 Loaded scraped data from: {self.scraped_data_file}")
            print(f"📊 Found {len(data)} entries")
            return data

        except FileNotFoundError as e:
            print(f"❌ File not found: {e}")
            print("💡 Make sure you have run the scraper first to generate the data file")
            return None
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON format: {e}")
            return None
        except Exception as e:
            print(f"❌ Error loading scraped data: {e}")
            return None

    def extract_from_content(self, content):
        """Extract triples from content using multiple methods"""
        all_triples = []

        try:
            # Validate content structure
            if not isinstance(content, dict):
                logger.warning("Invalid content structure")
                return all_triples

            # Extract from text content
            if 'text_content' in content and isinstance(content['text_content'], list):
                for text in content['text_content']:
                    if text and isinstance(text, str) and len(text.strip()) > 20:
                        # Use REBEL if available
                        if self.use_rebel and self.rebel_model:
                            rebel_triples = self.extract_with_rebel(text)
                            for triple in rebel_triples:
                                triple['source_text'] = text[:100] + "..." if len(text) > 100 else text
                                triple['source_url'] = content.get('url', 'unknown')
                                triple['source_type'] = 'text'
                            all_triples.extend(rebel_triples)

                        # Use improved patterns
                        if self.use_improved_patterns:
                            pattern_triples = self.extract_with_improved_patterns(text)
                            for triple in pattern_triples:
                                triple['source_text'] = text[:100] + "..." if len(text) > 100 else text
                                triple['source_url'] = content.get('url', 'unknown')
                                triple['source_type'] = 'text'
                            all_triples.extend(pattern_triples)

            # Extract from code examples with enhanced patterns
            if 'code_examples' in content and isinstance(content['code_examples'], list):
                for code in content['code_examples']:
                    if code and isinstance(code, str) and len(code.strip()) > 30:
                        pattern_triples = self.extract_with_improved_patterns(code)
                        for triple in pattern_triples:
                            triple['source_text'] = f"Code: {code[:80]}..." if len(code) > 80 else f"Code: {code}"
                            triple['source_url'] = content.get('url', 'unknown')
                            triple['source_type'] = 'code'
                            triple['confidence'] = min(1.0, triple.get('confidence', 0.6) + 0.1)  # Boost confidence for code
                        all_triples.extend(pattern_triples)

        except Exception as e:
            logger.error(f"Error extracting from content: {e}")

        return all_triples

    def deduplicate_triples(self, triples):
        """Remove duplicate triples and merge similar ones"""
        if not triples:
            return []

        seen_triples = {}

        for triple in triples:
            try:
                # Create a normalized key for deduplication
                key = (
                    str(triple.get('subject', '')).lower().strip(),
                    str(triple.get('relation', '')).lower().strip(),
                    str(triple.get('object', '')).lower().strip()
                )

                # Skip invalid triples
                if not all(key) or any(len(k) < 2 for k in key):
                    continue

                if key not in seen_triples:
                    seen_triples[key] = triple
                else:
                    # Keep the one with higher confidence
                    current_conf = float(triple.get('confidence', 0))
                    existing_conf = float(seen_triples[key].get('confidence', 0))
                    if current_conf > existing_conf:
                        seen_triples[key] = triple

            except Exception as e:
                logger.warning(f"Error processing triple during deduplication: {e}")
                continue

        return list(seen_triples.values())

    def extract_all_triples(self):
        """Main extraction method with improved quality"""
        print("=" * 60)
        print("🔍 Step 3: Advanced Triple Extraction")
        print("=" * 60)

        scraped_data = self.load_scraped_data()
        if not scraped_data:
            return None

        # Load REBEL model if requested
        if self.use_rebel:
            rebel_loaded = self.load_rebel_model()
            if not rebel_loaded:
                self.use_rebel = False

        all_triples = []
        total_pages = len(scraped_data)
        processed_pages = 0
        error_count = 0

        print(f"📊 Processing {total_pages} pages with advanced extraction...")
        print(f"🤖 REBEL model: {'✅ Enabled' if self.rebel_model else '❌ Disabled'}")
        print(f"🎯 Improved patterns: {'✅ Enabled' if self.use_improved_patterns else '❌ Disabled'}")

        for i, (url, content) in enumerate(scraped_data.items(), 1):
            try:
                if isinstance(content, dict) and 'error' in content:
                    error_count += 1
                    continue

                title = content.get('title', 'Unknown')[:50] if isinstance(content, dict) else 'Unknown'
                print(f"🔄 Processing {i}/{total_pages}: {title}...")

                triples = self.extract_from_content(content)
                all_triples.extend(triples)
                processed_pages += 1

                if i % 10 == 0:
                    print(f"📈 Progress: {i}/{total_pages} pages processed, {len(all_triples)} triples extracted")

            except Exception as e:
                logger.error(f"Error processing page {i}: {e}")
                error_count += 1
                continue

        print(f"✅ Extraction complete! Processed {processed_pages}/{total_pages} pages")
        print(f"⚠️ Errors encountered: {error_count}")
        print(f"📊 Found {len(all_triples)} raw triples")

        if not all_triples:
            print("❌ No triples extracted!")
            return None

        # Deduplicate and improve quality
        print("🧹 Deduplicating and improving quality...")
        deduplicated_triples = self.deduplicate_triples(all_triples)

        print(f"📊 After deduplication: {len(deduplicated_triples)} unique triples")
        return deduplicated_triples

    def save_triples(self, triples, filename="extracted_triples_improved.json"):
        """Save improved triples with detailed metadata"""
        try:
            if not triples:
                print("❌ No triples to save!")
                return None

            self.output_dir.mkdir(parents=True, exist_ok=True)
            filepath = self.output_dir / filename

            # Safely categorize triples
            high_confidence = []
            medium_confidence = []
            low_confidence = []
            rebel_triples = []
            pattern_triples = []

            for t in triples:
                conf = float(t.get('confidence', 0))
                if conf > 0.8:
                    high_confidence.append(t)
                elif conf >= 0.6:
                    medium_confidence.append(t)
                else:
                    low_confidence.append(t)

                method = t.get('extraction_method', '')
                if method == 'rebel':
                    rebel_triples.append(t)
                elif method == 'improved_patterns':
                    pattern_triples.append(t)

            # Calculate statistics safely
            relation_stats = defaultdict(lambda: {'count': 0, 'avg_confidence': 0})
            for triple in triples:
                rel = triple.get('relation', 'unknown')
                relation_stats[rel]['count'] += 1
                relation_stats[rel]['avg_confidence'] += float(triple.get('confidence', 0))

            for rel in relation_stats:
                if relation_stats[rel]['count'] > 0:
                    relation_stats[rel]['avg_confidence'] /= relation_stats[rel]['count']

            # Get unique entities safely
            unique_entities = set()
            for t in triples:
                unique_entities.add(str(t.get('subject', '')))
                unique_entities.add(str(t.get('object', '')))
            unique_entities.discard('')  # Remove empty strings

            avg_confidence = sum(float(t.get('confidence', 0)) for t in triples) / len(triples) if triples else 0

            output_data = {
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "extraction_methods": {
                        "rebel_enabled": self.use_rebel and self.rebel_model is not None,
                        "improved_patterns_enabled": self.use_improved_patterns,
                        "rebel_triples": len(rebel_triples),
                        "pattern_triples": len(pattern_triples)
                    },
                    "quality_metrics": {
                        "total_triples": len(triples),
                        "high_confidence": len(high_confidence),
                        "medium_confidence": len(medium_confidence),
                        "low_confidence": len(low_confidence),
                        "avg_confidence": avg_confidence,
                        "unique_relations": len(relation_stats),
                        "unique_entities": len(unique_entities)
                    },
                    "relation_statistics": dict(relation_stats)
                },
                "triples": {
                    "high_confidence": high_confidence,
                    "medium_confidence": medium_confidence,
                    "low_confidence": low_confidence,
                    "by_method": {
                        "rebel": rebel_triples,
                        "patterns": pattern_triples
                    },
                    "all": triples
                }
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            print(f"💾 Improved triples saved to: {filepath}")

            # Print quality summary
            self.print_quality_summary(output_data)

            return filepath

        except Exception as e:
            print(f"❌ Error saving triples: {e}")
            logger.error(f"Save error details: {e}")
            return None

    def print_quality_summary(self, data):
        """Print detailed quality summary"""
        try:
            metadata = data['metadata']
            quality = metadata['quality_metrics']

            print(f"\n📊 EXTRACTION QUALITY SUMMARY")
            print("=" * 35)
            print(f"🔢 Total triples: {quality['total_triples']:,}")

            total = quality['total_triples']
            if total > 0:
                print(f"🎯 High confidence (>0.8): {quality['high_confidence']:,} ({quality['high_confidence']/total*100:.1f}%)")
                print(f"📊 Medium confidence (0.6-0.8): {quality['medium_confidence']:,} ({quality['medium_confidence']/total*100:.1f}%)")

            print(f"📈 Average confidence: {quality['avg_confidence']:.3f}")
            print(f"🔗 Unique relations: {quality['unique_relations']}")
            print(f"🏷️ Unique entities: {quality['unique_entities']}")

            methods = metadata['extraction_methods']
            if methods['rebel_enabled']:
                print(f"🤖 REBEL triples: {methods['rebel_triples']:,}")
            print(f"🎯 Pattern triples: {methods['pattern_triples']:,}")

            # Top relations
            rel_stats = metadata['relation_statistics']
            if rel_stats:
                top_relations = sorted(rel_stats.items(), key=lambda x: x[1]['count'], reverse=True)[:10]

                print(f"\n🔗 TOP RELATIONS:")
                for rel, stats in top_relations:
                    print(f"   • {rel}: {stats['count']:,} ({stats['avg_confidence']:.2f} avg confidence)")

        except Exception as e:
            logger.error(f"Error printing quality summary: {e}")
            print("⚠️ Could not generate quality summary")

    def extract_and_save(self):
        """Main method with comprehensive extraction and quality analysis"""
        try:
            triples = self.extract_all_triples()

            if not triples:
                print("❌ No triples extracted")
                return None

            filepath = self.save_triples(triples)

            if filepath:
                print(f"\n✅ IMPROVED EXTRACTION COMPLETED!")
                print("-" * 40)
                print(f"📁 Output file: {filepath}")
                print(f"🚀 Ready for Step 4: Dynamic Ontology Update")
                return triples

            return None

        except Exception as e:
            print(f"❌ Error in extraction process: {e}")
            logger.error(f"Extraction process error: {e}")
            return None


class KGQualityTuner:
    """Additional tools for fine-tuning KG quality"""

    def __init__(self, triples_file="data/extracted_triples/extracted_triples_improved.json"):
        self.triples_file = Path(triples_file)

    def filter_low_quality_triples(self, min_confidence=0.5, min_entity_length=2):
        """Filter out low-quality triples"""
        try:
            if not self.triples_file.exists():
                print(f"❌ Triples file not found: {self.triples_file}")
                return []

            with open(self.triples_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            all_triples = data.get('triples', {}).get('all', [])

            if not all_triples:
                print("❌ No triples found in file")
                return []

            # Filter criteria
            filtered_triples = []
            for triple in all_triples:
                try:
                    confidence = float(triple.get('confidence', 0))
                    subject = str(triple.get('subject', '')).strip()
                    obj = str(triple.get('object', '')).strip()

                    if (confidence >= min_confidence and
                        len(subject) >= min_entity_length and
                        len(obj) >= min_entity_length and
                        subject.lower() != obj.lower() and
                        subject and obj):  # Ensure not empty
                        filtered_triples.append(triple)
                except Exception as e:
                    logger.warning(f"Error filtering triple: {e}")
                    continue

            print(f"🧹 Filtered {len(all_triples)} → {len(filtered_triples)} triples")
            if all_triples:
                print(f"📈 Quality improvement: {len(filtered_triples)/len(all_triples)*100:.1f}% retained")

            return filtered_triples

        except Exception as e:
            print(f"❌ Error filtering triples: {e}")
            return []

    def merge_similar_entities(self, triples):
        """Merge similar entity names"""
        if not triples:
            return []

        entity_mappings = {
            'string': 'String',
            'arraylist': 'ArrayList',
            'hashmap': 'HashMap',
            'exception': 'Exception',
            'integer': 'Integer',
            'list': 'List',
            'map': 'Map',
            'set': 'Set',
            'collection': 'Collection'
        }

        try:
            for triple in triples:
                subject = str(triple.get('subject', '')).strip().lower()
                obj = str(triple.get('object', '')).strip().lower()

                if subject in entity_mappings:
                    triple['subject'] = entity_mappings[subject]
                if obj in entity_mappings:
                    triple['object'] = entity_mappings[obj]

        except Exception as e:
            logger.error(f"Error merging similar entities: {e}")

        return triples

    def validate_triples(self, triples):
        """Validate triple structure and content"""
        if not triples:
            return []

        valid_triples = []
        invalid_count = 0

        for triple in triples:
            try:
                # Check required fields
                if not all(key in triple for key in ['subject', 'relation', 'object']):
                    invalid_count += 1
                    continue

                # Check non-empty values
                subject = str(triple['subject']).strip()
                relation = str(triple['relation']).strip()
                obj = str(triple['object']).strip()

                if not all([subject, relation, obj]):
                    invalid_count += 1
                    continue

                # Check reasonable lengths
                if any(len(x) < 2 for x in [subject, relation, obj]):
                    invalid_count += 1
                    continue

                # Check for circular references
                if subject.lower() == obj.lower():
                    invalid_count += 1
                    continue

                valid_triples.append(triple)

            except Exception as e:
                logger.warning(f"Error validating triple: {e}")
                invalid_count += 1
                continue

        if invalid_count > 0:
            print(f"⚠️ Removed {invalid_count} invalid triples")

        return valid_triples


def main():
    """Run improved triple extraction with comprehensive error handling"""
    print("🎯 Advanced Knowledge Graph Triple Extractor")
    print("-" * 50)

    # Configuration options
    use_rebel = True  # Set to False if you don't have transformers installed
    use_improved_patterns = True

    print(f"🤖 REBEL extraction: {'✅ Enabled' if use_rebel else '❌ Disabled'}")
    print(f"🎯 Improved patterns: {'✅ Enabled' if use_improved_patterns else '❌ Disabled'}")

    try:
        extractor = ImprovedTripleExtractor(
            use_rebel=use_rebel,
            use_improved_patterns=use_improved_patterns
        )

        triples = extractor.extract_and_save()

        if triples:
            print(f"\n🔍 SAMPLE IMPROVED TRIPLES")
            print("-" * 30)

            # Show diverse examples
            high_conf_triples = [t for t in triples if float(t.get('confidence', 0)) > 0.8]

            sample_count = min(5, len(high_conf_triples))
            for i, triple in enumerate(high_conf_triples[:sample_count], 1):
                method = triple.get('extraction_method', 'unknown')
                conf = float(triple.get('confidence', 0))
                print(f"{i}. {triple['subject']} --[{triple['relation']}]--> {triple['object']}")
                print(f"   Method: {method}, Confidence: {conf:.2f}")

            print(f"\n🎯 QUALITY IMPROVEMENTS ACHIEVED:")
            print("• More specific relationship types")
            print("• Better entity normalization")
            print("• Higher confidence scores")
            print("• Reduced noise and duplicates")
            print("• Framework-specific concepts captured")
            print("• Comprehensive error handling")

            print(f"\n✅ Ready for Step 4: Update Ontology with new concepts!")

            # Optional quality tuning
            print(f"\n🔧 OPTIONAL: Quality Tuning Available")
            print("Run quality tuner to further refine results:")
            print("tuner = KGQualityTuner()")
            print("filtered = tuner.filter_low_quality_triples(min_confidence=0.7)")

            return triples
        else:
            print("❌ Improved extraction failed!")
            print("\n🔍 TROUBLESHOOTING TIPS:")
            print("1. Check if scraped data file exists")
            print("2. Verify JSON format is valid")
            print("3. Ensure sufficient disk space")
            print("4. Check file permissions")
            print("5. Review log messages above for specific errors")
            return None

    except Exception as e:
        print(f"❌ Critical error in main execution: {e}")
        logger.error(f"Main execution error: {e}")

        print(f"\n🔍 COMMON SOLUTIONS:")
        print("1. Install required packages: pip install torch transformers sentencepiece")
        print("2. Check input file path and format")
        print("3. Ensure output directory is writable")
        print("4. Try with use_rebel=False if REBEL model fails")

        return None


def run_quality_analysis(triples_file="data/extracted_triples/extracted_triples_improved.json"):
    """Standalone function to analyze and improve triple quality"""
    print("🔍 Running Quality Analysis...")

    try:
        tuner = KGQualityTuner(triples_file)

        # Filter low quality
        filtered = tuner.filter_low_quality_triples(min_confidence=0.6)

        if filtered:
            # Merge similar entities
            merged = tuner.merge_similar_entities(filtered)

            # Validate final results
            validated = tuner.validate_triples(merged)

            print(f"✅ Quality analysis complete!")
            print(f"📊 Final count: {len(validated)} high-quality triples")

            return validated
        else:
            print("❌ No triples passed quality filter")
            return []

    except Exception as e:
        print(f"❌ Error in quality analysis: {e}")
        return []


if __name__ == "__main__":
    # Run main extraction
    result = main()

    # Optional: Run quality analysis if extraction succeeded
    if result:
        print(f"\n" + "="*50)
        print("🎯 Optional: Run Quality Analysis? (y/n)")
        try:
            # Note: In a real environment, you'd want user input here
            # For now, we'll skip the interactive part
            print("💡 To run quality analysis manually, call:")
            print("run_quality_analysis()")
        except:
            pass