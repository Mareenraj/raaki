import json
import os
import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import logging
from typing import List, Dict, Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OntologyAwareTripleExtractor:
    def __init__(
            self,
            technologies: List[str] = [
                "html", "css", "js", "python", "java", "sql",
                "bootstrap", "jquery", "json", "ajax", "xml", "api",
                "php", "csharp", "nodejs", "react", "typescript"
            ],
            input_dir="data/scraped_content",
            output_dir="data/extracted_triples",
            ontology_file="data/ontology/generalized_ontology.json",
            use_rebel=True,
            use_ontology_guided_extraction=True
    ):
        self.technologies = technologies
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.ontology_file = Path(ontology_file)
        self.use_rebel = use_rebel
        self.use_ontology_guided_extraction = use_ontology_guided_extraction

        # Load ontology data
        self.ontology = self.load_ontology()
        self.entities = set(self.ontology.get("entities", []))
        self.relations = set(self.ontology.get("relations", []))
        self.tech_mappings = self.ontology.get("technology_mappings", {})
        self.entity_categories = self.ontology.get("entity_categories", {})

        # REBEL model components
        self.rebel_model = None
        self.rebel_tokenizer = None

        # Create ontology-guided patterns
        self.ontology_patterns = self.create_ontology_guided_patterns()

    def load_ontology(self) -> Dict:
        """Load the generalized ontology created by GeneralizedOntologyCreator"""
        try:
            print(f"🔄 Loading ontology from: {self.ontology_file}")

            if not self.ontology_file.exists():
                print(f"⚠️ Ontology file not found, creating new one...")
                from create_ontology import GeneralizedOntologyCreator
                creator = GeneralizedOntologyCreator()
                ontology_data = creator.create_ontology()
                return ontology_data if ontology_data else {}

            with open(self.ontology_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract ontology from the full structure
            if "ontology" in data:
                ontology = data["ontology"]
                tech_mappings = data.get("technology_mappings", {})
                ontology["technology_mappings"] = tech_mappings
                ontology["entity_categories"] = ontology.get("entity_categories", {})
            else:
                ontology = data

            print(
                f"✅ Loaded ontology with {len(ontology.get('entities', []))} entities and {len(ontology.get('relations', []))} relations")
            return ontology

        except Exception as e:
            print(f"❌ Error loading ontology: {e}")
            # Fallback to basic ontology
            return {
                "entities": ["Component", "Function", "Class", "Property", "Method"],
                "relations": ["isA", "hasA", "uses", "contains", "implements"],
                "technology_mappings": {},
                "entity_categories": {}
            }

    def create_ontology_guided_patterns(self) -> Dict:
        """Create extraction patterns based on ontology entities and relations"""
        patterns = {}

        # Create patterns for each entity category
        for category, category_entities in self.entity_categories.items():
            patterns[category] = []

            for entity in category_entities:
                # Create flexible patterns for each entity
                entity_patterns = self.generate_entity_patterns(entity, category)
                patterns[category].extend(entity_patterns)

        # Add technology-specific patterns based on mappings
        for tech, mappings in self.tech_mappings.items():
            tech_patterns = []
            for general_entity, specific_terms in mappings.items():
                for specific_term in specific_terms:
                    pattern = self.create_tech_specific_pattern(specific_term, general_entity, tech)
                    if pattern:
                        tech_patterns.append(pattern)
            patterns[f"{tech}_specific"] = tech_patterns

        return patterns

    def generate_entity_patterns(self, entity: str, category: str) -> List[Tuple]:
        """Generate extraction patterns for a specific entity"""
        patterns = []
        entity_lower = entity.lower()

        # Basic entity mention patterns
        patterns.append((
            rf'\b{re.escape(entity_lower)}\b',
            lambda m, e=entity, c=category: self.extract_entity_mention(m, e, c)
        ))

        # Relationship patterns using ontology relations
        for relation in self.relations:
            if relation in ["isA", "isTypeOf", "instanceOf"]:
                # Pattern: "X is a/an Entity" or "X is Entity"
                patterns.append((
                    rf'(\w+)\s+(?:is\s+(?:a|an)\s+)?{re.escape(entity_lower)}',
                    lambda m, e=entity, r=relation: [(m.group(1), r, e)]
                ))

            elif relation in ["hasA", "contains", "includes"]:
                # Pattern: "Entity has/contains X"
                patterns.append((
                    rf'{re.escape(entity_lower)}\s+(?:has|contains|includes)\s+(\w+)',
                    lambda m, e=entity, r=relation: [(e, r, m.group(1))]
                ))

            elif relation in ["uses", "utilizes", "employs"]:
                # Pattern: "X uses Entity" or "Entity uses X"
                patterns.append((
                    rf'(\w+)\s+(?:uses|utilizes|employs)\s+{re.escape(entity_lower)}',
                    lambda m, e=entity, r=relation: [(m.group(1), r, e)]
                ))

        return patterns

    def create_tech_specific_pattern(self, specific_term: str, general_entity: str, tech: str) -> Optional[Tuple]:
        """Create technology-specific extraction patterns"""
        try:
            specific_lower = specific_term.lower()

            # Handle different term formats
            if " " in specific_term:
                # Multi-word terms like "CSS Property"
                pattern = rf'\b{re.escape(specific_lower)}\b'
            else:
                # Single word terms
                pattern = rf'\b{re.escape(specific_lower)}\b'

            def extractor(match):
                return [(specific_term, "isA", general_entity), (specific_term, "belongsTo", tech.upper())]

            return (pattern, extractor)

        except Exception as e:
            logger.warning(f"Error creating pattern for {specific_term}: {e}")
            return None

    def extract_entity_mention(self, match, entity: str, category: str) -> List[Tuple]:
        """Extract triples from entity mentions"""
        return [(match.group(0), "isA", entity), (entity, "belongsTo", category)]

    def extract_with_ontology_mappings(self, content: Dict, technology: str) -> List[Dict]:
        """Extract triples using pre-computed ontology mappings from web scraper"""
        triples = []

        # Use ontology mappings created by the web scraper
        ontology_mappings = content.get("ontology_mappings", {})

        for entity, text_snippets in ontology_mappings.items():
            if entity in self.entities:
                # Create triples from ontology mappings
                for snippet in text_snippets:
                    # Basic entity assertion
                    triples.append({
                        "subject": entity,
                        "relation": "appearsIn",
                        "object": technology.upper(),
                        "confidence": 0.9,
                        "extraction_method": "ontology_mapping",
                        "source_text": snippet,
                        "technology": technology
                    })

                    # Add category relationships
                    for category, entities_in_category in self.entity_categories.items():
                        if entity in entities_in_category:
                            triples.append({
                                "subject": entity,
                                "relation": "belongsTo",
                                "object": category.replace("_", " ").title(),
                                "confidence": 0.95,
                                "extraction_method": "ontology_mapping",
                                "source_text": snippet,
                                "technology": technology
                            })
                            break

                    # Add technology-specific mappings
                    if technology in self.tech_mappings:
                        tech_mapping = self.tech_mappings[technology]
                        if entity in tech_mapping:
                            for specific_term in tech_mapping[entity]:
                                if specific_term.lower() in snippet.lower():
                                    triples.append({
                                        "subject": specific_term,
                                        "relation": "isA",
                                        "object": entity,
                                        "confidence": 0.85,
                                        "extraction_method": "tech_mapping",
                                        "source_text": snippet,
                                        "technology": technology
                                    })

        return triples

    def extract_with_ontology_guided_patterns(self, text: str, technology: str) -> List[Dict]:
        """Extract triples using ontology-guided patterns"""
        triples = []

        if not text or not self.use_ontology_guided_extraction:
            return triples

        # Apply general ontology patterns
        for category, patterns in self.ontology_patterns.items():
            if category.endswith("_specific") and not category.startswith(technology):
                continue  # Skip other technology-specific patterns

            for pattern, extractor_func in patterns:
                try:
                    matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                    for match in matches:
                        try:
                            extracted_triples = extractor_func(match)
                            if extracted_triples:
                                for triple_data in extracted_triples:
                                    if len(triple_data) == 3:
                                        subject, relation, obj = triple_data
                                        if subject and relation and obj:
                                            triples.append({
                                                "subject": self.normalize_entity(subject),
                                                "relation": self.normalize_relation(relation),
                                                "object": self.normalize_entity(obj),
                                                "confidence": self.get_ontology_pattern_confidence(category),
                                                "extraction_method": "ontology_guided_patterns",
                                                "pattern_category": category,
                                                "source_text": text[max(0, match.start() - 20):match.end() + 20],
                                                "technology": technology
                                            })
                        except Exception as e:
                            logger.warning(f"Error processing match in {category}: {e}")
                            continue
                except Exception as e:
                    logger.error(f"Pattern extraction error in {category}: {e}")
                    continue

        return triples

    def get_ontology_pattern_confidence(self, category: str) -> float:
        """Get confidence score for ontology-guided patterns"""
        confidence_map = {
            'programming_constructs': 0.9,
            'data_entities': 0.88,
            'behavioral_entities': 0.85,
            'structural_entities': 0.87,
            'oop_entities': 0.9,
            'control_entities': 0.85,
            'communication_entities': 0.8,
            'storage_entities': 0.85,
            'resource_entities': 0.82,
            'processing_entities': 0.88,
            'security_entities': 0.9,
            'lifecycle_entities': 0.85,
            'quality_entities': 0.83,
            'ui_entities': 0.87,
            'network_entities': 0.85,
            'development_entities': 0.8,
            'metadata_entities': 0.82
        }

        # Technology-specific patterns get higher confidence
        if "_specific" in category:
            return 0.92

        return confidence_map.get(category, 0.75)

    def load_rebel_model(self):
        """Load REBEL model for advanced extraction"""
        if not self.use_rebel:
            return False

        try:
            print("🤖 Loading REBEL model...")
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch

            model_name = "Babelscape/rebel-large"
            self.rebel_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.rebel_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )

            if torch.cuda.is_available():
                self.rebel_model = self.rebel_model.cuda()

            print("✅ REBEL model loaded successfully!")
            return True

        except Exception as e:
            print(f"⚠️ Could not load REBEL model: {e}")
            return False

    def extract_with_rebel(self, text: str, technology: str) -> List[Dict]:
        """Extract triples using REBEL model with ontology validation"""
        if not self.rebel_model or not self.rebel_tokenizer:
            return []

        try:
            import torch

            # Truncate text if too long
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]

            inputs = self.rebel_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.rebel_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_length=256,
                    num_beams=3,
                    early_stopping=True
                )

            decoded = self.rebel_tokenizer.decode(outputs[0], skip_special_tokens=True)
            triples = self.parse_rebel_output(decoded, technology)

            # Filter triples using ontology knowledge
            validated_triples = self.validate_triples_with_ontology(triples)

            return validated_triples

        except Exception as e:
            logger.error(f"REBEL extraction error: {e}")
            return []

    def parse_rebel_output(self, rebel_text: str, technology: str) -> List[Dict]:
        """Parse REBEL output and enhance with ontology information"""
        triples = []

        try:
            pattern = r'<triplet>\s*(.*?)\s*<subj>\s*(.*?)\s*<obj>\s*(.*?)(?=<triplet>|$)'
            matches = re.findall(pattern, rebel_text, re.DOTALL)

            for match in matches:
                if len(match) == 3:
                    relation, subject, object_entity = [m.strip() for m in match]

                    if relation and subject and object_entity:
                        triples.append({
                            "subject": self.normalize_entity(subject),
                            "relation": self.normalize_relation(relation),
                            "object": self.normalize_entity(object_entity),
                            "confidence": 0.8,
                            "extraction_method": "rebel",
                            "technology": technology
                        })

        except Exception as e:
            logger.error(f"Error parsing REBEL output: {e}")

        return triples

    def validate_triples_with_ontology(self, triples: List[Dict]) -> List[Dict]:
        """Validate and enhance triples using ontology knowledge"""
        validated_triples = []

        for triple in triples:
            subject = triple.get("subject", "")
            relation = triple.get("relation", "")
            obj = triple.get("object", "")

            # Check if entities are in our ontology or can be mapped
            subject_valid = self.is_entity_valid(subject)
            object_valid = self.is_entity_valid(obj)
            relation_valid = relation in self.relations or self.find_similar_relation(relation)

            # Enhance confidence based on ontology validation
            base_confidence = float(triple.get("confidence", 0.5))

            if subject_valid and object_valid and relation_valid:
                triple["confidence"] = min(0.95, base_confidence + 0.2)
                triple["ontology_validated"] = True
            elif (subject_valid or object_valid) and relation_valid:
                triple["confidence"] = min(0.85, base_confidence + 0.1)
                triple["ontology_validated"] = "partial"
            else:
                triple["confidence"] = max(0.3, base_confidence - 0.1)
                triple["ontology_validated"] = False

            # Map relations to ontology relations if possible
            if not relation_valid:
                similar_relation = self.find_similar_relation(relation)
                if similar_relation:
                    triple["relation"] = similar_relation
                    triple["relation_mapped"] = True

            validated_triples.append(triple)

        return validated_triples

    def is_entity_valid(self, entity: str) -> bool:
        """Check if entity is valid according to ontology"""
        if not entity:
            return False

        # Direct match in ontology entities
        if entity in self.entities:
            return True

        # Check in technology mappings
        for tech_mappings in self.tech_mappings.values():
            for mapped_terms in tech_mappings.values():
                if entity in mapped_terms:
                    return True

        # Fuzzy matching (simple case-insensitive)
        entity_lower = entity.lower()
        for ont_entity in self.entities:
            if entity_lower == ont_entity.lower():
                return True

        return False

    def find_similar_relation(self, relation: str) -> Optional[str]:
        """Find similar relation in ontology"""
        if not relation:
            return None

        relation_lower = relation.lower()

        # Direct match
        for ont_relation in self.relations:
            if relation_lower == ont_relation.lower():
                return ont_relation

        # Semantic similarity (simple mapping)
        relation_mappings = {
            "type": "isA",
            "kind": "isA",
            "instance": "instanceOf",
            "part": "partOf",
            "member": "memberOf",
            "contain": "contains",
            "include": "includes",
            "use": "uses",
            "employ": "employs",
            "extend": "extends",
            "inherit": "inherits"
        }

        for key, mapped_relation in relation_mappings.items():
            if key in relation_lower and mapped_relation in self.relations:
                return mapped_relation

        return None

    def normalize_entity(self, entity: str) -> str:
        """Normalize entity names using ontology knowledge"""
        if not entity:
            return ""

        entity = str(entity).strip()

        # First, check if it matches an ontology entity exactly
        for ont_entity in self.entities:
            if entity.lower() == ont_entity.lower():
                return ont_entity

        # Check technology mappings
        for tech_mappings in self.tech_mappings.values():
            for general_entity, specific_terms in tech_mappings.items():
                for specific_term in specific_terms:
                    if entity.lower() == specific_term.lower():
                        return specific_term

        # Basic normalization
        entity = re.sub(r'[^\w.]', '', entity)
        if entity and entity[0].isalpha():
            return entity[0].upper() + entity[1:] if len(entity) > 1 else entity.upper()

        return entity

    def normalize_relation(self, relation: str) -> str:
        """Normalize relation names using ontology knowledge"""
        if not relation:
            return "relatedTo"

        # First check if it's already in ontology
        for ont_relation in self.relations:
            if relation.lower() == ont_relation.lower():
                return ont_relation

        # Try to find similar relation
        similar = self.find_similar_relation(relation)
        if similar:
            return similar

        # Basic normalization
        relation = str(relation).strip().lower()
        relation = re.sub(r'[^\w]', '', relation)

        return relation if relation else "relatedTo"

    def extract_from_content(self, content: Dict, technology: str) -> List[Dict]:
        """Main extraction method combining all approaches"""
        all_triples = []

        try:
            # Method 1: Use pre-computed ontology mappings from web scraper
            if self.use_ontology_guided_extraction:
                ontology_triples = self.extract_with_ontology_mappings(content, technology)
                all_triples.extend(ontology_triples)

            # Method 2: Extract from text content using ontology-guided patterns
            if 'text_content' in content and isinstance(content['text_content'], list):
                for text in content['text_content']:
                    if text and isinstance(text, str) and len(text.strip()) > 20:
                        # Ontology-guided pattern extraction
                        if self.use_ontology_guided_extraction:
                            pattern_triples = self.extract_with_ontology_guided_patterns(text, technology)
                            for triple in pattern_triples:
                                triple['source_url'] = content.get('url', 'unknown')
                                triple['source_type'] = 'text'
                            all_triples.extend(pattern_triples)

                        # REBEL extraction with ontology validation
                        if self.use_rebel and self.rebel_model:
                            rebel_triples = self.extract_with_rebel(text, technology)
                            for triple in rebel_triples:
                                triple['source_url'] = content.get('url', 'unknown')
                                triple['source_type'] = 'text'
                                triple['source_text'] = text[:100] + "..." if len(text) > 100 else text
                            all_triples.extend(rebel_triples)

            # Method 3: Extract from code examples with higher confidence
            if 'code_examples' in content and isinstance(content['code_examples'], list):
                for code in content['code_examples']:
                    if code and isinstance(code, str) and len(code.strip()) > 30:
                        code_triples = self.extract_with_ontology_guided_patterns(code, technology)
                        for triple in code_triples:
                            triple['source_url'] = content.get('url', 'unknown')
                            triple['source_type'] = 'code'
                            triple['source_text'] = f"Code: {code[:80]}..." if len(code) > 80 else f"Code: {code}"
                            # Boost confidence for code examples
                            triple['confidence'] = min(0.98, float(triple.get('confidence', 0.6)) + 0.15)
                        all_triples.extend(code_triples)

        except Exception as e:
            logger.error(f"Error extracting from content for {technology}: {e}")

        return all_triples

    def load_scraped_data(self, technology: str) -> Optional[Dict]:
        """Load scraped data for a technology"""
        try:
            data_file = self.input_dir / f"w3schools_{technology}_tutorials.json"

            if not data_file.exists():
                # Try alternative naming
                alt_files = [
                    self.input_dir / f"w3schools_{technology}.json",
                    self.input_dir / f"{technology}.json"
                ]

                for alt_file in alt_files:
                    if alt_file.exists():
                        data_file = alt_file
                        break
                else:
                    print(f"❌ No data file found for {technology}")
                    return None

            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            print(f"✅ Loaded data for {technology} from {data_file}")
            return data.get('content', {})

        except Exception as e:
            print(f"❌ Error loading data for {technology}: {e}")
            return None

    def deduplicate_triples(self, triples: List[Dict]) -> List[Dict]:
        """Enhanced deduplication considering ontology relationships"""
        if not triples:
            return []

        seen_triples = {}
        entity_aliases = defaultdict(set)

        # Build entity alias map using technology mappings
        for tech_mappings in self.tech_mappings.values():
            for general_entity, specific_terms in tech_mappings.items():
                for specific_term in specific_terms:
                    entity_aliases[general_entity].add(specific_term)
                    entity_aliases[specific_term].add(general_entity)

        for triple in triples:
            try:
                subject = str(triple.get('subject', '')).strip()
                relation = str(triple.get('relation', '')).strip()
                obj = str(triple.get('object', '')).strip()
                technology = triple.get('technology', '')

                if not all([subject, relation, obj]) or any(len(x) < 2 for x in [subject, relation, obj]):
                    continue

                # Create normalized key for deduplication
                key = (
                    subject.lower(),
                    relation.lower(),
                    obj.lower(),
                    technology
                )

                if key not in seen_triples:
                    seen_triples[key] = triple
                else:
                    # Keep the triple with higher confidence
                    current_conf = float(triple.get('confidence', 0))
                    existing_conf = float(seen_triples[key].get('confidence', 0))

                    if current_conf > existing_conf:
                        seen_triples[key] = triple
                    elif current_conf == existing_conf:
                        # Prefer ontology-validated triples
                        if triple.get('ontology_validated') and not seen_triples[key].get('ontology_validated'):
                            seen_triples[key] = triple

            except Exception as e:
                logger.warning(f"Error processing triple during deduplication: {e}")
                continue

        return list(seen_triples.values())

    def extract_and_save(self) -> Optional[Dict]:
        """Main extraction method with ontology integration"""
        print("=" * 70)
        print("🧠 Ontology-Aware Triple Extraction")
        print("=" * 70)

        print(f"📚 Ontology loaded: {len(self.entities)} entities, {len(self.relations)} relations")
        print(f"🔗 Technology mappings: {len(self.tech_mappings)} technologies")
        print(f"🎯 Ontology-guided extraction: {'✅ Enabled' if self.use_ontology_guided_extraction else '❌ Disabled'}")

        # Load REBEL if requested
        if self.use_rebel:
            rebel_loaded = self.load_rebel_model()
            if not rebel_loaded:
                self.use_rebel = False

        print(f"🤖 REBEL model: {'✅ Enabled' if self.rebel_model else '❌ Disabled'}")

        all_triples_by_tech = {}
        total_pages = 0
        processed_pages = 0

        for tech in self.technologies:
            print(f"\n📊 Processing {tech.upper()}...")
            scraped_data = self.load_scraped_data(tech)

            if not scraped_data:
                continue

            tech_triples = []
            tech_pages = len(scraped_data)
            total_pages += tech_pages

            print(f"📄 Processing {tech_pages} pages for {tech.upper()}...")

            for i, (url, content) in enumerate(scraped_data.items(), 1):
                try:
                    if isinstance(content, dict) and 'error' in content:
                        continue

                    title = content.get('title', 'Unknown')[:50]
                    print(f"🔄 Processing {i}/{tech_pages}: {title}...")

                    # Extract triples using all methods
                    triples = self.extract_from_content(content, tech)
                    tech_triples.extend(triples)
                    processed_pages += 1

                    if i % 10 == 0:
                        print(f"📈 Progress: {i}/{tech_pages} pages, {len(tech_triples)} triples extracted")

                except Exception as e:
                    logger.error(f"Error processing page {i} for {tech}: {e}")
                    continue

            if tech_triples:
                print(f"🧹 Deduplicating triples for {tech.upper()}...")
                deduplicated_triples = self.deduplicate_triples(tech_triples)
                all_triples_by_tech[tech] = deduplicated_triples

                # Show ontology integration stats
                ontology_validated = len([t for t in deduplicated_triples if t.get('ontology_validated') == True])
                ontology_guided = len(
                    [t for t in deduplicated_triples if t.get('extraction_method') == 'ontology_guided_patterns'])
                ontology_mapped = len(
                    [t for t in deduplicated_triples if t.get('extraction_method') == 'ontology_mapping'])

                print(f"📊 {tech.upper()}: {len(deduplicated_triples)} unique triples")
                print(f"   🎯 Ontology validated: {ontology_validated}")
                print(f"   🔍 Ontology guided: {ontology_guided}")
                print(f"   🗺️ Ontology mapped: {ontology_mapped}")

        # Save results
        if all_triples_by_tech:
            filepaths = self.save_triples(all_triples_by_tech)

            if filepaths:
                print(f"\n✅ ONTOLOGY-AWARE EXTRACTION COMPLETED!")
                print("-" * 50)

                total_triples = sum(len(triples) for triples in all_triples_by_tech.values())
                ontology_enhanced = sum(
                    len([t for t in triples if t.get('ontology_validated') or
                         t.get('extraction_method') in ['ontology_mapping', 'ontology_guided_patterns']])
                    for triples in all_triples_by_tech.values()
                )

                print(f"📊 Total triples extracted: {total_triples:,}")
                print(
                    f"🧠 Ontology-enhanced triples: {ontology_enhanced:,} ({ontology_enhanced / total_triples * 100:.1f}%)")
                print(f"📁 Output files: {len(filepaths)}")

                return all_triples_by_tech

        return None

    def save_triples(self, triples_by_tech: Dict, base_filename: str = "ontology_enhanced_triples_") -> List[str]:
        """Save triples with ontology enhancement metadata"""
        filepaths = []

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)

            for tech, triples in triples_by_tech.items():
                if not triples:
                    continue

                filepath = self.output_dir / f"{base_filename}{tech}.json"

                # Categorize triples by confidence and method
                high_confidence = [t for t in triples if float(t.get('confidence', 0)) > 0.8]
                medium_confidence = [t for t in triples if 0.6 <= float(t.get('confidence', 0)) <= 0.8]
                low_confidence = [t for t in triples if float(t.get('confidence', 0)) < 0.6]

                # Categorize by extraction method
                ontology_mapped = [t for t in triples if t.get('extraction_method') == 'ontology_mapping']
                ontology_guided = [t for t in triples if t.get('extraction_method') == 'ontology_guided_patterns']
                rebel_triples = [t for t in triples if t.get('extraction_method') == 'rebel']

                # Categorize by ontology validation
                ontology_validated = [t for t in triples if t.get('ontology_validated') == True]
                partially_validated = [t for t in triples if t.get('ontology_validated') == 'partial']
                not_validated = [t for t in triples if t.get('ontology_validated') == False]

                # Calculate statistics
                relation_stats = defaultdict(lambda: {'count': 0, 'avg_confidence': 0})
                entity_stats = defaultdict(lambda: {'as_subject': 0, 'as_object': 0})

                for triple in triples:
                    rel = triple.get('relation', 'unknown')
                    relation_stats[rel]['count'] += 1
                    relation_stats[rel]['avg_confidence'] += float(triple.get('confidence', 0))

                    subj = triple.get('subject', '')
                    obj = triple.get('object', '')
                    if subj:
                        entity_stats[subj]['as_subject'] += 1
                    if obj:
                        entity_stats[obj]['as_object'] += 1

                for rel in relation_stats:
                    if relation_stats[rel]['count'] > 0:
                        relation_stats[rel]['avg_confidence'] /= relation_stats[rel]['count']

                # Ontology coverage analysis
                entities_in_ontology = set()
                relations_in_ontology = set()

                for triple in triples:
                    subj = triple.get('subject', '')
                    rel = triple.get('relation', '')
                    obj = triple.get('object', '')

                    if subj in self.entities:
                        entities_in_ontology.add(subj)
                    if obj in self.entities:
                        entities_in_ontology.add(obj)
                    if rel in self.relations:
                        relations_in_ontology.add(rel)

                avg_confidence = sum(float(t.get('confidence', 0)) for t in triples) / len(triples) if triples else 0

                metadata = {
                    "created_at": datetime.now().isoformat(),
                    "technology": tech,
                    "ontology_integration": {
                        "ontology_file": str(self.ontology_file),
                        "total_ontology_entities": len(self.entities),
                        "total_ontology_relations": len(self.relations),
                        "entities_found_in_triples": len(entities_in_ontology),
                        "relations_found_in_triples": len(relations_in_ontology),
                        "ontology_entity_coverage": len(entities_in_ontology) / len(
                            self.entities) if self.entities else 0,
                        "ontology_relation_coverage": len(relations_in_ontology) / len(
                            self.relations) if self.relations else 0
                    },
                    "extraction_methods": {
                        "ontology_mapping_enabled": self.use_ontology_guided_extraction,
                        "ontology_guided_patterns_enabled": self.use_ontology_guided_extraction,
                        "rebel_enabled": self.use_rebel and self.rebel_model is not None,
                        "ontology_mapped_triples": len(ontology_mapped),
                        "ontology_guided_triples": len(ontology_guided),
                        "rebel_triples": len(rebel_triples)
                    },
                    "quality_metrics": {
                        "total_triples": len(triples),
                        "high_confidence": len(high_confidence),
                        "medium_confidence": len(medium_confidence),
                        "low_confidence": len(low_confidence),
                        "avg_confidence": avg_confidence,
                        "ontology_validated": len(ontology_validated),
                        "partially_validated": len(partially_validated),
                        "not_validated": len(not_validated),
                        "unique_relations": len(relation_stats),
                        "unique_entities": len(entity_stats)
                    },
                    "ontology_validation_stats": {
                        "fully_validated": len(ontology_validated),
                        "partially_validated": len(partially_validated),
                        "not_validated": len(not_validated),
                        "validation_rate": len(ontology_validated) / len(triples) if triples else 0
                    }
                }

                output_data = {
                    "metadata": metadata,
                    "ontology_info": {
                        "entities_used": sorted(list(entities_in_ontology)),
                        "relations_used": sorted(list(relations_in_ontology)),
                        "technology_mappings": self.tech_mappings.get(tech, {})
                    },
                    "triples": {
                        "by_confidence": {
                            "high_confidence": high_confidence,
                            "medium_confidence": medium_confidence,
                            "low_confidence": low_confidence
                        },
                        "by_method": {
                            "ontology_mapping": ontology_mapped,
                            "ontology_guided_patterns": ontology_guided,
                            "rebel": rebel_triples
                        },
                        "by_validation": {
                            "ontology_validated": ontology_validated,
                            "partially_validated": partially_validated,
                            "not_validated": not_validated
                        },
                        "all": triples
                    },
                    "statistics": {
                        "relation_statistics": dict(relation_stats),
                        "entity_statistics": dict(entity_stats)
                    }
                }

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)

                file_size = os.path.getsize(filepath) / (1024 * 1024)
                print(f"💾 Enhanced triples saved to: {filepath}")
                print(f"📏 File size: {file_size:.2f} MB")
                filepaths.append(str(filepath))

                self.print_ontology_integration_summary(output_data)

            return filepaths

        except Exception as e:
            print(f"❌ Error saving triples: {e}")
            return []

    def print_ontology_integration_summary(self, data: Dict):
        """Print detailed summary of ontology integration"""
        try:
            metadata = data['metadata']
            tech = metadata['technology'].upper()
            quality = metadata['quality_metrics']
            ontology = metadata['ontology_integration']
            validation = metadata['ontology_validation_stats']
            methods = metadata['extraction_methods']

            print(f"\n🧠 ONTOLOGY INTEGRATION SUMMARY FOR {tech}")
            print("=" * 45)

            # Basic stats
            print(f"🔢 Total triples: {quality['total_triples']:,}")
            print(f"📈 Average confidence: {quality['avg_confidence']:.3f}")

            # Ontology coverage
            print(f"\n📚 Ontology Coverage:")
            print(f"   • Entities in ontology: {ontology['total_ontology_entities']}")
            print(f"   • Relations in ontology: {ontology['total_ontology_relations']}")
            print(f"   • Entities found in triples: {ontology['entities_found_in_triples']}")
            print(f"   • Relations found in triples: {ontology['relations_found_in_triples']}")
            print(f"   • Entity coverage: {ontology['ontology_entity_coverage'] * 100:.1f}%")
            print(f"   • Relation coverage: {ontology['ontology_relation_coverage'] * 100:.1f}%")

            # Validation stats
            print(f"\n✅ Ontology Validation:")
            print(
                f"   • Fully validated: {validation['fully_validated']:,} ({validation['validation_rate'] * 100:.1f}%)")
            print(f"   • Partially validated: {validation['partially_validated']:,}")
            print(f"   • Not validated: {validation['not_validated']:,}")

            # Extraction methods
            print(f"\n🔍 Extraction Methods:")
            if methods['ontology_mapping_enabled']:
                print(f"   • Ontology mapping: {methods['ontology_mapped_triples']:,}")
            if methods['ontology_guided_patterns_enabled']:
                print(f"   • Ontology-guided patterns: {methods['ontology_guided_triples']:,}")
            if methods['rebel_enabled']:
                print(f"   • REBEL (ontology-enhanced): {methods['rebel_triples']:,}")

        except Exception as e:
            logger.error(f"Error printing ontology integration summary: {e}")
            print("⚠️ Could not generate integration summary")


def main():
    """Run ontology-aware triple extraction"""
    print("🧠 Ontology-Aware Knowledge Graph Triple Extractor")
    print("=" * 55)

    print("🎯 Features:")
    print("   • Leverages generalized ontology for guided extraction")
    print("   • Uses pre-computed ontology mappings from web scraper")
    print("   • Validates triples against ontology knowledge")
    print("   • Enhanced confidence scoring based on ontology validation")
    print("   • Technology-specific pattern matching")

    try:
        extractor = OntologyAwareTripleExtractor(
            use_rebel=True,
            use_ontology_guided_extraction=True
        )

        triples_by_tech = extractor.extract_and_save()

        if triples_by_tech:
            print(f"\n🔍 ONTOLOGY-ENHANCED SAMPLE TRIPLES")
            print("-" * 40)

            for tech, triples in list(triples_by_tech.items())[:3]:  # Show first 3 technologies
                if triples:
                    validated_triples = [t for t in triples if t.get('ontology_validated') == True]
                    sample_count = min(3, len(validated_triples))

                    print(f"\n{tech.upper()} - Ontology Validated Triples:")
                    for i, triple in enumerate(validated_triples[:sample_count], 1):
                        method = triple.get('extraction_method', 'unknown')
                        conf = float(triple.get('confidence', 0))
                        validation = triple.get('ontology_validated', False)

                        print(f"{i}. {triple['subject']} --[{triple['relation']}]--> {triple['object']}")
                        print(f"   Method: {method}, Confidence: {conf:.2f}, Validated: {validation}")

            print(f"\n🎯 ONTOLOGY INTEGRATION BENEFITS:")
            print("• Higher quality triples through ontology validation")
            print("• Consistent entity and relation naming")
            print("• Technology-agnostic knowledge representation")
            print("• Enhanced confidence scoring")
            print("• Systematic coverage of domain concepts")

            return triples_by_tech

        else:
            print("❌ Extraction failed!")
            return None

    except Exception as e:
        print(f"❌ Critical error: {e}")
        return None


if __name__ == "__main__":
    main()