import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Attempt to import google-generativeai, with fallback if not installed
try:
    import google.generativeai as genai
except ImportError:
    logger.error("Package 'google-generativeai' not found. Install it with 'pip install google-generativeai'.")
    genai = None

# Import tenacity for retry logic
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
except ImportError:
    logger.error("Package 'tenacity' not found. Install it with 'pip install tenacity'.")
    def retry(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    retry_if_exception_type = lambda x: lambda e: isinstance(e, x)

class DataValidator:
    def __init__(self,
                 ontology_file="data/ontology/updated_ontology.json",
                 triples_file="data/extracted_triples/extracted_triples_improved.json",
                 output_dir="data/validated_data",
                 use_gemini=False,
                 api_key=None,
                 api_endpoint="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
                 model_id="gemini-1.5-flash"):
        self.ontology_file = Path(ontology_file)
        self.triples_file = Path(triples_file)
        self.output_dir = Path(output_dir)
        self.use_gemini = use_gemini
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.model_id = model_id
        self.ontology = None
        self.validation_stats = defaultdict(int)

    def load_ontology(self):
        """Load the ontology for validation"""
        try:
            if not self.ontology_file.exists():
                initial_ontology = Path("data/ontology/initial_ontology.json")
                if initial_ontology.exists():
                    self.ontology_file = initial_ontology
                else:
                    raise FileNotFoundError("No ontology file found")
            with open(self.ontology_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.ontology = data['ontology'] if 'ontology' in data else data
            logger.info(f"Loaded ontology from: {self.ontology_file}")
            logger.info(f"Classes: {len(self.ontology.get('classes', []))}, Relations: {len(self.ontology.get('relations', []))}")
            return True
        except Exception as e:
            logger.error(f"Error loading ontology: {e}")
            return False

    def load_triples(self):
        """Load extracted triples for validation"""
        try:
            if not self.triples_file.exists():
                raise FileNotFoundError(f"Triples file not found: {self.triples_file}")
            with open(self.triples_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            all_triples = data['triples'].get('all', []) if 'triples' in data else data if isinstance(data, list) else []
            logger.info(f"Loaded {len(all_triples)} triples for validation")
            return all_triples
        except Exception as e:
            logger.error(f"Error loading triples: {e}")
            return []

    def normalize_entity(self, entity):
        """Normalize entity names for consistent validation"""
        if not entity:
            return ""
        entity = entity.strip()
        mappings = {
            'string': 'String',
            'integer': 'Integer', 'int': 'Integer',
            'double': 'Double', 'float': 'Float',
            'boolean': 'Boolean', 'char': 'Character',
            'arraylist': 'ArrayList', 'hashmap': 'HashMap',
            'list': 'List', 'set': 'Set', 'map': 'Map'
        }
        lower_entity = entity.lower()
        return mappings.get(lower_entity, entity.capitalize())

    def is_reasonable_entity(self, entity):
        """Check if an entity is a reasonable programming concept"""
        if not entity or len(entity) < 2:
            return False
        reasonable_patterns = [
            'method', 'function', 'variable', 'parameter', 'argument',
            'class', 'interface', 'enum', 'annotation', 'package',
            'exception', 'error', 'handler', 'listener', 'event',
            'thread', 'process', 'service', 'controller', 'model',
            'view', 'component', 'module', 'library', 'framework',
            'api', 'rest', 'http', 'json', 'xml', 'sql', 'database'
        ]
        entity_lower = entity.lower()
        if any(pattern in entity_lower for pattern in reasonable_patterns):
            return True
        if entity.replace('_', '').replace('-', '').isalnum() and entity[0].isalpha():
            return True
        return False

    def initialize_api_client(self):
        """Initialize Gemini API client"""
        if not self.use_gemini or not self.api_key:
            logger.warning("API key or Gemini usage not enabled. Falling back to rule-based validation.")
            self.use_gemini = False
            return False
        if genai is None:
            logger.error("Package 'google-generativeai' not available. Falling back to rule-based validation.")
            self.use_gemini = False
            return False
        try:
            genai.configure(api_key=self.api_key)
            logger.info(f"Initialized Gemini API client for model: {self.model_id}")
            return True
        except Exception as e:
            logger.error(f"Error initializing Gemini API client: {e}")
            self.use_gemini = False
            return False

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=30),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying API call (attempt {retry_state.attempt_number}) due to error: {retry_state.outcome.exception()}"
        )
    )
    def call_gemini_api(self, prompt):
        """Make a call to the Gemini API with retry logic"""
        if genai is None:
            raise ImportError("google-generativeai not available")
        model = genai.GenerativeModel(self.model_id)
        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 5,
                "temperature": 0.1
            }
        )
        return response.text.strip().lower()

    def is_technical_triple_with_gemini(self, triple, max_length=256):
        """
        Use Gemini API to determine if a triple is technical (programming-related).
        Returns: True if technical, False otherwise.
        """
        if not self.use_gemini:
            # Fallback to rule-based filtering
            sub = triple.get('subject', '').lower()
            rel = triple.get('relation', '').lower()
            obj = triple.get('object', '').lower()
            combined = f"{sub} {rel} {obj}"
            return any(p in combined for p in ['method', 'class', 'function', 'type', 'extends', 'implements', 'call'])

        prompt = f"""
        You are an expert in software engineering and knowledge graphs.
        Determine if the following triple describes a technical relationship in programming or software design.
        Answer only with 'Yes' or 'No'.
        Triple: {triple['subject']} --{triple['relation']}--> {triple['object']}
        Is this a technical programming-related fact?
        """

        try:
            response_text = self.call_gemini_api(prompt)
            return 'yes' in response_text
        except Exception as e:
            logger.error(f"Gemini API inference error: {e}")
            # Be conservative: allow if uncertain
            return True

    def validate_triple(self, triple):
        """Validate a single triple against the ontology and technical relevance"""
        validation_result = {
            'original_triple': triple,
            'is_valid': True,
            'is_technical': True,
            'issues': [],
            'normalized_triple': {}
        }
        try:
            subject = self.normalize_entity(triple.get('subject', ''))
            relation = triple.get('relation', '').strip()
            obj = self.normalize_entity(triple.get('object', ''))
            confidence = triple.get('confidence', 0)
            # Normalize
            normalized = {
                'subject': subject,
                'relation': relation,
                'object': obj,
                'confidence': confidence,
                'source_url': triple.get('source_url', ''),
                'source_type': triple.get('source_type', 'text')
            }
            validation_result['normalized_triple'] = normalized
            # 1. Technical Check via API or fallback
            if self.use_gemini:
                is_technical = self.is_technical_triple_with_gemini(triple)
                if not is_technical:
                    validation_result['issues'].append("Non-technical triple (filtered out)")
                    validation_result['is_valid'] = False
                    validation_result['is_technical'] = False
                    self.validation_stats['non_technical_triples'] += 1
                    return validation_result
            # 2. Ontology: Relation check
            valid_relations = self.ontology.get('relations', [])
            if relation not in valid_relations:
                validation_result['issues'].append(f"Unknown relation: {relation}")
                self.validation_stats['unknown_relations'] += 1
            # 3. Entity sanity check
            valid_classes = self.ontology.get('classes', [])
            if subject not in valid_classes and not self.is_reasonable_entity(subject):
                validation_result['issues'].append(f"Questionable subject: {subject}")
                self.validation_stats['questionable_subjects'] += 1
            if obj not in valid_classes and not self.is_reasonable_entity(obj):
                validation_result['issues'].append(f"Questionable object: {obj}")
                self.validation_stats['questionable_objects'] += 1
            # 4. Confidence
            if confidence < 0.3:
                validation_result['issues'].append(f"Low confidence: {confidence:.2f}")
                self.validation_stats['low_confidence'] += 1
            # Final validity
            critical_issues = ['Unknown relation', 'Questionable subject', 'Questionable object', 'Non-technical']
            has_critical = any(issue.split(':')[0] in critical_issues for issue in validation_result['issues'])
            if has_critical:
                validation_result['is_valid'] = False
                self.validation_stats['invalid_triples'] += 1
            else:
                self.validation_stats['valid_triples'] += 1
        except Exception as e:
            logger.error(f"Validation error: {e}")
            validation_result['is_valid'] = False
            validation_result['issues'].append(f"Validation error: {str(e)}")
            self.validation_stats['validation_errors'] += 1
        return validation_result

    def remove_duplicates(self, validated_triples):
        seen_triples = {}
        for result in validated_triples:
            if not result['is_valid']:
                continue
            triple = result['normalized_triple']
            key = (triple['subject'], triple['relation'], triple['object'])
            if key not in seen_triples or triple['confidence'] > seen_triples[key]['normalized_triple']['confidence']:
                seen_triples[key] = result
        unique_count = len(seen_triples)
        self.validation_stats['unique_valid_triples'] = unique_count
        return list(seen_triples.values())

    def create_nodes_and_edges(self, validated_triples):
        nodes = {}
        edges = []
        for result in validated_triples:
            if not result['is_valid']:
                continue
            triple = result['normalized_triple']
            subject = triple['subject']
            obj = triple['object']
            relation = triple['relation']
            for ent in [subject, obj]:
                if ent not in nodes:
                    nodes[ent] = {'name': ent, 'type': 'Entity', 'source_count': 0}
                nodes[ent]['source_count'] += 1
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
        logger.info("=" * 50)
        logger.info("Step 5: Validating Nodes and Relations (with Gemini API)")
        logger.info("=" * 50)
        if not self.load_ontology():
            return None
        triples = self.load_triples()
        if not triples:
            logger.error("No triples to validate")
            return None
        # Initialize API client if enabled
        if self.use_gemini:
            if not self.initialize_api_client():
                logger.warning("Gemini API client initialization failed. Falling back to rule-based filtering.")
        logger.info(f"Validating {len(triples)} triples...")
        validated_results = []
        batch_size = 2  # Reduced to avoid 429 errors
        for i, triple in enumerate(triples):
            result = self.validate_triple(triple)
            validated_results.append(result)
            if (i + 1) % batch_size == 0 and self.use_gemini:
                # Respect Gemini API rate limits (10 RPM)
                time.sleep(12)  # Increased delay to 12 seconds
            if (i + 1) % 100 == 0:
                logger.info(f"Progress: {i + 1}/{len(triples)} triples validated")
        # Deduplicate
        logger.info("Removing duplicates...")
        unique_valid_results = self.remove_duplicates(validated_results)
        # Create nodes and edges
        logger.info("Creating nodes and edges...")
        nodes, edges = self.create_nodes_and_edges(unique_valid_results)
        # Save
        output_data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'ontology_source': str(self.ontology_file),
                'triples_source': str(self.triples_file),
                'validation_stats': dict(self.validation_stats),
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'gemini_used': self.use_gemini
            },
            'nodes': nodes,
            'edges': edges,
            'validation_details': {
                'all_results': validated_results,
                'valid_unique_results': unique_valid_results
            }
        }
        return self.save_validated_data(output_data)

    def save_validated_data(self, data):
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            main_file = self.output_dir / "validated_data.json"
            with open(main_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            nodes_file = self.output_dir / "nodes.json"
            with open(nodes_file, 'w', encoding='utf-8') as f:
                json.dump(data['nodes'], f, indent=2, ensure_ascii=False)
            edges_file = self.output_dir / "edges.json"
            with open(edges_file, 'w', encoding='utf-8') as f:
                json.dump(data['edges'], f, indent=2, ensure_ascii=False)
            logger.info(f"Validated data saved to: {self.output_dir}")
            return data
        except Exception as e:
            logger.error(f"Error saving validated data: {e}")
            return None

def main():
    logger.info("Knowledge Graph Data Validator (with Gemini API)")
    logger.info("-" * 45)
    # ✅ Set to True to enable Gemini API-based validation
    # TODO: Add your Google API key here
    api_key = "AIzaSyDC_e3t4kqiY6kJa5453RsSlvJ2vhV8cjY"
    validator = DataValidator(
        use_gemini=True,
        api_key=api_key,
        model_id="gemini-1.5-flash",
        api_endpoint="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    )
    validated_data = validator.validate_all_data()
    if validated_data:
        stats = validated_data['metadata']['validation_stats']
        logger.info("\nVALIDATION SUMMARY")
        logger.info("-" * 25)
        total = stats.get('valid_triples', 0) + stats.get('invalid_triples', 0)
        logger.info(f"Total triples processed: {total}")
        logger.info(f"Valid triples: {stats.get('valid_triples', 0)}")
        logger.info(f"Invalid triples: {stats.get('invalid_triples', 0)}")
        logger.info(f"Non-technical filtered: {stats.get('non_technical_triples', 0)}")
        logger.info(f"Unique valid triples: {stats.get('unique_valid_triples', 0)}")
        logger.info(f"Total nodes: {validated_data['metadata']['total_nodes']}")
        logger.info(f"Total edges: {validated_data['metadata']['total_edges']}")
        logger.info(f"Unknown relations: {stats.get('unknown_relations', 0)}")
        logger.info(f"Low confidence: {stats.get('low_confidence', 0)}")
        success_rate = (stats.get('valid_triples', 0) / max(1, total)) * 100
        logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info("\nReady for Step 6: Neo4j Loading!")
        return validated_data
    else:
        logger.error("Data validation failed!")
        return None

if __name__ == "__main__":
    main()
