# # """
# # Technical Triple Filter and Validator
# # Removes non-technical/common sentence triples and keeps only technology-relevant ones
# # """
# #
# # import json
# # import re
# # from datetime import datetime
# # from pathlib import Path
# # from collections import defaultdict, Counter
# # from typing import Dict, List, Tuple, Optional, Set
# # import logging
# #
# # # Set up logging
# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)
# #
# #
# # class TechnicalTripleFilter:
# #     def __init__(self,
# #                  input_file="data/extracted_triples/ontology_enhanced_triples_python.json",
# #                  output_dir="data/filtered_triples",
# #                  ontology_file="data/ontology/generalized_ontology.json",
# #                  strictness_level="medium"):  # low, medium, high
# #
# #         self.input_file = Path(input_file)
# #         self.output_dir = Path(output_dir)
# #         self.ontology_file = Path(ontology_file)
# #         self.strictness_level = strictness_level
# #
# #         # Load ontology
# #         self.ontology = self.load_ontology()
# #         self.tech_entities = set(self.ontology.get('entities', []))
# #         self.tech_relations = set(self.ontology.get('relations', []))
# #         self.tech_mappings = self.ontology.get('technology_mappings', {})
# #
# #         # Initialize filtering components
# #         self.technical_keywords = self._initialize_technical_keywords()
# #         self.common_sentence_patterns = self._initialize_common_patterns()
# #         self.non_technical_indicators = self._initialize_non_technical_indicators()
# #
# #         # Statistics tracking
# #         self.filtering_stats = defaultdict(int)
# #         self.removed_examples = defaultdict(list)
# #
# #     def load_ontology(self) -> Dict:
# #         """Load ontology for technical validation"""
# #         try:
# #             with open(self.ontology_file, 'r', encoding='utf-8') as f:
# #                 data = json.load(f)
# #
# #             if 'ontology' in data:
# #                 return data['ontology']
# #             return data
# #
# #         except Exception as e:
# #             logger.error(f"Error loading ontology: {e}")
# #             return {'entities': [], 'relations': [], 'technology_mappings': {}}
# #
# #     def _initialize_technical_keywords(self) -> Set[str]:
# #         """Initialize comprehensive technical keyword set"""
# #         technical_keywords = set()
# #
# #         # Programming languages
# #         technical_keywords.update([
# #             'python', 'java', 'javascript', 'typescript', 'c', 'cpp', 'csharp',
# #             'go', 'rust', 'kotlin', 'swift', 'php', 'ruby', 'scala', 'r',
# #             'sql', 'html', 'css', 'jsx', 'tsx'
# #         ])
# #
# #         # Programming concepts
# #         technical_keywords.update([
# #             'class', 'function', 'method', 'variable', 'parameter', 'argument',
# #             'interface', 'enum', 'annotation', 'package', 'module', 'library',
# #             'framework', 'api', 'endpoint', 'service', 'controller', 'model',
# #             'component', 'element', 'object', 'instance', 'constructor',
# #             'inheritance', 'polymorphism', 'encapsulation', 'abstraction',
# #             'algorithm', 'data_structure', 'array', 'list', 'dictionary',
# #             'hashmap', 'arraylist', 'collection', 'iterator', 'stream'
# #         ])
# #
# #         # Data types
# #         technical_keywords.update([
# #             'string', 'integer', 'float', 'double', 'boolean', 'char',
# #             'byte', 'long', 'short', 'decimal', 'number', 'bigint'
# #         ])
# #
# #         # Web technologies
# #         technical_keywords.update([
# #             'http', 'https', 'rest', 'graphql', 'json', 'xml', 'yaml',
# #             'ajax', 'fetch', 'request', 'response', 'client', 'server',
# #             'browser', 'dom', 'node', 'element', 'attribute', 'event',
# #             'callback', 'promise', 'async', 'await', 'middleware'
# #         ])
# #
# #         # Database technologies
# #         technical_keywords.update([
# #             'database', 'sql', 'nosql', 'query', 'table', 'column', 'row',
# #             'index', 'schema', 'connection', 'transaction', 'cursor',
# #             'mongodb', 'mysql', 'postgresql', 'redis', 'elasticsearch',
# #             'orm', 'crud', 'join', 'foreign_key', 'primary_key'
# #         ])
# #
# #         # Frameworks and libraries
# #         technical_keywords.update([
# #             'react', 'vue', 'angular', 'nodejs', 'express', 'django',
# #             'flask', 'fastapi', 'spring', 'hibernate', 'junit', 'pytest',
# #             'bootstrap', 'jquery', 'lodash', 'axios', 'webpack', 'babel'
# #         ])
# #
# #         # Development tools
# #         technical_keywords.update([
# #             'git', 'docker', 'kubernetes', 'jenkins', 'maven', 'gradle',
# #             'npm', 'pip', 'composer', 'yarn', 'webpack', 'babel',
# #             'eslint', 'prettier', 'jest', 'mocha', 'cypress'
# #         ])
# #
# #         # Cloud and infrastructure
# #         technical_keywords.update([
# #             'aws', 'azure', 'gcp', 'cloud', 'microservice', 'container',
# #             'deployment', 'ci_cd', 'devops', 'infrastructure', 'scaling',
# #             'load_balancer', 'cache', 'cdn', 'ssl', 'authentication'
# #         ])
# #
# #         # Add ontology entities
# #         if self.tech_entities:
# #             technical_keywords.update(entity.lower() for entity in self.tech_entities)
# #
# #         return technical_keywords
# #
# #     def _initialize_common_patterns(self) -> List[Tuple[str, str]]:
# #         """Initialize patterns for common non-technical sentences"""
# #         patterns = [
# #             # Generic descriptions
# #             (r'^(the|a|an)\s+\w+\s+(is|are|was|were)\s+(a|an|the)\s+\w+$', 'generic_description'),
# #             (r'^\w+\s+(can|will|should|must|may)\s+\w+', 'modal_verb_pattern'),
# #             (r'^\w+\s+(and|or|but)\s+\w+\s+(is|are)', 'conjunction_pattern'),
# #
# #             # Time and location
# #             (r'\b(today|yesterday|tomorrow|now|then|here|there)\b', 'temporal_spatial'),
# #             (r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
# #              'date_reference'),
# #             (r'\b\d{1,2}:\d{2}\b|\b\d{1,2}(am|pm)\b', 'time_reference'),
# #
# #             # Common verbs that indicate non-technical content
# #             (r'\b(said|says|told|asked|replied|answered|explained|mentioned)\b', 'communication_verbs'),
# #             (r'\b(went|came|arrived|left|departed|traveled|visited)\b', 'movement_verbs'),
# #             (r'\b(ate|drank|slept|walked|ran|drove|flew)\b', 'daily_activity_verbs'),
# #
# #             # Personal pronouns and social context
# #             (r'\b(i|you|he|she|we|they)\s+(am|is|are|was|were|will|would|should)\b', 'personal_pronouns'),
# #             (r'\b(my|your|his|her|our|their)\s+\w+', 'possessive_pronouns'),
# #             (r'\b(family|friend|colleague|neighbor|boss|employee)\b', 'social_relations'),
# #
# #             # Common adjectives for everyday descriptions
# #             (r'\b(beautiful|ugly|nice|good|bad|happy|sad|angry|excited|tired)\b', 'emotional_adjectives'),
# #             (r'\b(big|small|large|tiny|huge|enormous|little)\b', 'size_adjectives'),
# #             (r'\b(red|blue|green|yellow|black|white|brown|pink|purple|orange)\b', 'color_adjectives'),
# #
# #             # Common objects/concepts not related to programming
# #             (r'\b(car|house|building|tree|flower|animal|food|book|movie|music)\b', 'everyday_objects'),
# #             (r'\b(school|university|hospital|restaurant|store|market|park)\b', 'places'),
# #             (r'\b(breakfast|lunch|dinner|morning|afternoon|evening|night)\b', 'daily_routine'),
# #
# #             # Weak technical connections
# #             (r'^\w+\s+(has|contains|includes)\s+(some|many|several|few|a lot of)\s+\w+$', 'vague_quantifiers'),
# #             (r'^\w+\s+(is|are)\s+(very|quite|really|extremely|somewhat)\s+\w+$', 'intensity_adverbs'),
# #         ]
# #
# #         return patterns
# #
# #     def _initialize_non_technical_indicators(self) -> Dict[str, List[str]]:
# #         """Initialize indicators of non-technical content"""
# #         return {
# #             'weak_subjects': [
# #                 'thing', 'stuff', 'item', 'element', 'part', 'piece', 'section',
# #                 'area', 'place', 'location', 'position', 'point', 'spot',
# #                 'person', 'people', 'individual', 'someone', 'anyone', 'everyone',
# #                 'something', 'anything', 'everything', 'nothing', 'somewhere'
# #             ],
# #             'weak_objects': [
# #                 'thing', 'stuff', 'item', 'element', 'part', 'piece',
# #                 'way', 'manner', 'method', 'approach', 'style', 'form',
# #                 'time', 'moment', 'period', 'duration', 'while', 'when',
# #                 'reason', 'cause', 'purpose', 'goal', 'aim', 'objective'
# #             ],
# #             'weak_relations': [
# #                 'is', 'are', 'was', 'were', 'be', 'being', 'been',
# #                 'has', 'have', 'had', 'having', 'get', 'got', 'getting',
# #                 'do', 'does', 'did', 'doing', 'done', 'make', 'makes', 'made',
# #                 'go', 'goes', 'went', 'going', 'come', 'comes', 'came', 'coming'
# #             ],
# #             'non_technical_domains': [
# #                 'weather', 'sports', 'politics', 'entertainment', 'food', 'travel',
# #                 'health', 'medicine', 'education', 'history', 'geography',
# #                 'biology', 'chemistry', 'physics', 'mathematics', 'literature',
# #                 'art', 'music', 'fashion', 'lifestyle', 'relationships'
# #             ]
# #         }
# #
# #     def is_technical_entity(self, entity: str) -> bool:
# #         """Check if entity is technical/programming related"""
# #         if not entity or len(entity.strip()) < 2:
# #             return False
# #
# #         entity_lower = entity.lower().strip()
# #
# #         # Direct match with technical keywords
# #         if entity_lower in self.technical_keywords:
# #             return True
# #
# #         # Check if entity contains technical keywords
# #         for keyword in self.technical_keywords:
# #             if keyword in entity_lower:
# #                 return True
# #
# #         # Check against ontology entities
# #         if entity in self.tech_entities:
# #             return True
# #
# #         # Check technology mappings
# #         for tech_mappings in self.tech_mappings.values():
# #             for mapped_terms in tech_mappings.values():
# #                 if entity in mapped_terms or entity_lower in [t.lower() for t in mapped_terms]:
# #                     return True
# #
# #         # Programming naming conventions
# #         programming_patterns = [
# #             r'[A-Z][a-z]+[A-Z][a-z]+',  # CamelCase
# #             r'[a-z]+_[a-z]+',  # snake_case
# #             r'[A-Z]+_[A-Z]+',  # UPPER_CASE
# #             r'\w+Service$',  # Service suffix
# #             r'\w+Controller$',  # Controller suffix
# #             r'\w+Manager$',  # Manager suffix
# #             r'\w+Handler$',  # Handler suffix
# #             r'\w+Factory$',  # Factory suffix
# #             r'\w+Builder$',  # Builder suffix
# #             r'\w+Config$',  # Config suffix
# #             r'\w+Utils?$',  # Utils suffix
# #             r'get\w+$',  # Getter methods
# #             r'set\w+$',  # Setter methods
# #             r'is\w+$',  # Boolean methods
# #             r'has\w+$',  # Has methods
# #         ]
# #
# #         for pattern in programming_patterns:
# #             if re.match(pattern, entity):
# #                 return True
# #
# #         return False
# #
# #     def is_technical_relation(self, relation: str) -> bool:
# #         """Check if relation is technical/programming related"""
# #         if not relation:
# #             return False
# #
# #         relation_lower = relation.lower().strip()
# #
# #         # Direct match with ontology relations
# #         if relation in self.tech_relations:
# #             return True
# #
# #         # Technical relation patterns
# #         technical_relations = {
# #             'implements', 'extends', 'inherits', 'overrides', 'calls', 'invokes',
# #             'returns', 'throws', 'catches', 'handles', 'processes', 'executes',
# #             'compiles', 'builds', 'deploys', 'configures', 'initializes',
# #             'instantiates', 'creates', 'destroys', 'allocates', 'deallocates',
# #             'imports', 'exports', 'includes', 'requires', 'depends_on',
# #             'uses', 'utilizes', 'employs', 'applies', 'operates_on',
# #             'manages', 'controls', 'monitors', 'validates', 'verifies',
# #             'connects_to', 'communicates_with', 'sends_to', 'receives_from'
# #         }
# #
# #         if relation_lower in technical_relations:
# #             return True
# #
# #         # Check for partial matches
# #         for tech_rel in technical_relations:
# #             if tech_rel in relation_lower or relation_lower in tech_rel:
# #                 return True
# #
# #         return False
# #
# #     def calculate_technical_score(self, triple: Dict) -> float:
# #         """Calculate technical relevance score for a triple"""
# #         score = 0.0
# #         max_score = 10.0
# #
# #         subject = triple.get('subject', '')
# #         relation = triple.get('relation', '')
# #         obj = triple.get('object', '')
# #         source_text = triple.get('source_text', '')
# #         technology = triple.get('technology', '')
# #
# #         # Subject technical score (0-3 points)
# #         if self.is_technical_entity(subject):
# #             score += 3.0
# #         elif any(keyword in subject.lower() for keyword in self.technical_keywords):
# #             score += 1.5
# #
# #         # Object technical score (0-3 points)
# #         if self.is_technical_entity(obj):
# #             score += 3.0
# #         elif any(keyword in obj.lower() for keyword in self.technical_keywords):
# #             score += 1.5
# #
# #         # Relation technical score (0-2 points)
# #         if self.is_technical_relation(relation):
# #             score += 2.0
# #         elif relation.lower() in ['uses', 'contains', 'has', 'creates', 'manages']:
# #             score += 1.0
# #
# #         # Technology context score (0-1 point)
# #         if technology and technology.lower() in self.technical_keywords:
# #             score += 1.0
# #
# #         # Source text technical density (0-1 point)
# #         if source_text:
# #             technical_word_count = sum(1 for word in source_text.lower().split()
# #                                        if word in self.technical_keywords)
# #             total_words = len(source_text.split())
# #             if total_words > 0:
# #                 technical_density = technical_word_count / total_words
# #                 score += technical_density
# #
# #         return min(score / max_score, 1.0)
# #
# #     def has_common_sentence_pattern(self, triple: Dict) -> Tuple[bool, str]:
# #         """Check if triple matches common non-technical sentence patterns"""
# #         subject = triple.get('subject', '')
# #         relation = triple.get('relation', '')
# #         obj = triple.get('object', '')
# #         source_text = triple.get('source_text', '')
# #
# #         # Create triple text for pattern matching
# #         triple_text = f"{subject} {relation} {obj}"
# #         if source_text:
# #             triple_text += f" {source_text}"
# #
# #         triple_text_lower = triple_text.lower()
# #
# #         # Check against common patterns
# #         for pattern, pattern_type in self.common_sentence_patterns:
# #             if re.search(pattern, triple_text_lower):
# #                 return True, pattern_type
# #
# #         # Check for weak entities/relations
# #         indicators = self.non_technical_indicators
# #
# #         if subject.lower() in indicators['weak_subjects']:
# #             return True, 'weak_subject'
# #
# #         if obj.lower() in indicators['weak_objects']:
# #             return True, 'weak_object'
# #
# #         if relation.lower() in indicators['weak_relations']:
# #             return True, 'weak_relation'
# #
# #         # Check for non-technical domain words
# #         for domain_word in indicators['non_technical_domains']:
# #             if domain_word in triple_text_lower:
# #                 return True, f'non_technical_domain_{domain_word}'
# #
# #         return False, ''
# #
# #     def get_strictness_threshold(self) -> float:
# #         """Get technical score threshold based on strictness level"""
# #         thresholds = {
# #             'low': 0.3,  # Keep more triples, less strict
# #             'medium': 0.5,  # Balanced approach
# #             'high': 0.7  # Very strict, only clearly technical triples
# #         }
# #         return thresholds.get(self.strictness_level, 0.5)
# #
# #     def should_keep_triple(self, triple: Dict) -> Tuple[bool, Dict]:
# #         """Determine if triple should be kept based on technical relevance"""
# #
# #         # Calculate technical score
# #         technical_score = self.calculate_technical_score(triple)
# #
# #         # Check for common sentence patterns
# #         has_common_pattern, pattern_type = self.has_common_sentence_pattern(triple)
# #
# #         # Get strictness threshold
# #         threshold = self.get_strictness_threshold()
# #
# #         # Decision logic
# #         decision_info = {
# #             'technical_score': technical_score,
# #             'threshold': threshold,
# #             'has_common_pattern': has_common_pattern,
# #             'pattern_type': pattern_type,
# #             'reasons': []
# #         }
# #
# #         # Primary decision based on technical score
# #         keep = technical_score >= threshold
# #
# #         # Override decisions for clear cases
# #         if has_common_pattern:
# #             # Common patterns usually indicate non-technical content
# #             if pattern_type in ['personal_pronouns', 'emotional_adjectives', 'everyday_objects', 'daily_routine']:
# #                 keep = False
# #                 decision_info['reasons'].append(f'Rejected: Common pattern ({pattern_type})')
# #             elif technical_score > 0.8:  # High technical score overrides some patterns
# #                 keep = True
# #                 decision_info['reasons'].append(f'Kept: High technical score overrides pattern ({pattern_type})')
# #             else:
# #                 keep = False
# #                 decision_info['reasons'].append(f'Rejected: Pattern + low technical score ({pattern_type})')
# #
# #         # Additional quality checks
# #         subject = triple.get('subject', '').strip()
# #         obj = triple.get('object', '').strip()
# #         relation = triple.get('relation', '').strip()
# #
# #         # Reject very short or empty entities
# #         if len(subject) < 2 or len(obj) < 2 or len(relation) < 2:
# #             keep = False
# #             decision_info['reasons'].append('Rejected: Too short entities')
# #
# #         # Reject if both subject and object are non-technical
# #         if not self.is_technical_entity(subject) and not self.is_technical_entity(obj):
# #             if technical_score < 0.6:  # Unless overall score is decent
# #                 keep = False
# #                 decision_info['reasons'].append('Rejected: Both entities non-technical')
# #
# #         # Keep if explicitly technical regardless of patterns
# #         if any(tech_word in f"{subject} {obj} {relation}".lower()
# #                for tech_word in ['class', 'function', 'method', 'api', 'database', 'server', 'client']):
# #             keep = True
# #             decision_info['reasons'].append('Kept: Contains explicit technical terms')
# #
# #         if keep:
# #             decision_info['reasons'].append(f'Technical score {technical_score:.3f} >= threshold {threshold:.3f}')
# #         else:
# #             decision_info['reasons'].append(f'Technical score {technical_score:.3f} < threshold {threshold:.3f}')
# #
# #         return keep, decision_info
# #
# #     def filter_triples(self, triples: List[Dict]) -> Tuple[List[Dict], List[Dict], Dict]:
# #         """Filter triples into technical and non-technical categories"""
# #
# #         technical_triples = []
# #         removed_triples = []
# #
# #         print(f"🔍 Filtering {len(triples)} triples with strictness level: {self.strictness_level}")
# #
# #         for i, triple in enumerate(triples):
# #             try:
# #                 keep, decision_info = self.should_keep_triple(triple)
# #
# #                 if keep:
# #                     technical_triples.append(triple)
# #                     self.filtering_stats['kept_technical'] += 1
# #                 else:
# #                     # Add decision info to removed triple
# #                     triple_with_info = triple.copy()
# #                     triple_with_info['removal_reason'] = decision_info
# #                     removed_triples.append(triple_with_info)
# #
# #                     self.filtering_stats['removed_non_technical'] += 1
# #
# #                     # Track removal reasons
# #                     pattern_type = decision_info.get('pattern_type', 'low_technical_score')
# #                     self.filtering_stats[f'removed_{pattern_type}'] += 1
# #
# #                     # Store examples for analysis
# #                     if len(self.removed_examples[pattern_type]) < 5:
# #                         self.removed_examples[pattern_type].append({
# #                             'triple': f"{triple.get('subject', '')} --[{triple.get('relation', '')}]--> {triple.get('object', '')}",
# #                             'score': decision_info['technical_score'],
# #                             'reasons': decision_info['reasons']
# #                         })
# #
# #                 # Progress update
# #                 if (i + 1) % 100 == 0:
# #                     progress = (i + 1) / len(triples) * 100
# #                     print(f"📈 Progress: {i + 1}/{len(triples)} ({progress:.1f}%) - "
# #                           f"Kept: {len(technical_triples)}, Removed: {len(removed_triples)}")
# #
# #             except Exception as e:
# #                 logger.error(f"Error processing triple {i}: {e}")
# #                 continue
# #
# #         # Final statistics
# #         filtering_summary = {
# #             'total_input': len(triples),
# #             'technical_kept': len(technical_triples),
# #             'non_technical_removed': len(removed_triples),
# #             'retention_rate': len(technical_triples) / len(triples) if triples else 0,
# #             'strictness_level': self.strictness_level,
# #             'threshold_used': self.get_strictness_threshold(),
# #             'detailed_stats': dict(self.filtering_stats),
# #             'removal_examples': dict(self.removed_examples)
# #         }
# #
# #         return technical_triples, removed_triples, filtering_summary
# #
# #     def load_triples(self) -> List[Dict]:
# #         """Load triples from input file"""
# #         try:
# #             with open(self.input_file, 'r', encoding='utf-8') as f:
# #                 data = json.load(f)
# #
# #             # Handle different file formats
# #             triples = []
# #             if 'triples' in data:
# #                 if 'all' in data['triples']:
# #                     triples = data['triples']['all']
# #                 else:
# #                     # Combine all categories
# #                     for category_triples in data['triples'].values():
# #                         if isinstance(category_triples, list):
# #                             triples.extend(category_triples)
# #             elif isinstance(data, list):
# #                 triples = data
# #             else:
# #                 # Try to find triples in nested structure
# #                 for key, value in data.items():
# #                     if isinstance(value, list) and value and isinstance(value[0], dict):
# #                         if 'subject' in value[0]:
# #                             triples = value
# #                             break
# #
# #             print(f"✅ Loaded {len(triples)} triples from {self.input_file}")
# #             return triples
# #
# #         except Exception as e:
# #             print(f"❌ Error loading triples: {e}")
# #             return []
# #
# #     def save_filtered_results(self, technical_triples: List[Dict], removed_triples: List[Dict],
# #                               filtering_summary: Dict) -> Optional[Dict]:
# #         """Save filtered results to separate files"""
# #         try:
# #             self.output_dir.mkdir(parents=True, exist_ok=True)
# #
# #             # Prepare comprehensive output data
# #             output_data = {
# #                 'metadata': {
# #                     'created_at': datetime.now().isoformat(),
# #                     'input_file': str(self.input_file),
# #                     'filtering_method': 'technical_relevance_filter',
# #                     'strictness_level': self.strictness_level,
# #                     'ontology_source': str(self.ontology_file),
# #                     'filtering_summary': filtering_summary
# #                 },
# #                 'technical_triples': technical_triples,
# #                 'removed_triples': removed_triples,
# #                 'statistics': filtering_summary
# #             }
# #
# #             # Save main filtered results
# #             main_file = self.output_dir / f"filtered_technical_triples_{self.strictness_level}.json"
# #             with open(main_file, 'w', encoding='utf-8') as f:
# #                 json.dump(output_data, f, indent=2, ensure_ascii=False)
# #
# #             # Save only technical triples (clean format for downstream use)
# #             clean_file = self.output_dir / f"technical_triples_clean_{self.strictness_level}.json"
# #             clean_data = {
# #                 'metadata': {
# #                     'created_at': datetime.now().isoformat(),
# #                     'total_triples': len(technical_triples),
# #                     'filtering_method': 'technical_relevance_filter',
# #                     'strictness_level': self.strictness_level,
# #                     'retention_rate': filtering_summary['retention_rate']
# #                 },
# #                 'triples': technical_triples
# #             }
# #             with open(clean_file, 'w', encoding='utf-8') as f:
# #                 json.dump(clean_data, f, indent=2, ensure_ascii=False)
# #
# #             # Save removed triples for analysis
# #             removed_file = self.output_dir / f"removed_non_technical_triples_{self.strictness_level}.json"
# #             removed_data = {
# #                 'metadata': {
# #                     'created_at': datetime.now().isoformat(),
# #                     'total_removed': len(removed_triples),
# #                     'removal_categories': dict(Counter(
# #                         r['removal_reason']['pattern_type']
# #                         for r in removed_triples
# #                         if 'removal_reason' in r
# #                     ))
# #                 },
# #                 'removed_triples': removed_triples
# #             }
# #             with open(removed_file, 'w', encoding='utf-8') as f:
# #                 json.dump(removed_data, f, indent=2, ensure_ascii=False)
# #
# #             # Save filtering analysis
# #             analysis_file = self.output_dir / f"filtering_analysis_{self.strictness_level}.json"
# #             analysis_data = {
# #                 'filtering_summary': filtering_summary,
# #                 'technical_keywords_used': list(self.technical_keywords)[:100],  # Sample
# #                 'pattern_examples': dict(self.removed_examples),
# #                 'recommendations': self._generate_filtering_recommendations(filtering_summary)
# #             }
# #             with open(analysis_file, 'w', encoding='utf-8') as f:
# #                 json.dump(analysis_data, f, indent=2, ensure_ascii=False)
# #
# #             # Print results
# #             self.print_filtering_summary(filtering_summary, technical_triples, removed_triples)
# #
# #             print(f"\n💾 Filtered results saved to: {self.output_dir}")
# #             print(f"📁 Files created:")
# #             print(f"   • {main_file.name} (complete results)")
# #             print(f"   • {clean_file.name} (technical triples only)")
# #             print(f"   • {removed_file.name} (removed triples)")
# #             print(f"   • {analysis_file.name} (filtering analysis)")
# #
# #             return output_data
# #
# #         except Exception as e:
# #             print(f"❌ Error saving filtered results: {e}")
# #             return None
# #
# #     def _generate_filtering_recommendations(self, summary: Dict) -> Dict:
# #         """Generate recommendations based on filtering results"""
# #         recommendations = {}
# #
# #         retention_rate = summary['retention_rate']
# #
# #         if retention_rate < 0.3:
# #             recommendations['strictness'] = {
# #                 'current': self.strictness_level,
# #                 'suggestion': 'Consider using "low" strictness level',
# #                 'reason': 'Very low retention rate suggests filtering might be too aggressive'
# #             }
# #         elif retention_rate > 0.9:
# #             recommendations['strictness'] = {
# #                 'current': self.strictness_level,
# #                 'suggestion': 'Consider using "high" strictness level',
# #                 'reason': 'High retention rate suggests you could be more selective'
# #             }
# #
# #         # Analyze removal patterns
# #         detailed_stats = summary.get('detailed_stats', {})
# #         top_removal_reasons = sorted(
# #             [(k, v) for k, v in detailed_stats.items() if k.startswith('removed_')],
# #             key=lambda x: x[1], reverse=True
# #         )[:3]
# #
# #         if top_removal_reasons:
# #             recommendations['top_removal_patterns'] = [
# #                 f"{reason.replace('removed_', '').replace('_', ' ').title()}: {count} triples"
# #                 for reason, count in top_removal_reasons
# #             ]
# #
# #         return recommendations
# #
# #     def print_filtering_summary(self, summary: Dict, technical_triples: List[Dict],
# #                                 removed_triples: List[Dict]):
# #         """Print comprehensive filtering summary"""
# #         print(f"\n🔬 TECHNICAL TRIPLE FILTERING SUMMARY")
# #         print("=" * 50)
# #
# #         print(f"📊 Filtering Results:")
# #         print(f"   Total Input Triples: {summary['total_input']:,}")
# #         print(f"   Technical Triples Kept: {summary['technical_kept']:,}")
# #         print(f"   Non-Technical Removed: {summary['non_technical_removed']:,}")
# #         print(f"   Retention Rate: {summary['retention_rate']:.1%}")
# #         print(f"   Strictness Level: {summary['strictness_level']}")
# #         print(f"   Technical Threshold: {summary['threshold_used']:.2f}")
# #
# #         # Show top removal reasons
# #         detailed_stats = summary.get('detailed_stats', {})
# #         removal_reasons = [(k, v) for k, v in detailed_stats.items() if k.startswith('removed_')]
# #         if removal_reasons:
# #             print(f"\n🚫 Top Removal Reasons:")
# #             sorted_reasons = sorted(removal_reasons, key=lambda x: x[1], reverse=True)[:5]
# #             for reason, count in sorted_reasons:
# #                 clean_reason = reason.replace('removed_', '').replace('_', ' ').title()
# #                 percentage = (count / summary['non_technical_removed']) * 100 if summary[
# #                                                                                      'non_technical_removed'] > 0 else 0
# #                 print(f"   • {clean_reason}: {count:,} ({percentage:.1f}%)")
# #
# #         # Show examples of removed triples
# #         removal_examples = summary.get('removal_examples', {})
# #         if removal_examples:
# #             print(f"\n📝 Examples of Removed Triples:")
# #             for pattern_type, examples in list(removal_examples.items())[:3]:
# #                 if examples:
# #                     print(f"\n   {pattern_type.replace('_', ' ').title()}:")
# #                     for example in examples[:2]:
# #                         print(f"     • {example['triple']}")
# #                         print(f"       Score: {example['score']:.3f}, Reasons: {', '.join(example['reasons'][:2])}")
# #
# #         # Quality indicators
# #         if technical_triples:
# #             avg_technical_score = sum(
# #                 self.calculate_technical_score(triple) for triple in technical_triples[:100]
# #             ) / min(100, len(technical_triples))
# #
# #             print(f"\n📈 Quality Metrics:")
# #             print(f"   Average Technical Score (kept triples): {avg_technical_score:.3f}")
# #             print(f"   Technical Keywords Recognized: {len(self.technical_keywords):,}")
# #
# #             # Show sample of kept triples
# #             print(f"\n✅ Sample Technical Triples (Kept):")
# #             for i, triple in enumerate(technical_triples[:3], 1):
# #                 subj = triple.get('subject', '')
# #                 rel = triple.get('relation', '')
# #                 obj = triple.get('object', '')
# #                 score = self.calculate_technical_score(triple)
# #                 print(f"   {i}. {subj} --[{rel}]--> {obj}")
# #                 print(f"      Technical Score: {score:.3f}")
# #
# #
# # def filter_triples_by_technology(input_file: str, output_dir: str = "data/filtered_triples",
# #                                  strictness: str = "medium") -> Optional[Dict]:
# #     """
# #     Main function to filter technical triples
# #
# #     Args:
# #         input_file: Path to input triples JSON file
# #         output_dir: Directory to save filtered results
# #         strictness: Filtering strictness level ('low', 'medium', 'high')
# #
# #     Returns:
# #         Dictionary with filtering results or None if failed
# #     """
# #
# #     print("🔬 Technical Triple Filter & Validator")
# #     print("=" * 40)
# #     print(f"🎯 Purpose: Remove common/non-technical triples, keep only technology-relevant ones")
# #     print(f"📁 Input: {input_file}")
# #     print(f"📊 Strictness: {strictness}")
# #     print(f"💾 Output: {output_dir}")
# #
# #     try:
# #         # Initialize filter
# #         filter_tool = TechnicalTripleFilter(
# #             input_file=input_file,
# #             output_dir=output_dir,
# #             strictness_level=strictness
# #         )
# #
# #         # Load triples
# #         triples = filter_tool.load_triples()
# #         if not triples:
# #             print("❌ No triples to filter")
# #             return None
# #
# #         # Filter triples
# #         technical_triples, removed_triples, filtering_summary = filter_tool.filter_triples(triples)
# #
# #         # Save results
# #         results = filter_tool.save_filtered_results(technical_triples, removed_triples, filtering_summary)
# #
# #         if results:
# #             print(f"\n🎉 Technical filtering completed successfully!")
# #             print(f"✨ Clean technical triples ready for validation and knowledge graph creation!")
# #
# #             # Provide next steps
# #             clean_file = Path(output_dir) / f"technical_triples_clean_{strictness}.json"
# #             print(f"\n🔄 Next Steps:")
# #             print(f"1. Use the clean file for validation: {clean_file}")
# #             print(f"2. Run your validator on the filtered triples")
# #             print(f"3. Load validated triples into Neo4j")
# #
# #             return results
# #         else:
# #             print("❌ Failed to save filtering results")
# #             return None
# #
# #     except Exception as e:
# #         print(f"❌ Error in technical filtering: {e}")
# #         return None
# #
# #
# # def main():
# #     """Run technical triple filtering with different strictness levels"""
# #     print("🔬 Technical Triple Filter for Knowledge Graphs")
# #     print("=" * 50)
# #
# #     # Configuration options
# #     strictness_options = {
# #         "1": {
# #             "level": "low",
# #             "description": "Keep more triples, less strict filtering (threshold: 0.3)",
# #             "use_case": "When you want to preserve more data and filter manually later"
# #         },
# #         "2": {
# #             "level": "medium",
# #             "description": "Balanced filtering (threshold: 0.5) - Recommended",
# #             "use_case": "Good balance between quality and quantity"
# #         },
# #         "3": {
# #             "level": "high",
# #             "description": "Very strict, only clearly technical triples (threshold: 0.7)",
# #             "use_case": "When you want only high-confidence technical content"
# #         },
# #         "4": {
# #             "level": "all",
# #             "description": "Run all three levels for comparison",
# #             "use_case": "To analyze different filtering approaches"
# #         }
# #     }
# #
# #     print("📋 Strictness Level Options:")
# #     for key, option in strictness_options.items():
# #         print(f"   {key}. {option['level'].title()}: {option['description']}")
# #         print(f"      Use case: {option['use_case']}")
# #
# #     # Default to medium strictness
# #     selected_strictness = "medium"
# #     print(f"\n🎯 Using: {selected_strictness} strictness level")
# #
# #     # Example input files - adjust paths as needed
# #     input_files = [
# #         "data/extracted_triples/ontology_enhanced_triples_python.json",
# #         "data/extracted_triples/ontology_enhanced_triples_java.json",
# #         "data/extracted_triples/ontology_enhanced_triples_javascript.json"
# #     ]
# #
# #     # Check which files exist
# #     existing_files = [f for f in input_files if Path(f).exists()]
# #
# #     if not existing_files:
# #         print(f"\n⚠️ No input files found. Expected files:")
# #         for f in input_files:
# #             print(f"   • {f}")
# #         print(f"\nUsing default file for demonstration...")
# #         input_file = input_files[0]  # Use first file as default
# #     else:
# #         input_file = existing_files[0]  # Use first existing file
# #         print(f"\n✅ Found input file: {input_file}")
# #
# #     try:
# #         if selected_strictness == "all":
# #             # Run all strictness levels
# #             results = {}
# #             for level in ["low", "medium", "high"]:
# #                 print(f"\n{'=' * 60}")
# #                 print(f"🔄 Running {level.upper()} strictness filtering...")
# #                 print(f"{'=' * 60}")
# #
# #                 result = filter_triples_by_technology(
# #                     input_file=input_file,
# #                     output_dir="data/filtered_triples",
# #                     strictness=level
# #                 )
# #
# #                 if result:
# #                     results[level] = result
# #                     summary = result['statistics']
# #                     print(f"✅ {level.title()} filtering: {summary['retention_rate']:.1%} retention rate")
# #
# #             # Compare results
# #             if results:
# #                 print(f"\n📊 COMPARISON SUMMARY")
# #                 print("-" * 30)
# #                 for level, result in results.items():
# #                     summary = result['statistics']
# #                     print(f"{level.title():>8}: {summary['technical_kept']:,} kept "
# #                           f"({summary['retention_rate']:.1%} retention)")
# #
# #                 print(f"\n💡 Recommendations:")
# #                 print(f"• Use 'low' if you want maximum coverage")
# #                 print(f"• Use 'medium' for balanced quality/quantity (recommended)")
# #                 print(f"• Use 'high' for highest quality technical content only")
# #         else:
# #             # Run single strictness level
# #             result = filter_triples_by_technology(
# #                 input_file=input_file,
# #                 output_dir="data/filtered_triples",
# #                 strictness=selected_strictness
# #             )
# #
# #             if result:
# #                 summary = result['statistics']
# #                 print(f"\n🎯 Final Result: {summary['technical_kept']:,} technical triples kept "
# #                       f"({summary['retention_rate']:.1%} retention rate)")
# #
# #                 # Show file paths for next steps
# #                 clean_file = f"data/filtered_triples/technical_triples_clean_{selected_strictness}.json"
# #                 print(f"\n📁 Use this file for your validator:")
# #                 print(f"   {clean_file}")
# #
# #                 return result
# #
# #     except Exception as e:
# #         print(f"❌ Error in main execution: {e}")
# #         return None
# #
# #
# # if __name__ == "__main__":
# #     main()
#
#
# """
# Advanced Gemma-Enhanced Validator with:
# 1. Adaptive threshold tuning
# 2. Domain-specific prompts
# 3. Robust offline fallback strategies
# """
#
# import json
# import torch
# import numpy as np
# from datetime import datetime
# from pathlib import Path
# from collections import defaultdict, Counter
# from typing import Dict, List, Tuple, Optional, Union
# import logging
# from dataclasses import dataclass
# import pickle
# import time
#
# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
#
# @dataclass
# class ValidationThresholds:
#     """Adaptive validation thresholds"""
#     confidence_threshold: float = 0.3
#     semantic_threshold: float = 0.7
#     combined_threshold: float = 0.6
#     rule_weight: float = 0.4
#     semantic_weight: float = 0.6
#
#     # Technology-specific thresholds
#     tech_specific: Dict[str, Dict[str, float]] = None
#
#     def __post_init__(self):
#         if self.tech_specific is None:
#             self.tech_specific = {}
#
#
# @dataclass
# class DomainPromptTemplate:
#     """Domain-specific prompt templates"""
#     name: str
#     validation_focus: List[str]
#     context_keywords: List[str]
#     common_patterns: List[str]
#     invalid_patterns: List[str]
#     prompt_template: str
#
#
# class AdvancedGemmaValidator:
#     def __init__(self,
#                  ontology_file="data/ontology/generalized_ontology.json",
#                  triples_file="data/extracted_triples/ontology_enhanced_triples_python.json",
#                  output_dir="data/validated_data",
#                  use_gemma=True,
#                  gemma_model_name="google/gemma-2b-it",
#                  validation_batch_size=10,
#                  adaptive_thresholds=True,
#                  enable_offline_mode=True):
#
#         self.ontology_file = Path(ontology_file)
#         self.triples_file = Path(triples_file)
#         self.output_dir = Path(output_dir)
#         self.use_gemma = use_gemma
#         self.gemma_model_name = gemma_model_name
#         self.validation_batch_size = validation_batch_size
#         self.adaptive_thresholds = adaptive_thresholds
#         self.enable_offline_mode = enable_offline_mode
#
#         # Initialize components
#         self.ontology = None
#         self.tech_mappings = {}
#         self.gemma_model = None
#         self.gemma_tokenizer = None
#         self.validation_stats = defaultdict(int)
#
#         # Adaptive thresholds
#         self.thresholds = ValidationThresholds()
#         self.threshold_history = []
#
#         # Domain-specific prompts
#         self.domain_prompts = self._initialize_domain_prompts()
#
#         # Offline fallback components
#         self.offline_mode = False
#         self.cached_embeddings = {}
#         self.pattern_cache = {}
#         self.fallback_rules = {}
#
#         # Performance tracking
#         self.validation_performance = {
#             'rule_based_time': 0,
#             'gemma_time': 0,
#             'total_time': 0,
#             'gemma_failures': 0,
#             'offline_fallbacks': 0
#         }
#
#     def _initialize_domain_prompts(self) -> Dict[str, DomainPromptTemplate]:
#         """Initialize domain-specific prompt templates"""
#
#         return {
#             'python': DomainPromptTemplate(
#                 name="Python Programming",
#                 validation_focus=[
#                     "Python-specific syntax and semantics",
#                     "Object-oriented programming concepts",
#                     "Python standard library relationships",
#                     "Framework patterns (Django, Flask, FastAPI)"
#                 ],
#                 context_keywords=[
#                     "class", "function", "method", "module", "package",
#                     "decorator", "generator", "comprehension", "exception",
#                     "django", "flask", "fastapi", "numpy", "pandas"
#                 ],
#                 common_patterns=[
#                     "Class inherits from BaseClass",
#                     "Function returns DataType",
#                     "Module contains Function",
#                     "Decorator modifies Function",
#                     "Exception inherits from BaseException"
#                 ],
#                 invalid_patterns=[
#                     "Function inherits from Class",
#                     "Module extends Interface",
#                     "Variable implements Method"
#                 ],
#                 prompt_template="""
# You are a Python programming expert validating code relationships.
#
# TRIPLE: "{subject}" --[{relation}]--> "{object}"
#
# PYTHON CONTEXT:
# - Focus on Python-specific patterns and conventions
# - Consider OOP relationships, module structure, and framework patterns
# - Validate against Python's type system and standard library
#
# VALIDATION CRITERIA:
# 1. Python Semantic Correctness: Does this follow Python conventions?
# 2. Type Compatibility: Are the types/concepts compatible in Python?
# 3. Framework Relevance: If framework-specific, is it accurate?
# 4. Standard Library Alignment: Does it align with Python stdlib patterns?
#
# Common Valid Patterns:
# - Class inherits from BaseClass
# - Function returns DataType
# - Module contains Function/Class
# - Decorator enhances Function
#
# Invalid Patterns:
# - Function inherits from Class
# - Variable implements Interface
# - Cross-language mixing without context
#
# Respond with JSON: {{"is_valid": bool, "confidence": 0.0-1.0, "reasoning": "explanation", "python_specific_score": 0.0-1.0}}
# """
#             ),
#
#             'java': DomainPromptTemplate(
#                 name="Java Programming",
#                 validation_focus=[
#                     "Java type system and inheritance",
#                     "Interface and abstract class relationships",
#                     "Package and module organization",
#                     "Framework patterns (Spring, Hibernate)"
#                 ],
#                 context_keywords=[
#                     "class", "interface", "abstract", "package", "method",
#                     "inheritance", "polymorphism", "annotation", "generic",
#                     "spring", "hibernate", "maven", "gradle"
#                 ],
#                 common_patterns=[
#                     "Class implements Interface",
#                     "Class extends SuperClass",
#                     "Method returns Type",
#                     "Package contains Class",
#                     "Annotation modifies Class/Method"
#                 ],
#                 invalid_patterns=[
#                     "Interface extends Class",
#                     "Method inherits from Class",
#                     "Package implements Interface"
#                 ],
#                 prompt_template="""
# You are a Java programming expert validating code relationships.
#
# TRIPLE: "{subject}" --[{relation}]--> "{object}"
#
# JAVA CONTEXT:
# - Focus on Java's strict type system and OOP principles
# - Consider interface/class relationships and package structure
# - Validate against Java conventions and enterprise patterns
#
# VALIDATION CRITERIA:
# 1. Java Type System: Does this respect Java's type hierarchy?
# 2. OOP Principles: Is the relationship valid in Java OOP?
# 3. Enterprise Patterns: If enterprise-related, is it accurate?
# 4. Convention Compliance: Does it follow Java naming/structural conventions?
#
# Valid Java Patterns:
# - Class implements Interface
# - Class extends SuperClass
# - Method returns Type
# - Package contains Class
#
# Invalid Java Patterns:
# - Interface extends Class
# - Method inherits Class
# - Primitive extends Object
#
# Respond with JSON: {{"is_valid": bool, "confidence": 0.0-1.0, "reasoning": "explanation", "java_specific_score": 0.0-1.0}}
# """
#             ),
#
#             'javascript': DomainPromptTemplate(
#                 name="JavaScript Programming",
#                 validation_focus=[
#                     "JavaScript dynamic typing and prototypes",
#                     "ES6+ features and modules",
#                     "Framework patterns (React, Vue, Angular)",
#                     "Node.js and browser API relationships"
#                 ],
#                 context_keywords=[
#                     "function", "object", "prototype", "class", "module",
#                     "async", "promise", "callback", "event", "dom",
#                     "react", "vue", "angular", "node", "express"
#                 ],
#                 common_patterns=[
#                     "Function returns Promise",
#                     "Object has Property",
#                     "Module exports Function",
#                     "Component has State",
#                     "Event triggers Handler"
#                 ],
#                 invalid_patterns=[
#                     "Function implements Interface",
#                     "Variable extends Class",
#                     "DOM inherits from Node"
#                 ],
#                 prompt_template="""
# You are a JavaScript/Node.js expert validating code relationships.
#
# TRIPLE: "{subject}" --[{relation}]--> "{object}"
#
# JAVASCRIPT CONTEXT:
# - Focus on JavaScript's dynamic nature and prototype-based inheritance
# - Consider modern ES6+ features, async patterns, and frameworks
# - Validate against browser/Node.js API relationships
#
# VALIDATION CRITERIA:
# 1. JavaScript Semantics: Does this make sense in JS's dynamic environment?
# 2. Prototype/Class Relationships: Are inheritance patterns valid?
# 3. Framework Accuracy: If framework-specific, is it technically correct?
# 4. API Compatibility: Does it align with browser/Node.js APIs?
#
# Valid JS Patterns:
# - Function returns Promise/Type
# - Object has Property
# - Module exports Function
# - Component manages State
#
# Invalid JS Patterns:
# - Function implements Interface (not native JS)
# - Variable extends Class
# - Incorrect async/await patterns
#
# Respond with JSON: {{"is_valid": bool, "confidence": 0.0-1.0, "reasoning": "explanation", "javascript_specific_score": 0.0-1.0}}
# """
#             ),
#
#             'web': DomainPromptTemplate(
#                 name="Web Technologies",
#                 validation_focus=[
#                     "HTML/CSS/JavaScript integration",
#                     "Web API and browser relationships",
#                     "HTTP and REST API patterns",
#                     "Frontend/backend communication"
#                 ],
#                 context_keywords=[
#                     "html", "css", "dom", "element", "attribute",
#                     "http", "rest", "api", "request", "response",
#                     "browser", "client", "server", "endpoint"
#                 ],
#                 common_patterns=[
#                     "Element has Attribute",
#                     "CSS styles Element",
#                     "Request returns Response",
#                     "API serves Data",
#                     "Browser renders HTML"
#                 ],
#                 invalid_patterns=[
#                     "CSS implements JavaScript",
#                     "HTML extends Database",
#                     "Element inherits from Server"
#                 ],
#                 prompt_template="""
# You are a web development expert validating web technology relationships.
#
# TRIPLE: "{subject}" --[{relation}]--> "{object}"
#
# WEB CONTEXT:
# - Focus on HTML/CSS/JS interaction and web standards
# - Consider client-server relationships and web APIs
# - Validate against browser capabilities and web protocols
#
# VALIDATION CRITERIA:
# 1. Web Standards Compliance: Does this align with web standards?
# 2. Client-Server Logic: Are client/server relationships accurate?
# 3. Technology Integration: Do the technologies work together?
# 4. Protocol Accuracy: Are HTTP/API patterns correct?
#
# Valid Web Patterns:
# - Element has Attribute
# - CSS styles Element
# - Request returns Response
# - API provides Service
#
# Invalid Web Patterns:
# - CSS inherits JavaScript
# - HTML implements Database
# - Browser extends Server
#
# Respond with JSON: {{"is_valid": bool, "confidence": 0.0-1.0, "reasoning": "explanation", "web_specific_score": 0.0-1.0}}
# """
#             ),
#
#             'database': DomainPromptTemplate(
#                 name="Database Technologies",
#                 validation_focus=[
#                     "SQL and NoSQL database concepts",
#                     "Table/collection relationships",
#                     "Query and transaction patterns",
#                     "Database design principles"
#                 ],
#                 context_keywords=[
#                     "table", "column", "row", "index", "query",
#                     "database", "sql", "nosql", "transaction", "schema",
#                     "mongodb", "mysql", "postgresql", "redis"
#                 ],
#                 common_patterns=[
#                     "Table has Column",
#                     "Query returns Result",
#                     "Index improves Performance",
#                     "Transaction contains Operation",
#                     "Database stores Data"
#                 ],
#                 invalid_patterns=[
#                     "Column extends Table",
#                     "Query implements Database",
#                     "Index inherits from Row"
#                 ],
#                 prompt_template="""
# You are a database expert validating database technology relationships.
#
# TRIPLE: "{subject}" --[{relation}]--> "{object}"
#
# DATABASE CONTEXT:
# - Focus on relational and NoSQL database concepts
# - Consider SQL query patterns and database design
# - Validate against ACID properties and database theory
#
# VALIDATION CRITERIA:
# 1. Database Theory: Does this align with database principles?
# 2. SQL/NoSQL Accuracy: Is the relationship valid for the database type?
# 3. Performance Logic: Are performance-related claims accurate?
# 4. Design Patterns: Does it follow good database design?
#
# Valid Database Patterns:
# - Table has Column/Row
# - Query returns ResultSet
# - Index optimizes Query
# - Database contains Table
#
# Invalid Database Patterns:
# - Column inherits Table
# - Query extends Database
# - Row implements Index
#
# Respond with JSON: {{"is_valid": bool, "confidence": 0.0-1.0, "reasoning": "explanation", "database_specific_score": 0.0-1.0}}
# """
#             )
#         }
#
#     def detect_technology_domain(self, triple: Dict) -> str:
#         """Automatically detect the technology domain for a triple"""
#
#         technology = triple.get('technology', '').lower()
#         subject = triple.get('subject', '').lower()
#         obj = triple.get('object', '').lower()
#         source_text = triple.get('source_text', '').lower()
#
#         # Direct technology mapping
#         if technology in self.domain_prompts:
#             return technology
#
#         # Content-based detection
#         all_text = f"{subject} {obj} {source_text}"
#
#         domain_scores = {}
#         for domain, template in self.domain_prompts.items():
#             score = 0
#             for keyword in template.context_keywords:
#                 score += all_text.count(keyword.lower())
#             domain_scores[domain] = score
#
#         # Return domain with highest score, default to 'python'
#         if domain_scores and max(domain_scores.values()) > 0:
#             return max(domain_scores, key=domain_scores.get)
#
#         return 'python'  # Default fallback
#
#     def create_domain_specific_prompt(self, triple: Dict, ontology_context: Dict) -> str:
#         """Create domain-specific validation prompt"""
#
#         domain = self.detect_technology_domain(triple)
#         template = self.domain_prompts.get(domain, self.domain_prompts['python'])
#
#         subject = triple.get('subject', '')
#         relation = triple.get('relation', '')
#         obj = triple.get('object', '')
#
#         # Customize prompt with ontology context
#         ontology_info = f"""
# ONTOLOGY ENTITIES: {', '.join(ontology_context.get('entities', [])[:15])}
# ONTOLOGY RELATIONS: {', '.join(ontology_context.get('relations', [])[:10])}
# DOMAIN: {template.name}
# """
#
#         final_prompt = template.prompt_template.format(
#             subject=subject,
#             relation=relation,
#             object=obj
#         ) + f"\n\n{ontology_info}"
#
#         return final_prompt
#
#     def analyze_validation_performance(self, results: List[Dict]) -> Dict:
#         """Analyze validation performance to tune thresholds"""
#
#         if not results:
#             return {}
#
#         analysis = {
#             'total_triples': len(results),
#             'valid_count': 0,
#             'invalid_count': 0,
#             'score_distribution': {
#                 'rule_based': [],
#                 'semantic': [],
#                 'combined': []
#             },
#             'confidence_stats': {},
#             'domain_performance': defaultdict(lambda: {'valid': 0, 'total': 0}),
#             'relation_performance': defaultdict(lambda: {'valid': 0, 'total': 0}),
#             'threshold_recommendations': {}
#         }
#
#         for result in results:
#             validation = result.get('validation_result', {})
#             triple = result.get('original_triple', {})
#
#             is_valid = validation.get('is_valid', False)
#             rule_score = validation.get('rule_based_score', 0)
#             semantic_score = validation.get('semantic_score', 0)
#             final_score = validation.get('final_score', 0)
#
#             if is_valid:
#                 analysis['valid_count'] += 1
#                 node['avg_validation_score'] = 0.0
#
#         return list(nodes.values()), edges
#
#     def _analyze_domain_performance(self, results: List[Dict]) -> Dict:
#         """Analyze performance by domain"""
#         domain_stats = defaultdict(lambda: {
#             'total': 0,
#             'valid': 0,
#             'avg_rule_score': 0,
#             'avg_semantic_score': 0,
#             'avg_final_score': 0,
#             'validation_methods': defaultdict(int),
#             'common_issues': defaultdict(int)
#         })
#
#         for result in results:
#             validation = result['validation_result']
#             domain = validation.get('domain', 'unknown')
#
#             stats = domain_stats[domain]
#             stats['total'] += 1
#
#             if validation['is_valid']:
#                 stats['valid'] += 1
#
#             stats['avg_rule_score'] += validation.get('rule_based_score', 0)
#             stats['avg_semantic_score'] += validation.get('semantic_score', 0)
#             stats['avg_final_score'] += validation.get('final_score', 0)
#
#             # Track validation methods
#             method = validation.get('validation_method', 'unknown')
#             stats['validation_methods'][method] += 1
#
#             # Track common issues
#             for issue in validation.get('issues', []):
#                 issue_type = issue.split(':')[0] if ':' in issue else issue
#                 stats['common_issues'][issue_type] += 1
#
#         # Calculate averages
#         for domain, stats in domain_stats.items():
#             if stats['total'] > 0:
#                 stats['success_rate'] = stats['valid'] / stats['total']
#                 stats['avg_rule_score'] /= stats['total']
#                 stats['avg_semantic_score'] /= stats['total']
#                 stats['avg_final_score'] /= stats['total']
#
#             # Convert defaultdicts to regular dicts
#             stats['validation_methods'] = dict(stats['validation_methods'])
#             stats['common_issues'] = dict(stats['common_issues'])
#
#         return dict(domain_stats)
#
#     def _generate_final_recommendations(self, results: List[Dict]) -> Dict:
#         """Generate final recommendations for improvement"""
#         if not results:
#             return {}
#
#         recommendations = {
#             'threshold_adjustments': {},
#             'domain_specific_improvements': {},
#             'validation_method_recommendations': {},
#             'data_quality_insights': {}
#         }
#
#         # Analyze overall performance
#         total_results = len(results)
#         valid_results = len([r for r in results if r['validation_result']['is_valid']])
#         success_rate = valid_results / total_results if total_results > 0 else 0
#
#         # Threshold recommendations
#         if success_rate < 0.7:
#             recommendations['threshold_adjustments']['lower_thresholds'] = {
#                 'reason': 'Low success rate detected',
#                 'suggested_combined_threshold': max(0.4, self.thresholds.combined_threshold - 0.1),
#                 'suggested_semantic_threshold': max(0.5, self.thresholds.semantic_threshold - 0.1)
#             }
#         elif success_rate > 0.95:
#             recommendations['threshold_adjustments']['raise_thresholds'] = {
#                 'reason': 'Very high success rate - can be more selective',
#                 'suggested_combined_threshold': min(0.8, self.thresholds.combined_threshold + 0.1),
#                 'suggested_semantic_threshold': min(0.9, self.thresholds.semantic_threshold + 0.1)
#             }
#
#         # Validation method recommendations
#         method_performance = defaultdict(lambda: {'total': 0, 'valid': 0})
#         for result in results:
#             method = result['validation_result'].get('validation_method', 'unknown')
#             method_performance[method]['total'] += 1
#             if result['validation_result']['is_valid']:
#                 method_performance[method]['valid'] += 1
#
#         for method, perf in method_performance.items():
#             if perf['total'] > 10:
#                 success_rate = perf['valid'] / perf['total']
#                 if success_rate < 0.6:
#                     recommendations['validation_method_recommendations'][method] = {
#                         'status': 'underperforming',
#                         'success_rate': success_rate,
#                         'suggestion': 'Consider adjusting method parameters or weights'
#                     }
#
#         # Data quality insights
#         issue_frequency = defaultdict(int)
#         for result in results:
#             for issue in result['validation_result'].get('issues', []):
#                 issue_type = issue.split(':')[0] if ':' in issue else issue
#                 issue_frequency[issue_type] += 1
#
#         top_issues = sorted(issue_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
#         recommendations['data_quality_insights']['top_issues'] = top_issues
#
#         if top_issues:
#             most_common_issue = top_issues[0][0]
#             recommendations['data_quality_insights']['priority_fix'] = {
#                 'issue': most_common_issue,
#                 'frequency': top_issues[0][1],
#                 'percentage': (top_issues[0][1] / total_results) * 100
#             }
#
#         return recommendations
#
#     def save_enhanced_results(self, data: Dict) -> Optional[Dict]:
#         """Save comprehensive enhanced results"""
#         try:
#             self.output_dir.mkdir(parents=True, exist_ok=True)
#
#             # Save main results
#             main_file = self.output_dir / "advanced_gemma_validation.json"
#             with open(main_file, 'w', encoding='utf-8') as f:
#                 json.dump(data, f, indent=2, ensure_ascii=False)
#
#             # Save graph data separately
#             graph_file = self.output_dir / "enhanced_graph_data.json"
#             with open(graph_file, 'w', encoding='utf-8') as f:
#                 json.dump(data['graph_data'], f, indent=2, ensure_ascii=False)
#
#             # Save threshold history
#             threshold_file = self.output_dir / "threshold_optimization_history.json"
#             with open(threshold_file, 'w', encoding='utf-8') as f:
#                 json.dump({
#                     'threshold_history': self.threshold_history,
#                     'final_thresholds': self.thresholds.__dict__,
#                     'domain_prompts': {k: v.name for k, v in self.domain_prompts.items()}
#                 }, f, indent=2, ensure_ascii=False)
#
#             # Save performance analysis
#             performance_file = self.output_dir / "validation_performance_analysis.json"
#             with open(performance_file, 'w', encoding='utf-8') as f:
#                 json.dump({
#                     'performance_metrics': self.validation_performance,
#                     'domain_performance': data.get('domain_performance', {}),
#                     'validation_analysis': data.get('validation_analysis', {}),
#                     'recommendations': data.get('threshold_recommendations', {})
#                 }, f, indent=2, ensure_ascii=False)
#
#             # Print comprehensive summary
#             self.print_comprehensive_summary(data)
#
#             print(f"\n💾 Advanced validation results saved to: {self.output_dir}")
#             print(f"📁 Files created:")
#             print(f"   • advanced_gemma_validation.json (main results)")
#             print(f"   • enhanced_graph_data.json (graph structure)")
#             print(f"   • threshold_optimization_history.json (threshold evolution)")
#             print(f"   • validation_performance_analysis.json (performance metrics)")
#
#             return data
#
#         except Exception as e:
#             print(f"❌ Error saving enhanced results: {e}")
#             return None
#
#     def print_comprehensive_summary(self, data: Dict):
#         """Print comprehensive validation summary"""
#         metadata = data['metadata']
#         stats = metadata['validation_stats']
#         performance = metadata['performance_metrics']
#         analysis = data.get('validation_analysis', {})
#         domain_perf = data.get('domain_performance', {})
#
#         print(f"\n🚀 ADVANCED GEMMA-ENHANCED VALIDATION SUMMARY")
#         print("=" * 60)
#
#         # Basic stats
#         print(f"📊 Validation Statistics:")
#         print(f"   Total Processed: {stats.get('total_processed', 0):,}")
#         print(f"   Valid Triples: {stats.get('valid_triples', 0):,}")
#         print(f"   Invalid Triples: {stats.get('invalid_triples', 0):,}")
#
#         if stats.get('total_processed', 0) > 0:
#             success_rate = stats.get('valid_triples', 0) / stats.get('total_processed', 1) * 100
#             print(f"   Success Rate: {success_rate:.1f}%")
#
#         # Validation method breakdown
#         print(f"\n🤖 Validation Method Performance:")
#         print(f"   Gemma Model: {metadata.get('gemma_model', 'N/A')}")
#         print(f"   Offline Mode: {'Yes' if metadata.get('offline_mode_used') else 'No'}")
#         print(f"   Adaptive Thresholds: {'Yes' if metadata.get('adaptive_thresholds') else 'No'}")
#         print(f"   Gemma Failures: {performance.get('gemma_failures', 0)}")
#         print(f"   Offline Fallbacks: {performance.get('offline_fallbacks', 0)}")
#
#         # Performance metrics
#         print(f"\n⚡ Performance Metrics:")
#         total_time = performance.get('total_time', 0)
#         rule_time = performance.get('rule_based_time', 0)
#         gemma_time = performance.get('gemma_time', 0)
#
#         print(f"   Total Time: {total_time:.2f}s")
#         print(f"   Rule-based Time: {rule_time:.2f}s ({rule_time / total_time * 100:.1f}%)")
#         print(f"   Semantic Time: {gemma_time:.2f}s ({gemma_time / total_time * 100:.1f}%)")
#
#         if stats.get('total_processed', 0) > 0:
#             avg_time = total_time / stats.get('total_processed', 1)
#             print(f"   Avg Time per Triple: {avg_time:.3f}s")
#
#         # Threshold information
#         print(f"\n🎯 Final Thresholds:")
#         final_thresholds = metadata.get('final_thresholds', {})
#         print(f"   Combined Threshold: {final_thresholds.get('combined_threshold', 'N/A')}")
#         print(f"   Semantic Threshold: {final_thresholds.get('semantic_threshold', 'N/A')}")
#         print(f"   Confidence Threshold: {final_thresholds.get('confidence_threshold', 'N/A')}")
#
#         # Domain performance
#         if domain_perf:
#             print(f"\n🌐 Domain Performance:")
#             for domain, perf in list(domain_perf.items())[:5]:  # Show top 5 domains
#                 success_rate = perf.get('success_rate', 0) * 100
#                 total = perf.get('total', 0)
#                 print(f"   {domain.capitalize()}: {success_rate:.1f}% ({total} triples)")
#
#         # Quality metrics
#         if analysis and 'confidence_stats' in analysis:
#             conf_stats = analysis['confidence_stats']
#             if 'combined' in conf_stats:
#                 combined = conf_stats['combined']
#                 print(f"\n📈 Quality Distribution:")
#                 print(f"   Mean Score: {combined.get('mean', 0):.3f}")
#                 print(f"   Median Score: {combined.get('median', 0):.3f}")
#                 print(f"   Score Range: {combined.get('q25', 0):.3f} - {combined.get('q75', 0):.3f}")
#
#         # Graph structure
#         print(f"\n🏗️ Knowledge Graph:")
#         print(f"   Nodes: {metadata.get('total_nodes', 0):,}")
#         print(f"   Edges: {metadata.get('total_edges', 0):,}")
#
#         # Recommendations
#         recommendations = data.get('threshold_recommendations', {})
#         if recommendations:
#             print(f"\n💡 Key Recommendations:")
#
#             if 'threshold_adjustments' in recommendations:
#                 for adj_type, details in recommendations['threshold_adjustments'].items():
#                     print(f"   • {adj_type.replace('_', ' ').title()}: {details.get('reason', 'N/A')}")
#
#             if 'data_quality_insights' in recommendations:
#                 insights = recommendations['data_quality_insights']
#                 if 'priority_fix' in insights:
#                     priority = insights['priority_fix']
#                     print(
#                         f"   • Priority Issue: {priority.get('issue', 'N/A')} ({priority.get('percentage', 0):.1f}% of triples)")
#
#         print(f"\n✅ Advanced validation completed successfully!")
#
#
# def main():
#     """Run advanced Gemma-enhanced validation with all features"""
#     print("🚀 Advanced Gemma-Enhanced Knowledge Graph Validator")
#     print("=" * 60)
#     print("Features:")
#     print("• 🤖 Gemma LLM integration with domain-specific prompts")
#     print("• 📊 Adaptive threshold optimization")
#     print("• 🔄 Intelligent offline fallback strategies")
#     print("• 🌐 Multi-domain validation (Python, Java, JavaScript, Web, Database)")
#     print("• ⚡ Performance monitoring and optimization")
#     print("• 🎯 Automated quality analysis and recommendations")
#
#     # Configuration options
#     config_options = {
#         "1": {
#             "name": "Full Featured (Recommended)",
#             "config": {
#                 "use_gemma": True,
#                 "gemma_model_name": "google/gemma-2b-it",
#                 "adaptive_thresholds": True,
#                 "enable_offline_mode": True,
#                 "validation_batch_size": 10
#             }
#         },
#         "2": {
#             "name": "Lightweight (Faster)",
#             "config": {
#                 "use_gemma": True,
#                 "gemma_model_name": "google/gemma-2b-it",
#                 "adaptive_thresholds": False,
#                 "enable_offline_mode": True,
#                 "validation_batch_size": 20
#             }
#         },
#         "3": {
#             "name": "Offline Only (No Internet Required)",
#             "config": {
#                 "use_gemma": False,
#                 "adaptive_thresholds": True,
#                 "enable_offline_mode": True,
#                 "validation_batch_size": 50
#             }
#         },
#         "4": {
#             "name": "High Quality (Slower, Better Results)",
#             "config": {
#                 "use_gemma": True,
#                 "gemma_model_name": "google/gemma-7b-it",
#                 "adaptive_thresholds": True,
#                 "enable_offline_mode": True,
#                 "validation_batch_size": 5
#             }
#         }
#     }
#
#     print(f"\n📋 Configuration Options:")
#     for key, option in config_options.items():
#         print(f"   {key}. {option['name']}")
#
#     # Use default configuration (Full Featured)
#     selected_config = config_options["1"]["config"]
#     print(f"\n🎯 Using: {config_options['1']['name']} configuration")
#
#     try:
#         validator = AdvancedGemmaValidator(**selected_config)
#
#         print(f"\n🔧 Configuration:")
#         print(f"   • Gemma Model: {validator.gemma_model_name if validator.use_gemma else 'Disabled'}")
#         print(f"   • Adaptive Thresholds: {'Enabled' if validator.adaptive_thresholds else 'Disabled'}")
#         print(f"   • Offline Fallback: {'Enabled' if validator.enable_offline_mode else 'Disabled'}")
#         print(f"   • Batch Size: {validator.validation_batch_size}")
#
#         results = validator.validate_all_enhanced()
#
#         if results:
#             print(f"\n🎉 Advanced validation completed successfully!")
#             print(f"🔗 High-quality, semantically validated knowledge graph ready!")
#
#             # Show key insights
#             metadata = results.get('metadata', {})
#             stats = metadata.get('validation_stats', {})
#             performance = metadata.get('performance_metrics', {})
#
#             print(f"\n📋 Quick Summary:")
#             print(f"   • Processed: {stats.get('total_processed', 0):,} triples")
#             print(f"   • Valid: {stats.get('valid_triples', 0):,} triples")
#             print(
#                 f"   • Success Rate: {stats.get('valid_triples', 0) / max(1, stats.get('total_processed', 1)) * 100:.1f}%")
#             print(f"   • Processing Time: {performance.get('total_time', 0):.1f}s")
#             print(
#                 f"   • Graph Size: {metadata.get('total_nodes', 0):,} nodes, {metadata.get('total_edges', 0):,} edges")
#
#             return results
#         else:
#             print("❌ Advanced validation failed!")
#             return None
#
#     except Exception as e:
#         print(f"❌ Critical error in advanced validation: {e}")
#         logger.exception("Detailed error information:")
#         return None
#
#
# if __name__ == "__main__":
#     main()
#     analysis['invalid_count'] += 1
#
#     # Collect scores
# analysis['score_distribution']['rule_based'].append(rule_score)
# analysis['score_distribution']['semantic'].append(semantic_score)
# analysis['score_distribution']['combined'].append(final_score)
#
# # Domain performance
# domain = self.detect_technology_domain(triple)
# analysis['domain_performance'][domain]['total'] += 1
# if is_valid:
#     analysis['domain_performance'][domain]['valid'] += 1
#
# # Relation performance
# relation = triple.get('relation', 'unknown')
# analysis['relation_performance'][relation]['total'] += 1
# if is_valid:
#     analysis['relation_performance'][relation]['valid'] += 1
#
# # Calculate statistics
# for score_type, scores in analysis['score_distribution'].items():
#     if scores:
#         analysis['confidence_stats'][score_type] = {
#             'mean': np.mean(scores),
#             'std': np.std(scores),
#             'median': np.median(scores),
#             'q25': np.percentile(scores, 25),
#             'q75': np.percentile(scores, 75)
#         }
#
#     # Generate threshold recommendations
# analysis['threshold_recommendations'] = self._generate_threshold_recommendations(analysis)
#
# return analysis
#
#
# def _generate_threshold_recommendations(self, analysis: Dict) -> Dict:
#     """Generate adaptive threshold recommendations"""
#
#     recommendations = {}
#
#     # Overall thresholds based on score distribution
#     if 'combined' in analysis['confidence_stats']:
#         combined_stats = analysis['confidence_stats']['combined']
#
#         # Recommend threshold at 25th percentile for more inclusive validation
#         # or 50th percentile for balanced approach
#         recommendations['combined_threshold'] = max(0.4, combined_stats['q25'])
#
#         # Semantic threshold based on semantic score distribution
#         if 'semantic' in analysis['confidence_stats']:
#             semantic_stats = analysis['confidence_stats']['semantic']
#             recommendations['semantic_threshold'] = max(0.6, semantic_stats['q25'])
#
#     # Domain-specific recommendations
#     domain_recommendations = {}
#     for domain, perf in analysis['domain_performance'].items():
#         if perf['total'] > 10:  # Only if sufficient data
#             success_rate = perf['valid'] / perf['total']
#             if success_rate < 0.7:
#                 # Lower thresholds for domains with low success rates
#                 domain_recommendations[domain] = {
#                     'combined_threshold': max(0.3, self.thresholds.combined_threshold - 0.1),
#                     'semantic_threshold': max(0.5, self.thresholds.semantic_threshold - 0.1)
#                 }
#             elif success_rate > 0.9:
#                 # Raise thresholds for high-performing domains
#                 domain_recommendations[domain] = {
#                     'combined_threshold': min(0.8, self.thresholds.combined_threshold + 0.1),
#                     'semantic_threshold': min(0.9, self.thresholds.semantic_threshold + 0.1)
#                 }
#
#     recommendations['domain_specific'] = domain_recommendations
#
#     return recommendations
#
#
# def apply_adaptive_thresholds(self, analysis: Dict):
#     """Apply adaptive threshold adjustments"""
#
#     if not self.adaptive_thresholds or not analysis.get('threshold_recommendations'):
#         return
#
#     recommendations = analysis['threshold_recommendations']
#
#     # Apply overall threshold adjustments (conservative approach)
#     if 'combined_threshold' in recommendations:
#         new_threshold = recommendations['combined_threshold']
#         # Only adjust if change is significant and reasonable
#         if abs(new_threshold - self.thresholds.combined_threshold) > 0.05:
#             old_threshold = self.thresholds.combined_threshold
#             self.thresholds.combined_threshold = new_threshold
#             print(f"📊 Adjusted combined threshold: {old_threshold:.3f} → {new_threshold:.3f}")
#
#     if 'semantic_threshold' in recommendations:
#         new_threshold = recommendations['semantic_threshold']
#         if abs(new_threshold - self.thresholds.semantic_threshold) > 0.05:
#             old_threshold = self.thresholds.semantic_threshold
#             self.thresholds.semantic_threshold = new_threshold
#             print(f"📊 Adjusted semantic threshold: {old_threshold:.3f} → {new_threshold:.3f}")
#
#     # Apply domain-specific thresholds
#     domain_recommendations = recommendations.get('domain_specific', {})
#     for domain, thresholds in domain_recommendations.items():
#         if domain not in self.thresholds.tech_specific:
#             self.thresholds.tech_specific[domain] = {}
#
#         self.thresholds.tech_specific[domain].update(thresholds)
#         print(f"📊 Applied domain-specific thresholds for {domain}: {thresholds}")
#
#     # Save threshold history for analysis
#     self.threshold_history.append({
#         'timestamp': datetime.now().isoformat(),
#         'thresholds': self.thresholds.__dict__.copy(),
#         'analysis': analysis['confidence_stats']
#     })
#
#
# def get_domain_thresholds(self, domain: str) -> Dict[str, float]:
#     """Get thresholds for specific domain"""
#
#     domain_thresholds = self.thresholds.tech_specific.get(domain, {})
#
#     return {
#         'combined_threshold': domain_thresholds.get('combined_threshold', self.thresholds.combined_threshold),
#         'semantic_threshold': domain_thresholds.get('semantic_threshold', self.thresholds.semantic_threshold),
#         'confidence_threshold': domain_thresholds.get('confidence_threshold', self.thresholds.confidence_threshold)
#     }
#
#
# def initialize_offline_fallback(self):
#     """Initialize offline fallback components"""
#
#     print("🔄 Initializing offline fallback strategies...")
#
#     # Create pattern-based fallback rules
#     self.fallback_rules = {
#         'high_confidence_patterns': [
#             # Strong positive patterns
#             (r'(\w+Class|Class\w+)\s+(extends|inherits)\s+(\w+Class|Class\w+)', 0.9),
#             (r'(\w+Interface|Interface\w+)\s+(implements|extends)\s+(\w+Interface|Interface\w+)', 0.9),
#             (r'(\w+Method|Method\w+)\s+(returns|produces)\s+(\w+Type|\w+Class)', 0.85),
#             (r'(\w+Function|Function\w+)\s+(takes|accepts)\s+(\w+Parameter|\w+Type)', 0.85),
#             (r'(\w+Service|Service\w+)\s+(uses|utilizes)\s+(\w+Database|Database\w+)', 0.8),
#             (r'(\w+Component|Component\w+)\s+(contains|includes)\s+(\w+Element|Element\w+)', 0.8),
#         ],
#         'medium_confidence_patterns': [
#             (r'(\w+)\s+(has|contains)\s+(\w+)', 0.7),
#             (r'(\w+)\s+(uses|utilizes)\s+(\w+)', 0.65),
#             (r'(\w+)\s+(provides|supplies)\s+(\w+)', 0.65),
#             (r'(\w+)\s+(manages|controls)\s+(\w+)', 0.6),
#         ],
#         'negative_patterns': [
#             # Patterns that should be flagged as invalid
#             (r'(\w+)\s+(inherits|extends)\s+(String|Integer|Boolean|Float)', 0.1),
#             (r'(String|Integer|Boolean|Float)\s+(contains|includes)\s+(\w+Class)', 0.1),
#             (r'(\w+Variable)\s+(implements|extends)\s+(\w+)', 0.1),
#             (r'(\w+HTML)\s+(extends|inherits)\s+(\w+Java|\w+Python)', 0.1),
#         ]
#     }
#
#     # Load cached patterns if available
#     self._load_pattern_cache()
#
#     print("✅ Offline fallback initialized")
#
#
# def _load_pattern_cache(self):
#     """Load cached validation patterns"""
#     cache_file = self.output_dir / "validation_pattern_cache.pkl"
#
#     try:
#         if cache_file.exists():
#             with open(cache_file, 'rb') as f:
#                 self.pattern_cache = pickle.load(f)
#             print(f"📁 Loaded {len(self.pattern_cache)} cached patterns")
#     except Exception as e:
#         logger.warning(f"Could not load pattern cache: {e}")
#         self.pattern_cache = {}
#
#
# def _save_pattern_cache(self):
#     """Save validation patterns for offline use"""
#     cache_file = self.output_dir / "validation_pattern_cache.pkl"
#
#     try:
#         self.output_dir.mkdir(parents=True, exist_ok=True)
#         with open(cache_file, 'wb') as f:
#             pickle.dump(self.pattern_cache, f)
#         print(f"💾 Saved {len(self.pattern_cache)} patterns to cache")
#     except Exception as e:
#         logger.error(f"Could not save pattern cache: {e}")
#
#
# def offline_semantic_validation(self, triple: Dict, domain: str) -> Dict:
#     """Fallback semantic validation using pattern matching"""
#
#     self.validation_performance['offline_fallbacks'] += 1
#
#     subject = triple.get('subject', '')
#     relation = triple.get('relation', '')
#     obj = triple.get('object', '')
#
#     # Create triple text for pattern matching
#     triple_text = f"{subject} {relation} {obj}"
#
#     # Check cache first
#     cache_key = (subject, relation, obj, domain)
#     if cache_key in self.pattern_cache:
#         return self.pattern_cache[cache_key]
#
#     # Apply pattern-based validation
#     max_confidence = 0.5  # Default neutral confidence
#     matched_pattern = None
#     validation_reasoning = "Pattern-based offline validation"
#
#     # Check high confidence patterns
#     for pattern, confidence in self.fallback_rules['high_confidence_patterns']:
#         if self._matches_pattern(triple_text, pattern):
#             max_confidence = max(max_confidence, confidence)
#             matched_pattern = "high_confidence"
#             validation_reasoning = f"Matched high-confidence pattern: {pattern[:50]}..."
#
#     # Check medium confidence patterns
#     for pattern, confidence in self.fallback_rules['medium_confidence_patterns']:
#         if self._matches_pattern(triple_text, pattern):
#             max_confidence = max(max_confidence, confidence)
#             matched_pattern = "medium_confidence"
#             validation_reasoning = f"Matched medium-confidence pattern: {pattern[:50]}..."
#
#     # Check negative patterns
#     for pattern, confidence in self.fallback_rules['negative_patterns']:
#         if self._matches_pattern(triple_text, pattern):
#             max_confidence = min(max_confidence, confidence)
#             matched_pattern = "negative"
#             validation_reasoning = f"Matched negative pattern: {pattern[:50]}..."
#
#     # Domain-specific adjustments
#     domain_template = self.domain_prompts.get(domain)
#     if domain_template:
#         # Check for domain-specific valid patterns
#         for valid_pattern in domain_template.common_patterns:
#             if self._semantic_similarity(triple_text, valid_pattern) > 0.7:
#                 max_confidence = max(max_confidence, 0.75)
#                 validation_reasoning = f"Matches domain pattern: {valid_pattern}"
#
#         # Check for domain-specific invalid patterns
#         for invalid_pattern in domain_template.invalid_patterns:
#             if self._semantic_similarity(triple_text, invalid_pattern) > 0.7:
#                 max_confidence = min(max_confidence, 0.3)
#                 validation_reasoning = f"Matches invalid domain pattern: {invalid_pattern}"
#
#     result = {
#         'is_valid': max_confidence > 0.5,
#         'confidence': max_confidence,
#         'reasoning': validation_reasoning,
#         'semantic_score': max_confidence,
#         'technical_accuracy': max_confidence,
#         'ontology_alignment': max_confidence * 0.9,
#         'suggestions': [],
#         'validation_method': 'offline_pattern_matching',
#         'matched_pattern_type': matched_pattern
#     }
#
#     # Cache result for future use
#     self.pattern_cache[cache_key] = result
#
#     return result
#
#
# def _matches_pattern(self, text: str, pattern: str) -> bool:
#     """Check if text matches pattern (simplified regex matching)"""
#     import re
#     try:
#         return bool(re.search(pattern, text, re.IGNORECASE))
#     except re.error:
#         return False
#
#
# def _semantic_similarity(self, text1: str, text2: str) -> float:
#     """Simple semantic similarity using word overlap"""
#
#     words1 = set(text1.lower().split())
#     words2 = set(text2.lower().split())
#
#     if not words1 or not words2:
#         return 0.0
#
#     intersection = words1.intersection(words2)
#     union = words1.union(words2)
#
#     return len(intersection) / len(union) if union else 0.0
#
#
# def load_gemma_model(self) -> bool:
#     """Load Gemma model with connection testing"""
#     if not self.use_gemma:
#         return False
#
#     try:
#         print(f"🤖 Loading Gemma model: {self.gemma_model_name}")
#
#         # Test internet connection first
#         if not self._test_model_availability():
#             print("⚠️ Model not available, enabling offline mode")
#             self.offline_mode = True
#             return False
#
#         from transformers import AutoTokenizer, AutoModelForCausalLM
#
#         self.gemma_tokenizer = AutoTokenizer.from_pretrained(
#             self.gemma_model_name,
#             trust_remote_code=True
#         )
#
#         self.gemma_model = AutoModelForCausalLM.from_pretrained(
#             self.gemma_model_name,
#             torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#             device_map="auto" if torch.cuda.is_available() else None,
#             trust_remote_code=True
#         )
#
#         if self.gemma_tokenizer.pad_token is None:
#             self.gemma_tokenizer.pad_token = self.gemma_tokenizer.eos_token
#
#         print("✅ Gemma model loaded successfully!")
#         return True
#
#     except Exception as e:
#         print(f"❌ Error loading Gemma model: {e}")
#         print("🔄 Enabling offline mode with pattern-based fallback")
#         self.offline_mode = True
#         return False
#
#
# def _test_model_availability(self) -> bool:
#     """Test if model is available for download"""
#     try:
#         import requests
#         from transformers import AutoTokenizer
#
#         # Quick test - try to load tokenizer config
#         AutoTokenizer.from_pretrained(self.gemma_model_name, trust_remote_code=True)
#         return True
#
#     except Exception:
#         return False
#
#
# def validate_with_gemma_enhanced(self, triple: Dict, ontology_context: Dict) -> Dict:
#     """Enhanced Gemma validation with domain-specific prompts and fallback"""
#
#     start_time = time.time()
#
#     # If offline mode or Gemma unavailable, use offline fallback
#     if self.offline_mode or not self.gemma_model:
#         domain = self.detect_technology_domain(triple)
#         result = self.offline_semantic_validation(triple, domain)
#         self.validation_performance['gemma_time'] += time.time() - start_time
#         return result
#
#     try:
#         # Create domain-specific prompt
#         prompt = self.create_domain_specific_prompt(triple, ontology_context)
#
#         # Tokenize with error handling
#         try:
#             inputs = self.gemma_tokenizer(
#                 prompt,
#                 return_tensors="pt",
#                 truncation=True,
#                 max_length=1024,
#                 padding=True
#             )
#
#             if torch.cuda.is_available():
#                 inputs = {k: v.cuda() for k, v in inputs.items()}
#
#         except Exception as e:
#             logger.warning(f"Tokenization error: {e}")
#             raise Exception("Tokenization failed")
#
#         # Generate response with timeout protection
#         try:
#             with torch.no_grad():
#                 outputs = self.gemma_model.generate(
#                     **inputs,
#                     max_new_tokens=300,
#                     temperature=0.3,
#                     do_sample=True,
#                     pad_token_id=self.gemma_tokenizer.pad_token_id,
#                     eos_token_id=self.gemma_tokenizer.eos_token_id,
#                     repetition_penalty=1.1
#                 )
#
#             response = self.gemma_tokenizer.decode(
#                 outputs[0][inputs['input_ids'].shape[1]:],
#                 skip_special_tokens=True
#             )
#
#             result = self.parse_gemma_response(response, triple)
#
#         except Exception as e:
#             logger.warning(f"Generation error: {e}")
#             raise Exception("Generation failed")
#
#     except Exception as e:
#         # Fallback to offline validation on any error
#         logger.warning(f"Gemma validation failed: {e}, using offline fallback")
#         self.validation_performance['gemma_failures'] += 1
#         domain = self.detect_technology_domain(triple)
#         result = self.offline_semantic_validation(triple, domain)
#
#     self.validation_performance['gemma_time'] += time.time() - start_time
#     return result
#
#
# def load_ontology(self) -> bool:
#     """Load ontology with enhanced error handling"""
#     try:
#         if not self.ontology_file.exists():
#             print(f"❌ Ontology file not found: {self.ontology_file}")
#             return False
#
#         with open(self.ontology_file, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#
#         # Handle different ontology formats
#         if 'ontology' in data:
#             self.ontology = data['ontology']
#             self.tech_mappings = data.get('technology_mappings', {})
#         else:
#             self.ontology = data
#             self.tech_mappings = {}
#
#         print(f"✅ Loaded ontology with {len(self.ontology.get('entities', []))} entities")
#         return True
#
#     except Exception as e:
#         print(f"❌ Error loading ontology: {e}")
#         return False
#
#
# def parse_gemma_response(self, response: str, triple: Dict) -> Dict:
#     """Enhanced response parsing with better error handling"""
#     try:
#         # Try to extract JSON from response
#         json_start = response.find('{')
#         json_end = response.rfind('}') + 1
#
#         if json_start != -1 and json_end > json_start:
#             json_str = response[json_start:json_end]
#             parsed = json.loads(json_str)
#
#             # Validate and normalize required fields
#             result = {
#                 'is_valid': bool(parsed.get('is_valid', False)),
#                 'confidence': max(0.0, min(1.0, float(parsed.get('confidence', 0.5)))),
#                 'reasoning': str(parsed.get('reasoning', 'No reasoning provided')),
#                 'semantic_score': max(0.0, min(1.0, float(parsed.get('semantic_score', 0.5)))),
#                 'technical_accuracy': max(0.0, min(1.0, float(parsed.get('technical_accuracy', 0.5)))),
#                 'ontology_alignment': max(0.0, min(1.0, float(parsed.get('ontology_alignment', 0.5)))),
#                 'suggestions': parsed.get('suggestions', []),
#                 'raw_response': response,
#                 'validation_method': 'gemma_structured'
#             }
#
#             # Add domain-specific scores if available
#             domain = self.detect_technology_domain(triple)
#             domain_score_key = f"{domain}_specific_score"
#             if domain_score_key in parsed:
#                 result[domain_score_key] = max(0.0, min(1.0, float(parsed[domain_score_key])))
#
#             return result
#
#         # Fallback to heuristic parsing
#         return self.heuristic_parse_gemma_response(response, triple)
#
#     except Exception as e:
#         logger.warning(f"Error parsing Gemma response: {e}")
#         return self.get_default_gemma_response()
#
#
# def heuristic_parse_gemma_response(self, response: str, triple: Dict) -> Dict:
#     """Enhanced heuristic parsing with domain awareness"""
#     response_lower = response.lower()
#
#     # Enhanced validation indicators
#     positive_indicators = [
#         'is valid', 'makes sense', 'correct', 'accurate', 'true',
#         'appropriate', 'reasonable', 'logical', 'consistent',
#         'follows convention', 'technically sound', 'semantically correct'
#     ]
#
#     negative_indicators = [
#         'invalid', 'incorrect', 'wrong', 'false', 'not accurate',
#         'inappropriate', 'unreasonable', 'illogical', 'inconsistent',
#         'violates', 'contradicts', 'semantically incorrect'
#     ]
#
#     # Count indicators with weights
#     positive_score = sum(2 if indicator in response_lower else 0 for indicator in positive_indicators)
#     negative_score = sum(2 if indicator in response_lower else 0 for indicator in negative_indicators)
#
#     # Additional domain-specific indicator checks
#     domain = self.detect_technology_domain(triple)
#     domain_template = self.domain_prompts.get(domain)
#
#     if domain_template:
#         # Check for domain keywords
#         for keyword in domain_template.context_keywords:
#             if keyword in response_lower:
#                 positive_score += 0.5
#
#     # Calculate confidence based on indicators
#     total_indicators = positive_score + negative_score
#     if total_indicators > 0:
#         confidence = positive_score / total_indicators
#     else:
#         confidence = 0.5  # Neutral if no clear indicators
#
#     # Adjust confidence based on response length and detail
#     if len(response) > 100:
#         confidence += 0.1  # Bonus for detailed response
#     if len(response) < 20:
#         confidence -= 0.1  # Penalty for too brief response
#
#     confidence = max(0.1, min(0.9, confidence))
#     is_valid = confidence > 0.5
#
#     return {
#         'is_valid': is_valid,
#         'confidence': confidence,
#         'reasoning': response[:300] + "..." if len(response) > 300 else response,
#         'semantic_score': confidence,
#         'technical_accuracy': confidence * 0.9,
#         'ontology_alignment': confidence * 0.8,
#         'suggestions': [],
#         'raw_response': response,
#         'validation_method': 'gemma_heuristic',
#         'positive_indicators': positive_score,
#         'negative_indicators': negative_score
#     }
#
#
# def get_default_gemma_response(self) -> Dict:
#     """Enhanced default response with context"""
#     return {
#         'is_valid': True,
#         'confidence': 0.5,
#         'reasoning': 'Gemma validation unavailable - using neutral assessment',
#         'semantic_score': 0.5,
#         'technical_accuracy': 0.5,
#         'ontology_alignment': 0.5,
#         'suggestions': ['Consider manual review due to validation unavailability'],
#         'raw_response': '',
#         'validation_method': 'default_fallback',
#         'error': True
#     }
#
#
# def rule_based_validation(self, triple: Dict) -> Dict:
#     """Enhanced rule-based validation with domain awareness"""
#     start_time = time.time()
#
#     issues = []
#     score = 1.0
#
#     subject = str(triple.get('subject', '')).strip()
#     relation = str(triple.get('relation', '')).strip()
#     obj = str(triple.get('object', '')).strip()
#     confidence = float(triple.get('confidence', 0))
#
#     # Normalize entities
#     subject = self.normalize_entity(subject)
#     obj = self.normalize_entity(obj)
#
#     # Domain-specific validation
#     domain = self.detect_technology_domain(triple)
#     domain_thresholds = self.get_domain_thresholds(domain)
#
#     # Check relation validity
#     valid_relations = self.ontology.get('relations', [])
#     if relation not in valid_relations:
#         issues.append(f"Unknown relation: {relation}")
#         score -= 0.3
#
#     # Check entity reasonableness with domain context
#     if not self.is_reasonable_entity_enhanced(subject, domain):
#         issues.append(f"Questionable subject for {domain}: {subject}")
#         score -= 0.25
#
#     if not self.is_reasonable_entity_enhanced(obj, domain):
#         issues.append(f"Questionable object for {domain}: {obj}")
#         score -= 0.25
#
#     # Use domain-specific confidence threshold
#     domain_conf_threshold = domain_thresholds.get('confidence_threshold', self.thresholds.confidence_threshold)
#     if confidence < domain_conf_threshold:
#         issues.append(f"Low confidence for {domain}: {confidence} < {domain_conf_threshold}")
#         score -= 0.1
#
#     # Check minimum length
#     if len(subject) < 2 or len(obj) < 2:
#         issues.append("Entity too short")
#         score -= 0.2
#
#     # Domain-specific pattern validation
#     if domain in self.domain_prompts:
#         domain_template = self.domain_prompts[domain]
#
#         # Check against known invalid patterns for this domain
#         triple_text = f"{subject} {relation} {obj}"
#         for invalid_pattern in domain_template.invalid_patterns:
#             if self._semantic_similarity(triple_text.lower(), invalid_pattern.lower()) > 0.7:
#                 issues.append(f"Matches invalid {domain} pattern: {invalid_pattern}")
#                 score -= 0.3
#
#     score = max(0.0, score)
#
#     # Use domain-specific combined threshold for validity
#     domain_combined_threshold = domain_thresholds.get('combined_threshold', 0.6)
#     is_valid = score >= domain_combined_threshold and len(issues) <= 2
#
#     self.validation_performance['rule_based_time'] += time.time() - start_time
#
#     return {
#         'is_valid': is_valid,
#         'score': score,
#         'issues': issues,
#         'normalized_subject': subject,
#         'normalized_object': obj,
#         'domain': domain,
#         'domain_thresholds_used': domain_thresholds
#     }
#
#
# def is_reasonable_entity_enhanced(self, entity: str, domain: str) -> bool:
#     """Enhanced entity validation with domain-specific patterns"""
#
#     if not entity or len(entity) < 2:
#         return False
#
#     # Check base reasonableness
#     if self.is_reasonable_entity(entity):
#         return True
#
#     # Domain-specific checks
#     if domain in self.domain_prompts:
#         domain_template = self.domain_prompts[domain]
#         entity_lower = entity.lower()
#
#         # Check if entity contains domain-specific keywords
#         for keyword in domain_template.context_keywords:
#             if keyword in entity_lower:
#                 return True
#
#     return False
#
#
# def is_reasonable_entity(self, entity: str) -> bool:
#     """Base entity reasonableness check (enhanced from original)"""
#     if not entity or len(entity) < 2:
#         return False
#
#     # Extended programming patterns (from original code enhanced)
#     programming_patterns = [
#         # Core concepts
#         'method', 'function', 'variable', 'parameter', 'argument',
#         'class', 'interface', 'enum', 'annotation', 'package',
#         'exception', 'error', 'handler', 'listener', 'event',
#         'thread', 'process', 'service', 'controller', 'model',
#         'view', 'component', 'module', 'library', 'framework',
#
#         # Web and API (enhanced)
#         'api', 'rest', 'http', 'json', 'xml', 'html', 'css',
#         'request', 'response', 'endpoint', 'route', 'middleware',
#         'client', 'server', 'browser', 'dom', 'element',
#
#         # Data and Database (enhanced)
#         'database', 'sql', 'query', 'table', 'column', 'row',
#         'index', 'schema', 'connection', 'transaction', 'cursor',
#         'nosql', 'mongodb', 'redis', 'elasticsearch',
#
#         # Programming constructs (enhanced)
#         'loop', 'condition', 'statement', 'expression', 'operator',
#         'syntax', 'keyword', 'literal', 'identifier', 'scope',
#         'block', 'closure', 'callback', 'promise', 'async',
#
#         # Object-oriented (enhanced)
#         'inheritance', 'polymorphism', 'encapsulation', 'abstraction',
#         'constructor', 'destructor', 'getter', 'setter', 'accessor',
#         'static', 'final', 'abstract', 'virtual', 'override',
#
#         # Data types (enhanced)
#         'string', 'integer', 'float', 'boolean', 'array', 'list',
#         'map', 'set', 'collection', 'iterator', 'stream', 'queue',
#         'stack', 'tree', 'graph', 'hash', 'dict', 'tuple'
#     ]
#
#     entity_lower = entity.lower()
#
#     # Check for programming patterns
#     for pattern in programming_patterns:
#         if pattern in entity_lower:
#             return True
#
#     # Enhanced identifier format checking
#     clean_entity = entity.replace('_', '').replace('-', '').replace('.', '')
#     if clean_entity.isalnum() and entity[0].isalpha():
#         return True
#
#     # Check for common naming conventions (enhanced)
#     naming_patterns = [
#         entity.endswith('Service'), entity.endswith('Controller'),
#         entity.endswith('Manager'), entity.endswith('Handler'),
#         entity.endswith('Factory'), entity.endswith('Builder'),
#         entity.endswith('Config'), entity.endswith('Utils'),
#         entity.endswith('Helper'), entity.endswith('Provider'),
#         entity.endswith('Adapter'), entity.endswith('Wrapper'),
#         entity.startswith('get'), entity.startswith('set'),
#         entity.startswith('is'), entity.startswith('has'),
#         entity.startswith('create'), entity.startswith('build'),
#         entity.startswith('init'), entity.startswith('load')
#     ]
#
#     if any(naming_patterns):
#         return True
#
#     return False
#
#
# def normalize_entity(self, entity: str) -> str:
#     """Enhanced entity normalization with domain awareness"""
#     if not entity:
#         return ""
#
#     entity = entity.strip()
#
#     # Extended mapping dictionary (enhanced from original)
#     mappings = {
#         # Basic types
#         'string': 'String', 'str': 'String',
#         'integer': 'Integer', 'int': 'Integer',
#         'double': 'Double', 'float': 'Float',
#         'boolean': 'Boolean', 'bool': 'Boolean',
#         'char': 'Character', 'character': 'Character',
#
#         # Collections
#         'arraylist': 'ArrayList', 'array_list': 'ArrayList',
#         'hashmap': 'HashMap', 'hash_map': 'HashMap',
#         'list': 'List', 'set': 'Set', 'map': 'Map',
#         'dict': 'Dictionary', 'dictionary': 'Dictionary',
#         'queue': 'Queue', 'stack': 'Stack',
#
#         # Programming constructs
#         'func': 'Function', 'function': 'Function',
#         'method': 'Method', 'class': 'Class',
#         'interface': 'Interface', 'api': 'API',
#         'obj': 'Object', 'object': 'Object',
#
#         # Web technologies
#         'http': 'HTTP', 'https': 'HTTPS',
#         'json': 'JSON', 'xml': 'XML',
#         'html': 'HTML', 'css': 'CSS',
#         'js': 'JavaScript', 'javascript': 'JavaScript',
#
#         # Database
#         'sql': 'SQL', 'db': 'Database',
#         'database': 'Database', 'table': 'Table',
#         'query': 'Query', 'index': 'Index'
#     }
#
#     entity_lower = entity.lower()
#     if entity_lower in mappings:
#         return mappings[entity_lower]
#
#     return entity.capitalize()
#
#
# def combine_validation_results_enhanced(self, rule_result: Dict, gemma_result: Dict, triple: Dict) -> Dict:
#     """Enhanced result combination with domain-aware thresholds"""
#
#     domain = rule_result.get('domain', 'python')
#     domain_thresholds = self.get_domain_thresholds(domain)
#
#     # Extract scores
#     rule_score = rule_result['score']
#     semantic_score = gemma_result.get('semantic_score', 0.5)
#
#     # Use adaptive weighting based on domain performance
#     rule_weight = self.thresholds.rule_weight
#     semantic_weight = self.thresholds.semantic_weight
#
#     # Adjust weights based on validation method reliability
#     if gemma_result.get('validation_method') == 'offline_pattern_matching':
#         # Lower weight for offline semantic validation
#         rule_weight = 0.6
#         semantic_weight = 0.4
#     elif gemma_result.get('validation_method') == 'default_fallback':
#         # Heavy weight on rule-based when Gemma unavailable
#         rule_weight = 0.8
#         semantic_weight = 0.2
#
#     # Weighted combination
#     final_score = (rule_score * rule_weight) + (semantic_score * semantic_weight)
#
#     # Determine validity with domain-specific logic
#     rule_valid = rule_result['is_valid']
#     gemma_valid = gemma_result.get('is_valid', True)
#
#     # Use domain-specific thresholds
#     semantic_threshold = domain_thresholds.get('semantic_threshold', self.thresholds.semantic_threshold)
#     combined_threshold = domain_thresholds.get('combined_threshold', self.thresholds.combined_threshold)
#
#     # Enhanced validity logic
#     if self.use_gemma and not self.offline_mode:
#         # Both validators available
#         is_valid = (rule_valid and gemma_valid) or (semantic_score > semantic_threshold)
#     elif self.offline_mode:
#         # Offline mode - more conservative
#         is_valid = rule_valid and (semantic_score > 0.6)
#     else:
#         # Rule-based only
#         is_valid = rule_valid
#
#     # Final threshold check
#     if final_score < combined_threshold:
#         is_valid = False
#
#     # Combine issues
#     issues = rule_result['issues'].copy()
#     if not gemma_valid:
#         issues.append(f"Semantic validation failed ({domain}): {gemma_result.get('reasoning', 'Unknown')}")
#
#     return {
#         'is_valid': is_valid,
#         'confidence': final_score,
#         'issues': issues,
#         'gemma_analysis': gemma_result,
#         'rule_based_score': rule_score,
#         'semantic_score': semantic_score,
#         'final_score': final_score,
#         'domain': domain,
#         'thresholds_used': domain_thresholds,
#         'weights_used': {'rule': rule_weight, 'semantic': semantic_weight},
#         'validation_method': gemma_result.get('validation_method', 'unknown')
#     }
#
#
# def load_triples(self) -> List[Dict]:
#     """Load triples with enhanced format support"""
#     try:
#         with open(self.triples_file, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#
#         # Handle multiple file formats
#         triples = []
#
#         if 'triples' in data:
#             # Enhanced ontology extractor format
#             if 'all' in data['triples']:
#                 triples = data['triples']['all']
#             else:
#                 # Combine all triple categories
#                 for category, category_triples in data['triples'].items():
#                     if isinstance(category_triples, list):
#                         triples.extend(category_triples)
#         elif isinstance(data, list):
#             # Simple list format
#             triples = data
#         elif 'metadata' in data and 'triples' in data:
#             # Alternative format
#             triples = data['triples']
#         else:
#             # Try to find triples in any nested structure
#             def find_triples(obj, path=""):
#                 if isinstance(obj, list) and len(obj) > 0:
#                     if isinstance(obj[0], dict) and 'subject' in obj[0]:
#                         return obj
#                 elif isinstance(obj, dict):
#                     for key, value in obj.items():
#                         result = find_triples(value, path + "." + key)
#                         if result:
#                             return result
#                 return None
#
#             found_triples = find_triples(data)
#             if found_triples:
#                 triples = found_triples
#
#         print(f"✅ Loaded {len(triples)} triples for validation")
#
#         # Add source file info to triples if missing
#         for triple in triples:
#             if 'source_file' not in triple:
#                 triple['source_file'] = str(self.triples_file)
#
#         return triples
#
#     except Exception as e:
#         print(f"❌ Error loading triples: {e}")
#         return []
#
#
# def validate_all_enhanced(self) -> Optional[Dict]:
#     """Main enhanced validation with all features"""
#     print("=" * 80)
#     print("🚀 Advanced Gemma-Enhanced Triple Validation")
#     print("=" * 80)
#
#     # Initialize offline fallback
#     if self.enable_offline_mode:
#         self.initialize_offline_fallback()
#
#     # Load components
#     if not self.load_ontology():
#         return None
#
#     # Try to load Gemma model
#     if self.use_gemma:
#         gemma_loaded = self.load_gemma_model()
#         if not gemma_loaded and not self.enable_offline_mode:
#             print("❌ Gemma unavailable and offline mode disabled")
#             return None
#
#     triples = self.load_triples()
#     if not triples:
#         return None
#
#     print(f"🔍 Validating {len(triples)} triples with advanced method...")
#     print(f"📊 Validation mode: {'Online + Offline' if self.use_gemma and not self.offline_mode else 'Offline only'}")
#     print(f"🎯 Adaptive thresholds: {'Enabled' if self.adaptive_thresholds else 'Disabled'}")
#
#     # Process triples
#     start_time = time.time()
#     validated_results = []
#     batch_size = self.validation_batch_size
#
#     for i in range(0, len(triples), batch_size):
#         batch = triples[i:i + batch_size]
#
#         for j, triple in enumerate(batch):
#             try:
#                 # Enhanced validation
#                 rule_result = self.rule_based_validation(triple)
#
#                 # Gemma/offline semantic validation
#                 ontology_context = {
#                     'entities': self.ontology.get('entities', []),
#                     'relations': self.ontology.get('relations', [])
#                 }
#
#                 gemma_result = self.validate_with_gemma_enhanced(triple, ontology_context)
#
#                 # Combine results
#                 combined_result = self.combine_validation_results_enhanced(rule_result, gemma_result, triple)
#
#                 # Update statistics
#                 if combined_result['is_valid']:
#                     self.validation_stats['valid_triples'] += 1
#                 else:
#                     self.validation_stats['invalid_triples'] += 1
#
#                 self.validation_stats['total_processed'] += 1
#
#                 # Create result record
#                 result_record = {
#                     'original_triple': triple,
#                     'validation_result': combined_result,
#                     'performance_metrics': {
#                         'domain': combined_result.get('domain', 'unknown'),
#                         'validation_method': combined_result.get('validation_method', 'unknown'),
#                         'thresholds_used': combined_result.get('thresholds_used', {}),
#                         'weights_used': combined_result.get('weights_used', {})
#                     }
#                 }
#
#                 validated_results.append(result_record)
#
#             except Exception as e:
#                 logger.error(f"Error validating triple {i + j}: {e}")
#                 continue
#
#         # Progress update
#         processed = min(i + batch_size, len(triples))
#         print(f"📈 Progress: {processed}/{len(triples)} triples ({processed / len(triples) * 100:.1f}%)")
#
#         # Periodic cleanup and adaptive threshold update
#         if torch.cuda.is_available() and (i + 1) % (batch_size * 5) == 0:
#             torch.cuda.empty_cache()
#
#         # Update adaptive thresholds every 100 triples
#         if self.adaptive_thresholds and len(validated_results) % 100 == 0 and len(validated_results) > 0:
#             analysis = self.analyze_validation_performance(validated_results[-100:])
#             self.apply_adaptive_thresholds(analysis)
#
#     # Final analysis and threshold optimization
#     if self.adaptive_thresholds and len(validated_results) > 50:
#         print("🔧 Performing final threshold optimization...")
#         final_analysis = self.analyze_validation_performance(validated_results)
#         self.apply_adaptive_thresholds(final_analysis)
#
#     # Create final output
#     total_time = time.time() - start_time
#     self.validation_performance['total_time'] = total_time
#
#     # Save pattern cache for offline use
#     if self.enable_offline_mode:
#         self._save_pattern_cache()
#
#     # Generate final results
#     valid_results = [r for r in validated_results if r['validation_result']['is_valid']]
#     nodes, edges = self.create_enhanced_graph_data([r['original_triple'] for r in valid_results])
#
#     # Comprehensive output
#     output_data = {
#         'metadata': {
#             'created_at': datetime.now().isoformat(),
#             'validation_method': 'advanced_gemma_enhanced',
#             'gemma_model': self.gemma_model_name if not self.offline_mode else f"{self.gemma_model_name} (offline)",
#             'adaptive_thresholds': self.adaptive_thresholds,
#             'offline_mode_enabled': self.enable_offline_mode,
#             'offline_mode_used': self.offline_mode,
#             'ontology_source': str(self.ontology_file),
#             'triples_source': str(self.triples_file),
#             'validation_stats': dict(self.validation_stats),
#             'performance_metrics': self.validation_performance,
#             'final_thresholds': self.thresholds.__dict__,
#             'threshold_history': self.threshold_history,
#             'domain_prompts_used': list(self.domain_prompts.keys()),
#             'total_nodes': len(nodes),
#             'total_edges': len(edges)
#         },
#         'validation_analysis': self.analyze_validation_performance(validated_results) if validated_results else {},
#         'validation_results': {
#             'all_results': validated_results,
#             'valid_results': valid_results,
#             'invalid_results': [r for r in validated_results if not r['validation_result']['is_valid']]
#         },
#         'graph_data': {
#             'nodes': nodes,
#             'edges': edges
#         },
#         'domain_performance': self._analyze_domain_performance(validated_results),
#         'threshold_recommendations': self._generate_final_recommendations(validated_results)
#     }
#
#     return self.save_enhanced_results(output_data)
#
#
# def create_enhanced_graph_data(self, triples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
#     """Create enhanced graph data with performance metrics"""
#     nodes = {}
#     edges = []
#
#     for triple in triples:
#         subject = triple.get('subject', '')
#         obj = triple.get('object', '')
#         relation = triple.get('relation', '')
#         technology = triple.get('technology', 'unknown')
#
#         # Create/update nodes with enhanced metadata
#         for entity in [subject, obj]:
#             if entity not in nodes:
#                 nodes[entity] = {
#                     'name': entity,
#                     'type': 'Entity',
#                     'technologies': set(),
#                     'source_count': 0,
#                     'validation_scores': [],
#                     'domains': set()
#                 }
#
#             nodes[entity]['source_count'] += 1
#             nodes[entity]['technologies'].add(technology)
#
#             # Add domain information
#             domain = self.detect_technology_domain(triple)
#             nodes[entity]['domains'].add(domain)
#
#         # Create enhanced edge
#         edges.append({
#             'from': subject,
#             'to': obj,
#             'relation': relation,
#             'technology': technology,
#             'domain': self.detect_technology_domain(triple),
#             'source_url': triple.get('source_url', ''),
#             'source_type': triple.get('source_type', 'text'),
#             'extraction_method': triple.get('extraction_method', 'unknown')
#         })
#
#     # Convert sets to lists for JSON serialization
#     for node in nodes.values():
#         node['technologies'] = list(node['technologies'])
#         node['domains'] = list(node['domains'])
#         if node['validation_scores']:
#             node['avg_validation_score'] = sum(node['validation_scores']) / len(node['validation_scores'])



"""
Step 5: Validate Nodes and Relations against ontology (Enhanced with Gemma)
"""




























import json
import torch
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

class DataValidator:
    def __init__(self,
                 ontology_file="data/ontology/updated_ontology.json",
                 triples_file="data/extracted_triples/extracted_triples_improved.json",
                 output_dir="data/validated_data",
                 use_gemma=False,
                 gemma_model_id="google/gemma-2b-it"):
        self.ontology_file = Path(ontology_file)
        self.triples_file = Path(triples_file)
        self.output_dir = Path(output_dir)
        self.use_gemma = use_gemma
        self.gemma_model_id = gemma_model_id
        self.ontology = None
        self.validation_stats = defaultdict(int)

        # Gemma components (loaded only if used)
        self.gemma_tokenizer = None
        self.gemma_model = None

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
            all_triples = data['triples'].get('all', []) if 'triples' in data else data if isinstance(data, list) else []
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

    def load_gemma_model(self):
        """Load Gemma model and tokenizer (only if needed)"""
        if not self.use_gemma:
            return False
        try:
            print("🧠 Loading Gemma model for enhanced validation...")
            self.gemma_tokenizer = AutoTokenizer.from_pretrained(self.gemma_model_id)
            self.gemma_model = AutoModelForCausalLM.from_pretrained(
                self.gemma_model_id,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
            print("✅ Gemma model loaded successfully.")
            return True
        except Exception as e:
            print(f"❌ Failed to load Gemma: {e}")
            print("⚠️  Proceeding without Gemma.")
            self.use_gemma = False
            return False

    def is_technical_triple_with_gemma(self, triple, max_length=256):
        """
        Use Gemma to determine if a triple is technical (programming-related).
        Returns: True if technical, False otherwise.
        """
        if not self.use_gemma:
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
            inputs = self.gemma_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(self.gemma_model.device)
            with torch.no_grad():
                outputs = self.gemma_model.generate(
                    **inputs,
                    max_new_tokens=5,
                    temperature=0.1,
                    pad_token_id=self.gemma_tokenizer.eos_token_id
                )
            response = self.gemma_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            return 'yes' in response.lower()
        except Exception as e:
            print(f"⚠️  Gemma inference error: {e}")
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

            # 1. Technical Check via Gemma or fallback
            if self.use_gemma or not self.is_technical_triple_with_gemma(triple):
                is_technical = self.is_technical_triple_with_gemma(triple)
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
        print("=" * 50)
        print("✅ Step 5: Validating Nodes and Relations (with Gemma)")
        print("=" * 50)

        if not self.load_ontology():
            return None
        triples = self.load_triples()
        if not triples:
            print("❌ No triples to validate")
            return None

        # Load Gemma if enabled
        if self.use_gemma:
            if not self.load_gemma_model():
                print("⚠️  Gemma failed to load. Falling back to rule-based filtering.")

        print(f"🔍 Validating {len(triples)} triples...")
        validated_results = []
        batch_size = 10  # Reduce if memory issues

        for i, triple in enumerate(triples):
            result = self.validate_triple(triple)
            validated_results.append(result)

            if (i + 1) % batch_size == 0 and self.use_gemma:
                # Avoid rate/CUDA issues
                time.sleep(0.1)

            if (i + 1) % 100 == 0:
                print(f"📈 Progress: {i + 1}/{len(triples)} triples validated")

        # Deduplicate
        print("🧹 Removing duplicates...")
        unique_valid_results = self.remove_duplicates(validated_results)

        # Create nodes and edges
        print("🏗️ Creating nodes and edges...")
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
                'gemma_used': self.use_gemma
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

            print(f"💾 Validated data saved to: {self.output_dir}")
            return data
        except Exception as e:
            print(f"❌ Error saving validated data: {e}")
            return None


def main():
    print("🎯 Knowledge Graph Data Validator (with Gemma)")
    print("-" * 45)

    # ✅ Set to True to enable Gemma
    validator = DataValidator(use_gemma=True)  # Change to False to skip Gemma

    validated_data = validator.validate_all_data()

    if validated_data:
        stats = validated_data['metadata']['validation_stats']
        print(f"\n📊 VALIDATION SUMMARY")
        print("-" * 25)
        total = stats.get('valid_triples', 0) + stats.get('invalid_triples', 0)
        print(f"Total triples processed: {total}")
        print(f"Valid triples: {stats.get('valid_triples', 0)}")
        print(f"Invalid triples: {stats.get('invalid_triples', 0)}")
        print(f"Non-technical filtered: {stats.get('non_technical_triples', 0)}")
        print(f"Unique valid triples: {stats.get('unique_valid_triples', 0)}")
        print(f"Total nodes: {validated_data['metadata']['total_nodes']}")
        print(f"Total edges: {validated_data['metadata']['total_edges']}")
        print(f"Unknown relations: {stats.get('unknown_relations', 0)}")
        print(f"Low confidence: {stats.get('low_confidence', 0)}")

        success_rate = (stats.get('valid_triples', 0) / max(1, total)) * 100
        print(f"Success rate: {success_rate:.1f}%")
        print(f"\n✅ Ready for Step 6: Neo4j Loading!")
        return validated_data
    else:
        print("❌ Data validation failed!")
        return None


if __name__ == "__main__":
    main()