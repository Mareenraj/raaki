"""
LLM-Based Technical Triple Validator
Validates extracted triples using various LLM approaches
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Set
import logging
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMTripleValidator:
    def __init__(self,
                 triples_file="data/extracted_triples/extracted_triples_improved.json",
                 ontology_file="data/ontology/generalized_ontology.json",
                 output_dir="data/validated_data",
                 validation_method="gemini",  # gemini, openai, local, hybrid
                 api_key=None,
                 batch_size=50,
                 max_retries=3):
        
        self.triples_file = Path(triples_file)
        self.ontology_file = Path(ontology_file)
        self.output_dir = Path(output_dir)
        self.validation_method = validation_method
        self.api_key = api_key or os.getenv('GEMINI_API_KEY') or os.getenv('OPENAI_API_KEY')
        self.batch_size = batch_size
        self.max_retries = max_retries
        
        # Statistics tracking
        self.validation_stats = defaultdict(int)
        self.validation_results = {
            'valid_triples': [],
            'invalid_triples': [],
            'uncertain_triples': [],
            'validation_errors': []
        }
        
        # Load ontology and triples
        self.ontology = self.load_ontology()
        self.triples = self.load_triples()
        
    def load_ontology(self) -> Dict:
        """Load ontology for validation context"""
        try:
            with open(self.ontology_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'ontology' in data:
                ontology = data['ontology']
            else:
                ontology = data
                
            logger.info(f"Loaded ontology with {len(ontology.get('entities', []))} entities")
            return ontology
            
        except Exception as e:
            logger.error(f"Error loading ontology: {e}")
            return {'entities': [], 'relations': []}
    
    def load_triples(self) -> List[Dict]:
        """Load triples for validation"""
        try:
            with open(self.triples_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract all triples
            if 'triples' in data:
                all_triples = []
                for confidence_level in ['high_confidence', 'medium_confidence', 'low_confidence']:
                    if confidence_level in data['triples']:
                        all_triples.extend(data['triples'][confidence_level])
            else:
                all_triples = data if isinstance(data, list) else []
            
            logger.info(f"Loaded {len(all_triples)} triples for validation")
            return all_triples
            
        except Exception as e:
            logger.error(f"Error loading triples: {e}")
            return []
    
    def create_validation_prompt(self, triples_batch: List[Dict]) -> str:
        """Create comprehensive validation prompt for LLM"""
        
        # Get ontology context
        entities = self.ontology.get('entities', [])
        relations = self.ontology.get('relations', [])
        
        prompt = f"""You are a technical knowledge graph validator. Your task is to validate if the following triples are technically accurate and meaningful for a programming/technology knowledge graph.

ONTOLOGY CONTEXT:
- Valid entities: {', '.join(entities[:50])}... (and {len(entities)-50} more)
- Valid relations: {', '.join(relations[:30])}... (and {len(relations)-30} more)

VALIDATION CRITERIA:
1. TECHNICAL ACCURACY: The triple should represent a true technical relationship
2. MEANINGFUL RELATIONSHIP: The relationship should be meaningful for programming/technology
3. ENTITY VALIDITY: Both subject and object should be valid technical concepts
4. RELATION VALIDITY: The relation should be appropriate for the entities
5. CONTEXT RELEVANCE: The triple should be relevant to programming/technology

VALIDATION RULES:
- Accept: Technical concepts, programming constructs, data types, methods, classes, etc.
- Reject: Generic sentences, non-technical relationships, vague concepts
- Uncertain: When you're not sure about technical accuracy

For each triple, respond with:
- VALID: If the triple is technically accurate and meaningful
- INVALID: If the triple is incorrect, non-technical, or meaningless
- UNCERTAIN: If you're not sure about the technical accuracy

TRIPLES TO VALIDATE:
"""
        
        for i, triple in enumerate(triples_batch, 1):
            subject = triple.get('subject', '')
            relation = triple.get('relation', '')
            object_entity = triple.get('object', '')
            source_text = triple.get('source_text', '')[:200]  # Truncate for prompt
            
            prompt += f"""
{i}. Subject: {subject}
   Relation: {relation}
   Object: {object_entity}
   Context: {source_text}
   Response: """
        
        prompt += "\n\nProvide your validation responses (VALID/INVALID/UNCERTAIN) for each triple:"
        
        return prompt
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def call_gemini_api(self, prompt: str) -> str:
        """Call Gemini API for validation"""
        try:
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 2048,
                    "topP": 0.8,
                    "topK": 40
                }
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text']
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def call_openai_api(self, prompt: str) -> str:
        """Call OpenAI API for validation"""
        try:
            url = "https://api.openai.com/v1/chat/completions"
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a technical knowledge graph validator. Provide clear VALID/INVALID/UNCERTAIN responses."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 2048
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def parse_llm_response(self, response: str, triples_batch: List[Dict]) -> List[str]:
        """Parse LLM response to extract validation results"""
        try:
            # Extract validation responses
            lines = response.strip().split('\n')
            validations = []
            
            for line in lines:
                line = line.strip().lower()
                if 'valid' in line:
                    validations.append('VALID')
                elif 'invalid' in line:
                    validations.append('INVALID')
                elif 'uncertain' in line:
                    validations.append('UNCERTAIN')
            
            # Ensure we have the right number of responses
            if len(validations) != len(triples_batch):
                logger.warning(f"Expected {len(triples_batch)} validations, got {len(validations)}")
                # Pad with UNCERTAIN if needed
                while len(validations) < len(triples_batch):
                    validations.append('UNCERTAIN')
                # Truncate if too many
                validations = validations[:len(triples_batch)]
            
            return validations
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return ['UNCERTAIN'] * len(triples_batch)
    
    def validate_triple_batch(self, triples_batch: List[Dict]) -> List[Dict]:
        """Validate a batch of triples using LLM"""
        try:
            prompt = self.create_validation_prompt(triples_batch)
            
            if self.validation_method == "gemini":
                response = self.call_gemini_api(prompt)
            elif self.validation_method == "openai":
                response = self.call_openai_api(prompt)
            else:
                # Fallback to basic validation
                return self.basic_validation(triples_batch)
            
            validations = self.parse_llm_response(response, triples_batch)
            
            # Apply validations to triples
            validated_triples = []
            for triple, validation in zip(triples_batch, validations):
                triple['llm_validation'] = validation
                triple['validation_method'] = self.validation_method
                triple['validation_timestamp'] = datetime.now().isoformat()
                
                if validation == 'VALID':
                    validated_triples.append(triple)
                    self.validation_stats['valid'] += 1
                elif validation == 'INVALID':
                    self.validation_stats['invalid'] += 1
                else:  # UNCERTAIN
                    self.validation_stats['uncertain'] += 1
            
            return validated_triples
            
        except Exception as e:
            logger.error(f"Error validating batch: {e}")
            return self.basic_validation(triples_batch)
    
    def basic_validation(self, triples_batch: List[Dict]) -> List[Dict]:
        """Basic validation without LLM (fallback)"""
        validated_triples = []
        
        for triple in triples_batch:
            subject = triple.get('subject', '')
            relation = triple.get('relation', '')
            object_entity = triple.get('object', '')
            
            # Basic technical validation
            is_technical = (
                len(subject) > 1 and len(object_entity) > 1 and
                relation in self.ontology.get('relations', []) and
                not any(word in subject.lower() for word in ['the', 'a', 'an', 'is', 'are', 'was', 'were']) and
                not any(word in object_entity.lower() for word in ['the', 'a', 'an', 'is', 'are', 'was', 'were'])
            )
            
            if is_technical:
                triple['llm_validation'] = 'VALID'
                triple['validation_method'] = 'basic'
                triple['validation_timestamp'] = datetime.now().isoformat()
                validated_triples.append(triple)
                self.validation_stats['valid'] += 1
            else:
                self.validation_stats['invalid'] += 1
        
        return validated_triples
    
    def validate_all_triples(self) -> Dict:
        """Validate all triples using LLM"""
        logger.info(f"Starting LLM validation of {len(self.triples)} triples using {self.validation_method}")
        
        all_validated_triples = []
        
        # Process in batches
        for i in range(0, len(self.triples), self.batch_size):
            batch = self.triples[i:i + self.batch_size]
            logger.info(f"Validating batch {i//self.batch_size + 1}/{(len(self.triples) + self.batch_size - 1)//self.batch_size}")
            
            validated_batch = self.validate_triple_batch(batch)
            all_validated_triples.extend(validated_batch)
            
            # Add delay to avoid rate limits
            time.sleep(1)
        
        # Create validation results
        validation_results = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'validation_method': self.validation_method,
                'total_triples': len(self.triples),
                'validated_triples': len(all_validated_triples),
                'validation_stats': dict(self.validation_stats),
                'ontology_used': str(self.ontology_file)
            },
            'validated_triples': all_validated_triples
        }
        
        logger.info(f"Validation complete: {len(all_validated_triples)} valid triples out of {len(self.triples)} total")
        return validation_results
    
    def save_validation_results(self, results: Dict) -> str:
        """Save validation results to file"""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"llm_validated_triples_{self.validation_method}_{timestamp}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Validation results saved to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving validation results: {e}")
            return None
    
    def generate_validation_report(self, results: Dict) -> str:
        """Generate detailed validation report"""
        stats = results['metadata']['validation_stats']
        total = results['metadata']['total_triples']
        valid = results['metadata']['validated_triples']
        
        report = f"""
LLM TRIPLE VALIDATION REPORT
============================
Validation Method: {results['metadata']['validation_method']}
Total Triples: {total}
Valid Triples: {valid}
Invalid Triples: {stats.get('invalid', 0)}
Uncertain Triples: {stats.get('uncertain', 0)}

Validation Rate: {(valid/total*100):.1f}%
Invalid Rate: {(stats.get('invalid', 0)/total*100):.1f}%
Uncertain Rate: {(stats.get('uncertain', 0)/total*100):.1f}%

Top Relations in Valid Triples:
"""
        
        # Analyze relations in valid triples
        relations = Counter()
        for triple in results['validated_triples']:
            relations[triple.get('relation', '')] += 1
        
        for relation, count in relations.most_common(10):
            report += f"  {relation}: {count}\n"
        
        return report

def main():
    """Main validation function"""
    print("🤖 LLM-Based Technical Triple Validator")
    print("=" * 50)
    
    # Configuration
    validator = LLMTripleValidator(
        triples_file="data/extracted_triples/extracted_triples_improved.json",
        ontology_file="data/ontology/generalized_ontology.json",
        validation_method="gemini",  # or "openai", "basic"
        batch_size=20  # Smaller batches for better LLM performance
    )
    
    # Run validation
    results = validator.validate_all_triples()
    
    # Save results
    filepath = validator.save_validation_results(results)
    
    # Generate report
    report = validator.generate_validation_report(results)
    print(report)
    
    if filepath:
        print(f"✅ Validation complete! Results saved to: {filepath}")
    else:
        print("❌ Validation failed!")

if __name__ == "__main__":
    main() 