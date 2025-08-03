"""
Enhanced Gemma LLM Triple Validator
Processes all technology-specific triple files and validates using HuggingFace Gemma
"""

import os
import json
import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class GemmaConfig:
    """Configuration for LLM access via HuggingFace Inference API"""
    provider: str = "huggingface"
    model_name: str = "microsoft/DialoGPT-medium"  # Alternative working model
    api_key: str = ""
    max_tokens: int = 150
    temperature: float = 0.1
    timeout: int = 60


class EnhancedGemmaValidator:
    """Enhanced validator with batch processing for all technologies"""

    def __init__(self,
                 triples_dir="data/extracted_triples",
                 output_file="data/all_validated_triples.json",
                 gemma_config: GemmaConfig = None,
                 rule_weight=0.3,
                 llm_weight=0.7,
                 batch_size=3,
                 delay_between_batches=2.0,
                 max_triples_per_tech=100):

        self.triples_dir = Path(triples_dir)
        self.output_file = Path(output_file)
        self.gemma_config = gemma_config or GemmaConfig()

        # Validation settings
        self.rule_weight = rule_weight
        self.llm_weight = llm_weight
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches
        self.max_triples_per_tech = max_triples_per_tech

        # Stats tracking
        self.validation_stats = {
            'total_processed': 0,
            'total_valid': 0,
            'by_technology': {},
            'api_calls': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }

        print(f"🤖 Enhanced Gemma Validator initialized")
        print(f"   Model: {self.gemma_config.model_name}")
        print(f"   Batch size: {batch_size}")
        print(f"   Max triples per tech: {max_triples_per_tech}")

    async def call_huggingface_api(self, prompt: str) -> str:
        """Call HuggingFace API with error handling and retries"""

        # List of working models to try in order
        working_models = [
            "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-large",
            "EleutherAI/gpt-neo-1.3B",
            "EleutherAI/gpt-neo-2.7B",
            "distilgpt2",
            "gpt2"
        ]

        # Try the configured model first, then fallbacks
        models_to_try = [self.gemma_config.model_name] + [m for m in working_models if m != self.gemma_config.model_name]

        for model_name in models_to_try:
            try:
                url = f"https://api-inference.huggingface.co/models/{model_name}"
                headers = {
                    "Authorization": f"Bearer {self.gemma_config.api_key}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": self.gemma_config.max_tokens,
                        "temperature": self.gemma_config.temperature,
                        "return_full_text": False,
                        "do_sample": True,
                        "top_p": 0.9
                    },
                    "options": {
                        "wait_for_model": True,
                        "use_cache": False
                    }
                }

                max_retries = 2
                for attempt in range(max_retries):
                    try:
                        timeout = aiohttp.ClientTimeout(total=self.gemma_config.timeout)
                        async with aiohttp.ClientSession(timeout=timeout) as session:
                            async with session.post(url, headers=headers, json=payload) as response:
                                self.validation_stats['api_calls'] += 1

                                if response.status == 200:
                                    data = await response.json()

                                    if isinstance(data, list) and len(data) > 0:
                                        result = data[0].get("generated_text", "").strip()
                                        if result:
                                            logger.info(f"Successfully used model: {model_name}")
                                            return result
                                    elif isinstance(data, dict) and "generated_text" in data:
                                        result = data["generated_text"].strip()
                                        if result:
                                            logger.info(f"Successfully used model: {model_name}")
                                            return result

                                elif response.status == 503:  # Model loading
                                    if attempt < max_retries - 1:
                                        wait_time = 10
                                        logger.warning(f"Model {model_name} loading, waiting {wait_time}s...")
                                        await asyncio.sleep(wait_time)
                                        continue
                                    else:
                                        break  # Try next model

                                elif response.status == 404:
                                    logger.warning(f"Model {model_name} not found, trying next model...")
                                    break  # Try next model

                                else:
                                    error_text = await response.text()
                                    logger.warning(f"API error {response.status} for {model_name}: {error_text}")
                                    break  # Try next model

                    except Exception as e:
                        if attempt < max_retries - 1:
                            wait_time = 3
                            logger.warning(f"API call failed for {model_name} (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                            await asyncio.sleep(wait_time)
                        else:
                            logger.warning(f"All attempts failed for {model_name}, trying next model: {e}")
                            break

            except Exception as e:
                logger.warning(f"Failed to try model {model_name}: {e}")
                continue

        # If all models fail, return a fallback response
        self.validation_stats['errors'] += 1
        raise Exception("All available models failed to respond")

    def create_validation_prompt(self, triple: Dict, technology: str) -> str:
        """Create optimized validation prompt for any LLM"""

        subject = triple.get('subject', '').strip()
        relation = triple.get('relation', '').strip()
        obj = triple.get('object', '').strip()
        confidence = triple.get('confidence', 0)

        prompt = f"""Task: Validate this {technology.upper()} knowledge relationship:

Subject: "{subject}"
Relation: "{relation}"
Object: "{obj}"
Technology: {technology.upper()}

Is this a valid {technology.upper()} relationship? Respond with:
VALID: true/false
SCORE: 0.0-1.0  
CONFIDENCE: high/medium/low
REASON: brief explanation

Examples:
VALID: true, SCORE: 0.9, CONFIDENCE: high, REASON: "Property can have value in CSS"
VALID: false, SCORE: 0.1, CONFIDENCE: high, REASON: "Color cannot perform database operations"

Response:"""

        return prompt

    def parse_llm_response(self, response: str) -> Dict:
        """Parse LLM response with robust fallback parsing"""

        try:
            # Try to find structured response first
            response_upper = response.upper()

            # Extract VALID
            valid = False
            if "VALID: TRUE" in response_upper or "VALID:TRUE" in response_upper:
                valid = True
            elif "VALID: FALSE" in response_upper or "VALID:FALSE" in response_upper:
                valid = False
            else:
                # Fallback to keyword analysis
                positive_words = ['valid', 'correct', 'good', 'true', 'yes', 'accurate']
                negative_words = ['invalid', 'incorrect', 'bad', 'false', 'no', 'wrong']

                pos_count = sum(1 for word in positive_words if word in response.lower())
                neg_count = sum(1 for word in negative_words if word in response.lower())
                valid = pos_count > neg_count

            # Extract SCORE
            score = 0.5
            import re
            score_match = re.search(r'SCORE:\s*([0-9.]+)', response_upper)
            if score_match:
                try:
                    score = float(score_match.group(1))
                    score = max(0.0, min(1.0, score))
                except:
                    score = 0.9 if valid else 0.1
            else:
                score = 0.8 if valid else 0.2

            # Extract CONFIDENCE
            confidence = 'medium'
            if "CONFIDENCE: HIGH" in response_upper:
                confidence = 'high'
            elif "CONFIDENCE: LOW" in response_upper:
                confidence = 'low'
            elif any(word in response.lower() for word in ['definitely', 'clearly', 'obviously']):
                confidence = 'high'
            elif any(word in response.lower() for word in ['maybe', 'possibly', 'uncertain']):
                confidence = 'low'

            # Extract REASON
            reason_match = re.search(r'REASON:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
            if reason_match:
                reason = reason_match.group(1).strip()
            else:
                reason = response[:100].strip() + "..." if len(response) > 100 else response.strip()

            return {
                'valid': valid,
                'score': score,
                'confidence': confidence,
                'reason': reason
            }

        except Exception as e:
            logger.debug(f"Structured parsing failed: {e}")
            return self.fallback_parse_response(response)

    def fallback_parse_response(self, response: str) -> Dict:
        """Robust fallback parser for non-JSON responses"""

        response_lower = response.lower()

        # Determine validity
        positive_indicators = ['valid', 'correct', 'good', 'true', 'yes', 'accurate', 'makes sense', 'logical']
        negative_indicators = ['invalid', 'incorrect', 'bad', 'false', 'no', 'wrong', 'nonsensical', 'illogical']

        positive_score = sum(2 if indicator in response_lower else 0 for indicator in positive_indicators)
        negative_score = sum(2 if indicator in response_lower else 0 for indicator in negative_indicators)

        # Check for explicit boolean values
        if 'true' in response_lower and 'false' not in response_lower:
            positive_score += 3
        elif 'false' in response_lower and 'true' not in response_lower:
            negative_score += 3

        is_valid = positive_score > negative_score

        # Determine confidence level
        high_conf_words = ['definitely', 'clearly', 'obviously', 'certainly', 'strong', 'high']
        low_conf_words = ['maybe', 'possibly', 'might', 'uncertain', 'low', 'weak']

        if any(word in response_lower for word in high_conf_words):
            confidence = 'high'
            score = 0.9 if is_valid else 0.1
        elif any(word in response_lower for word in low_conf_words):
            confidence = 'low'
            score = 0.6 if is_valid else 0.4
        else:
            confidence = 'medium'
            score = 0.75 if is_valid else 0.25

        # Extract reason (first 100 chars)
        reason = response[:100].strip()
        if len(response) > 100:
            reason += "..."

        return {
            'valid': is_valid,
            'score': score,
            'confidence': confidence,
            'reason': reason
        }

    def rule_based_validation(self, triple: Dict, technology: str) -> Dict:
        """Enhanced rule-based validation"""

        try:
            subject = triple.get('subject', '').strip()
            relation = triple.get('relation', '').strip()
            obj = triple.get('object', '').strip()
            confidence = float(triple.get('confidence', 0))

            score = 0.0
            issues = []

            # Length checks
            if len(subject) >= 2 and subject.isalpha():
                score += 0.25
            else:
                issues.append("Invalid subject")

            if len(relation) >= 2:
                score += 0.25
            else:
                issues.append("Invalid relation")

            if len(obj) >= 2 and obj.isalpha():
                score += 0.25
            else:
                issues.append("Invalid object")

            # Confidence bonus
            if confidence >= 0.8:
                score += 0.15
            elif confidence >= 0.6:
                score += 0.1
            elif confidence >= 0.4:
                score += 0.05
            else:
                issues.append("Low confidence")

            # Technology-specific rules
            tech_bonus = self.get_technology_bonus(subject, relation, obj, technology)
            score += tech_bonus

            final_score = max(0.0, min(1.0, score))

            return {
                'rule_score': final_score,
                'rule_issues': issues,
                'rule_valid': final_score >= 0.5,
                'tech_bonus': tech_bonus
            }

        except Exception as e:
            return {
                'rule_score': 0.0,
                'rule_issues': [f"Rule validation error: {str(e)}"],
                'rule_valid': False,
                'tech_bonus': 0.0
            }

    def get_technology_bonus(self, subject: str, relation: str, obj: str, technology: str) -> float:
        """Technology-specific validation bonus"""

        tech_keywords = {
            'css': ['style', 'property', 'selector', 'rule', 'color', 'font', 'layout', 'margin', 'padding'],
            'html': ['element', 'tag', 'attribute', 'document', 'node', 'content', 'markup'],
            'javascript': ['function', 'variable', 'object', 'method', 'event', 'callback', 'promise'],
            'python': ['class', 'method', 'module', 'import', 'function', 'variable', 'object'],
            'java': ['class', 'method', 'object', 'interface', 'package', 'inheritance']
        }

        keywords = tech_keywords.get(technology.lower(), [])

        bonus = 0.0
        text_to_check = f"{subject} {relation} {obj}".lower()

        for keyword in keywords:
            if keyword in text_to_check:
                bonus += 0.02

        return min(bonus, 0.1)  # Cap at 0.1

    async def validate_triple_with_llm(self, triple: Dict, technology: str) -> Dict:
        """Validate single triple using LLM"""

        try:
            # Create prompt
            prompt = self.create_validation_prompt(triple, technology)

            # Get LLM response
            response = await self.call_huggingface_api(prompt)

            # Parse response
            llm_result = self.parse_llm_response(response)

            return {
                'llm_score': llm_result.get('score', 0.5),
                'llm_confidence': llm_result.get('confidence', 'medium'),
                'llm_reasoning': llm_result.get('reason', 'No reasoning provided'),
                'llm_valid': llm_result.get('valid', False),
                'raw_response': response[:200]  # Truncate for storage
            }

        except Exception as e:
            logger.error(f"LLM validation error for {technology}: {e}")
            return {
                'llm_score': 0.5,
                'llm_confidence': 'low',
                'llm_reasoning': f'Validation failed: {str(e)}',
                'llm_valid': False,
                'raw_response': ''
            }

    async def validate_triples_batch(self, triples: List[Dict], technology: str) -> List[Dict]:
        """Validate batch of triples with progress tracking"""

        results = []
        total_batches = (len(triples) - 1) // self.batch_size + 1

        print(f"   Processing {len(triples)} triples in {total_batches} batches...")

        for i in range(0, len(triples), self.batch_size):
            batch = triples[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1

            print(f"   Batch {batch_num}/{total_batches} ({len(batch)} triples)")

            # Process batch
            batch_results = []
            for j, triple in enumerate(batch):
                try:
                    # Rule-based validation
                    rule_result = self.rule_based_validation(triple, technology)

                    # Gemma validation
                    llm_result = await self.validate_triple_with_llm(triple, technology)

                    # Combined scoring
                    combined_score = (
                        self.rule_weight * rule_result['rule_score'] +
                        self.llm_weight * llm_result['llm_score']
                    )

                    # Create final result
                    final_result = {
                        **triple,
                        'technology': technology,
                        'validation': {
                            'rule_based': rule_result,
                            'llm': llm_result,
                            'combined_score': combined_score,
                            'is_valid': combined_score >= 0.5,
                            'validation_method': 'llm_hybrid',
                            'validated_at': datetime.now().isoformat()
                        }
                    }

                    batch_results.append(final_result)
                    self.validation_stats['total_processed'] += 1

                    if final_result['validation']['is_valid']:
                        self.validation_stats['total_valid'] += 1

                except Exception as e:
                    logger.error(f"Error validating triple in {technology}: {e}")
                    self.validation_stats['errors'] += 1

            results.extend(batch_results)

            # Rate limiting delay
            if batch_num < total_batches:
                await asyncio.sleep(self.delay_between_batches)

        return results

    def discover_technology_files(self) -> List[str]:
        """Discover all technology-specific triple files"""

        technologies = []

        if not self.triples_dir.exists():
            logger.error(f"Triples directory not found: {self.triples_dir}")
            return technologies

        # Look for files matching the pattern
        patterns = [
            "*triples*.json",
            "*_css.json",
            "*_html.json",
            "*_javascript.json",
            "*_python.json",
            "*_java.json"
        ]

        found_files = []
        for pattern in patterns:
            found_files.extend(self.triples_dir.glob(pattern))

        # Extract technology names
        for file_path in found_files:
            filename = file_path.stem

            # Try different extraction methods
            if "triples_" in filename:
                tech = filename.split("triples_")[-1]
            elif filename.endswith(("_css", "_html", "_javascript", "_python", "_java")):
                tech = filename.split("_")[-1]
            else:
                tech = filename

            if tech and tech not in technologies:
                technologies.append(tech)

        print(f"📁 Found {len(technologies)} technology files: {technologies}")
        return technologies

    def load_triples_for_technology(self, technology: str) -> List[Dict]:
        """Load triples for a specific technology"""

        # Try different file naming patterns
        possible_filenames = [
            f"ontology_enhanced_triples_{technology}.json",
            f"extracted_triples_{technology}.json",
            f"triples_{technology}.json",
            f"{technology}_triples.json",
            f"{technology}.json"
        ]

        for filename in possible_filenames:
            file_path = self.triples_dir / filename

            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Extract triples from different data structures
                    triples = []

                    if isinstance(data, list):
                        triples = data
                    elif isinstance(data, dict):
                        # Try different keys
                        if 'triples' in data:
                            triples_data = data['triples']
                            if isinstance(triples_data, dict):
                                # Check for nested structure
                                if 'all' in triples_data:
                                    triples = triples_data['all']
                                elif 'high_confidence' in triples_data:
                                    triples = triples_data['high_confidence']
                                else:
                                    # Combine all lists in triples
                                    for key, value in triples_data.items():
                                        if isinstance(value, list):
                                            triples.extend(value)
                            elif isinstance(triples_data, list):
                                triples = triples_data
                        elif 'results' in data:
                            triples = data['results']
                        elif 'data' in data:
                            triples = data['data']

                    # Limit triples per technology
                    if len(triples) > self.max_triples_per_tech:
                        triples = triples[:self.max_triples_per_tech]
                        print(f"   Limited to {self.max_triples_per_tech} triples for {technology}")

                    print(f"   Loaded {len(triples)} triples from {filename}")
                    return triples

                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
                    continue

        logger.warning(f"No valid triple file found for {technology}")
        return []

    async def validate_all_technologies(self) -> Dict:
        """Main validation function for all technologies"""

        print("🤖 ENHANCED GEMMA VALIDATION")
        print("=" * 40)

        self.validation_stats['start_time'] = datetime.now()

        # Discover technologies
        technologies = self.discover_technology_files()

        if not technologies:
            logger.error("No technology files found!")
            return {}

        print(f"🎯 Processing {len(technologies)} technologies")
        print(f"   Model: {self.gemma_config.model_name}")
        print(f"   Batch size: {self.batch_size}")

        all_validated_triples = []

        for i, technology in enumerate(technologies, 1):
            print(f"\n📊 Processing {technology.upper()} ({i}/{len(technologies)})")

            # Load triples
            triples = self.load_triples_for_technology(technology)

            if not triples:
                print(f"   ⚠️ No triples found for {technology}")
                continue

            # Initialize technology stats
            self.validation_stats['by_technology'][technology] = {
                'total': len(triples),
                'valid': 0,
                'invalid': 0,
                'avg_score': 0,
                'processing_time': 0
            }

            # Validate triples
            tech_start_time = time.time()

            try:
                validated_triples = await self.validate_triples_batch(triples, technology)
                all_validated_triples.extend(validated_triples)

                # Update technology stats
                valid_count = len([t for t in validated_triples if t['validation']['is_valid']])
                avg_score = sum(t['validation']['combined_score'] for t in validated_triples) / len(validated_triples)

                tech_stats = self.validation_stats['by_technology'][technology]
                tech_stats['valid'] = valid_count
                tech_stats['invalid'] = len(validated_triples) - valid_count
                tech_stats['avg_score'] = avg_score
                tech_stats['processing_time'] = time.time() - tech_start_time

                print(f"   ✅ {valid_count}/{len(validated_triples)} valid ({valid_count/len(validated_triples)*100:.1f}%)")
                print(f"   📊 Average score: {avg_score:.3f}")

            except Exception as e:
                logger.error(f"Error processing {technology}: {e}")
                self.validation_stats['errors'] += 1

        self.validation_stats['end_time'] = datetime.now()

        # Create final output
        final_data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_technologies': len(technologies),
                'total_triples_processed': self.validation_stats['total_processed'],
                'total_valid_triples': self.validation_stats['total_valid'],
                'overall_validation_rate': self.validation_stats['total_valid'] / max(1, self.validation_stats['total_processed']),
                'llm_config': {
                    'model_name': self.gemma_config.model_name,
                    'provider': self.gemma_config.provider,
                    'max_tokens': self.gemma_config.max_tokens,
                    'temperature': self.gemma_config.temperature
                },
                'validation_settings': {
                    'rule_weight': self.rule_weight,
                    'llm_weight': self.llm_weight,
                    'batch_size': self.batch_size,
                    'max_triples_per_tech': self.max_triples_per_tech
                },
                'processing_stats': self.validation_stats
            },
            'validated_triples': all_validated_triples,
            'valid_triples_only': [t for t in all_validated_triples if t['validation']['is_valid']],
            'by_technology': {}
        }

        # Group by technology
        for triple in all_validated_triples:
            tech = triple['technology']
            if tech not in final_data['by_technology']:
                final_data['by_technology'][tech] = []
            final_data['by_technology'][tech].append(triple)

        # Save results
        await self.save_results(final_data)

        # Print summary
        self.print_validation_summary(final_data)

        return final_data

    async def save_results(self, data: Dict):
        """Save validation results to file"""

        try:
            # Create output directory
            self.output_file.parent.mkdir(parents=True, exist_ok=True)

            # Save main results
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Save valid-only file
            valid_only_file = self.output_file.parent / "valid_triples_only.json"
            with open(valid_only_file, 'w', encoding='utf-8') as f:
                json.dump(data['valid_triples_only'], f, indent=2, ensure_ascii=False)

            print(f"\n💾 Results saved:")
            print(f"   📄 All results: {self.output_file}")
            print(f"   ✅ Valid only: {valid_only_file}")

        except Exception as e:
            logger.error(f"Error saving results: {e}")

    def print_validation_summary(self, data: Dict):
        """Print comprehensive validation summary"""

        metadata = data['metadata']
        tech_stats = metadata['processing_stats']['by_technology']

        print(f"\n🎯 VALIDATION SUMMARY")
        print("=" * 30)
        print(f"Total technologies: {metadata['total_technologies']}")
        print(f"Total triples processed: {metadata['total_triples_processed']}")
        print(f"Total valid triples: {metadata['total_valid_triples']}")
        print(f"Overall validation rate: {metadata['overall_validation_rate']*100:.1f}%")
        print(f"API calls made: {metadata['processing_stats']['api_calls']}")
        print(f"Errors encountered: {metadata['processing_stats']['errors']}")

        print(f"\n📊 BY TECHNOLOGY:")
        print("-" * 20)

        for tech, stats in tech_stats.items():
            if stats['total'] > 0:
                rate = stats['valid'] / stats['total'] * 100
                print(f"{tech.upper():>12}: {stats['valid']:>3}/{stats['total']:<3} ({rate:>5.1f}%) | Score: {stats['avg_score']:.3f}")

        total_time = (self.validation_stats['end_time'] - self.validation_stats['start_time']).total_seconds()
        print(f"\n⏱️ Total processing time: {total_time:.1f} seconds")


# Setup and run validation
async def main():
    """Main execution function"""

    print("🤖 Enhanced Gemma Triple Validator")
    print("=" * 40)

    # Configuration
    api_key = os.getenv("HUGGINGFACE_API_KEY", "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")

    if api_key == "your-huggingface-api-key":
        print("⚠️ Please set your HuggingFace API key!")
        print("   Export HUGGINGFACE_API_KEY=your_token")
        print("   Get token from: https://huggingface.co/settings/tokens")
        return

    # Setup LLM config
    llm_config = GemmaConfig(
        provider="huggingface",
        model_name="microsoft/DialoGPT-medium",  # Working alternative
        api_key=api_key,
        max_tokens=150,
        temperature=0.1,
        timeout=60
    )

    # Initialize validator
    validator = EnhancedGemmaValidator(
        triples_dir="data/extracted_triples",  # Adjust path as needed
        output_file="data/all_validated_triples.json",
        gemma_config=llm_config,
        rule_weight=0.3,
        llm_weight=0.7,
        batch_size=3,  # Small batches to avoid rate limits
        delay_between_batches=2.0,  # 2 second delay
        max_triples_per_tech=100  # Limit for testing
    )

    # Run validation
    try:
        results = await validator.validate_all_technologies()

        if results:
            print("\n🎉 Validation completed successfully!")
            return results
        else:
            print("\n❌ Validation failed!")
            return None

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return None


if __name__ == "__main__":
    # Set your HuggingFace API key
    # os.environ["HUGGINGFACE_API_KEY"] = "your_actual_token_here"

    # Run validation
    results = asyncio.run(main())