#!/usr/bin/env python3
"""
Simple LLM Validation Runner
Run LLM-based validation of technical triples with different configurations
"""

import os
import sys
from pathlib import Path
from scripts.llm_triple_validator import LLMTripleValidator

def main():
    """Run LLM validation with user-selected options"""
    
    print("🤖 LLM Technical Triple Validator")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv('GEMINI_API_KEY') or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ No API key found! Set GEMINI_API_KEY or OPENAI_API_KEY environment variable")
        print("💡 Get API keys from:")
        print("   - Gemini: https://makersuite.google.com/app/apikey")
        print("   - OpenAI: https://platform.openai.com/api-keys")
        return
    
    # Configuration options
    print("\n📋 Validation Options:")
    print("1. Gemini API (Recommended)")
    print("2. OpenAI API")
    print("3. Basic validation (no LLM)")
    
    choice = input("\nSelect validation method (1-3): ").strip()
    
    if choice == "1":
        validation_method = "gemini"
        print("✅ Using Gemini API")
    elif choice == "2":
        validation_method = "openai"
        print("✅ Using OpenAI API")
    elif choice == "3":
        validation_method = "basic"
        print("✅ Using basic validation")
    else:
        print("❌ Invalid choice, using Gemini")
        validation_method = "gemini"
    
    # Batch size selection
    print("\n📦 Batch Size Options:")
    print("1. Small (10 triples) - More accurate, slower")
    print("2. Medium (20 triples) - Balanced")
    print("3. Large (50 triples) - Faster, less accurate")
    
    batch_choice = input("Select batch size (1-3): ").strip()
    
    if batch_choice == "1":
        batch_size = 10
    elif batch_choice == "2":
        batch_size = 20
    elif batch_choice == "3":
        batch_size = 50
    else:
        batch_size = 20
        print("✅ Using medium batch size (20)")
    
    # File selection
    triples_files = [
        "data/extracted_triples/extracted_triples_improved.json",
        "data/extracted_triples/generalized_triples_multi_topic.json",
        "data/extracted_triples/improved_triples_multi_topic.json"
    ]
    
    print("\n📁 Available triple files:")
    for i, file in enumerate(triples_files, 1):
        if Path(file).exists():
            print(f"{i}. {file}")
        else:
            print(f"{i}. {file} (not found)")
    
    file_choice = input("Select triples file (1-3): ").strip()
    
    try:
        file_index = int(file_choice) - 1
        if 0 <= file_index < len(triples_files):
            triples_file = triples_files[file_index]
        else:
            triples_file = triples_files[0]
    except ValueError:
        triples_file = triples_files[0]
    
    if not Path(triples_file).exists():
        print(f"❌ File not found: {triples_file}")
        return
    
    print(f"✅ Using file: {triples_file}")
    
    # Create validator
    print(f"\n🚀 Starting validation...")
    print(f"   Method: {validation_method}")
    print(f"   Batch size: {batch_size}")
    print(f"   File: {triples_file}")
    
    validator = LLMTripleValidator(
        triples_file=triples_file,
        validation_method=validation_method,
        batch_size=batch_size,
        api_key=api_key
    )
    
    try:
        # Run validation
        results = validator.validate_all_triples()
        
        # Save results
        filepath = validator.save_validation_results(results)
        
        # Generate and display report
        report = validator.generate_validation_report(results)
        print("\n" + "="*50)
        print("📊 VALIDATION RESULTS")
        print("="*50)
        print(report)
        
        if filepath:
            print(f"\n✅ Validation complete!")
            print(f"📁 Results saved to: {filepath}")
            
            # Show sample of validated triples
            valid_triples = results['validated_triples']
            if valid_triples:
                print(f"\n📋 Sample Valid Triples:")
                for i, triple in enumerate(valid_triples[:5], 1):
                    print(f"{i}. {triple.get('subject', '')} --{triple.get('relation', '')}--> {triple.get('object', '')}")
                if len(valid_triples) > 5:
                    print(f"   ... and {len(valid_triples) - 5} more")
        else:
            print("❌ Failed to save results")
            
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        print("💡 Try:")
        print("   - Check your API key")
        print("   - Reduce batch size")
        print("   - Check internet connection")

if __name__ == "__main__":
    main() 