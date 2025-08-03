# LLM Validation Quick Start Guide

## 🚀 Quick Start

### 1. **Setup API Keys**
```bash
# Set your API key (choose one)
export GEMINI_API_KEY="your_gemini_api_key_here"
# OR
export OPENAI_API_KEY="your_openai_api_key_here"
```

### 2. **Install Dependencies**
```bash
pip install -r requirements_llm_validation.txt
```

### 3. **Run Validation**
```bash
# Interactive mode
python scripts/run_llm_validation.py

# Direct mode
python scripts/llm_triple_validator.py
```

## 📋 Validation Options

### **Method 1: Interactive Runner** (Recommended)
```bash
python scripts/run_llm_validation.py
```
- Choose validation method (Gemini/OpenAI/Basic)
- Select batch size
- Pick triples file
- Get detailed results

### **Method 2: Direct Script**
```bash
python scripts/llm_triple_validator.py
```
- Uses default settings
- Gemini API with batch size 20
- Processes `extracted_triples_improved.json`

### **Method 3: Custom Configuration**
```python
from scripts.llm_triple_validator import LLMTripleValidator

validator = LLMTripleValidator(
    triples_file="data/extracted_triples/extracted_triples_improved.json",
    validation_method="gemini",  # or "openai", "basic"
    batch_size=30,
    api_key="your_api_key"
)

results = validator.validate_all_triples()
```

## 🔧 Configuration Options

### **Validation Methods**
- **`gemini`**: Google Gemini API (recommended)
- **`openai`**: OpenAI GPT API
- **`basic`**: Rule-based validation (no API calls)

### **Batch Sizes**
- **Small (10)**: More accurate, slower, higher cost
- **Medium (20)**: Balanced approach
- **Large (50)**: Faster, less accurate, lower cost

### **Input Files**
- `extracted_triples_improved.json` (1.6MB, 1190 triples)
- `generalized_triples_multi_topic.json` (3.2MB)
- `improved_triples_multi_topic.json` (326B)

## 📊 Expected Results

### **Sample Output**
```
🤖 LLM-Based Technical Triple Validator
==================================================
Starting LLM validation of 1190 triples using gemini
Validating batch 1/60
Validating batch 2/60
...
Validation complete: 856 valid triples out of 1190 total

LLM TRIPLE VALIDATION REPORT
============================
Validation Method: gemini
Total Triples: 1190
Valid Triples: 856
Invalid Triples: 234
Uncertain Triples: 100

Validation Rate: 71.9%
Invalid Rate: 19.7%
Uncertain Rate: 8.4%

Top Relations in Valid Triples:
  istypeof: 245
  hasreturntype: 198
  hasmethod: 89
  belongsto: 67
  ...
```

## 💰 Cost Estimation

### **Gemini API**
- ~$0.001 per 1K tokens
- Average: $2-5 for 1000 triples
- Batch size 20: ~$3-8 total

### **OpenAI API**
- ~$0.003 per 1K tokens
- Average: $5-15 for 1000 triples
- Batch size 20: ~$8-20 total

## 🛠️ Troubleshooting

### **API Key Issues**
```bash
# Check if API key is set
echo $GEMINI_API_KEY
echo $OPENAI_API_KEY

# Set API key if missing
export GEMINI_API_KEY="your_key_here"
```

### **Rate Limiting**
- Reduce batch size to 10-15
- Add delays between batches
- Use basic validation as fallback

### **Memory Issues**
- Reduce batch size
- Process smaller files
- Use basic validation

### **Network Issues**
- Check internet connection
- Try different API endpoints
- Use basic validation as fallback

## 📈 Performance Tips

### **For Speed**
- Use larger batch sizes (30-50)
- Use Gemini API (faster than OpenAI)
- Process in parallel (advanced)

### **For Accuracy**
- Use smaller batch sizes (10-20)
- Use multiple LLM consensus
- Review uncertain cases manually

### **For Cost**
- Use basic validation first
- Filter obvious cases before LLM
- Cache validation results

## 🔍 Advanced Usage

### **Multi-LLM Consensus**
```python
# Validate with multiple LLMs
validators = [
    LLMTripleValidator(validation_method="gemini"),
    LLMTripleValidator(validation_method="openai"),
    LLMTripleValidator(validation_method="basic")
]

# Compare results
consensus_results = compare_validation_results(validators)
```

### **Custom Validation Rules**
```python
# Add custom validation logic
def custom_validation(triple):
    # Your custom logic here
    return "VALID" if custom_check(triple) else "INVALID"

validator.custom_validation = custom_validation
```

### **Batch Processing**
```python
# Process specific batches
validator = LLMTripleValidator()
for batch in validator.get_batches():
    results = validator.validate_triple_batch(batch)
    # Process results
```

## 📁 Output Files

### **Validation Results**
- `llm_validated_triples_gemini_20241201_143022.json`
- Contains validated triples with metadata
- Includes validation statistics

### **Sample Structure**
```json
{
  "metadata": {
    "created_at": "2024-12-01T14:30:22.123456",
    "validation_method": "gemini",
    "total_triples": 1190,
    "validated_triples": 856,
    "validation_stats": {
      "valid": 856,
      "invalid": 234,
      "uncertain": 100
    }
  },
  "validated_triples": [
    {
      "subject": "Main",
      "relation": "istypeof",
      "object": "Class",
      "llm_validation": "VALID",
      "validation_method": "gemini",
      "validation_timestamp": "2024-12-01T14:30:22.123456"
    }
  ]
}
```

## 🎯 Next Steps

1. **Run validation** on your triples
2. **Review results** and statistics
3. **Adjust parameters** based on performance
4. **Iterate** on validation quality
5. **Integrate** validated triples into knowledge graph

## 📞 Support

- Check the `validation_strategy.md` for detailed strategies
- Review `llm_triple_validator.py` for implementation details
- Use `run_llm_validation.py` for interactive validation
- Monitor costs and performance carefully 