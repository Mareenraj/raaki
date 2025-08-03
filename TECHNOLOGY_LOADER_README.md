# Technology-Specific Neo4j Knowledge Graph Loader

This loader allows you to input triples from separated technology files into a Neo4j database, creating a comprehensive knowledge graph that preserves technology-specific information while enabling cross-technology analysis.

## 🚀 Features

- **Technology-Specific Loading**: Load triples from individual technology files (Python, Java, JavaScript, etc.)
- **Cross-Technology Analysis**: Identify concepts that appear across multiple technologies
- **Flexible Data Structure**: Supports various triple formats (subject-predicate-object, nodes-edges, nested triples)
- **Batch Processing**: Efficient loading with configurable batch sizes
- **Technology Tagging**: Automatically tags nodes and relationships with their source technology
- **Comprehensive Statistics**: Detailed analytics on technology distribution and relationships
- **Sample Queries**: Pre-built Cypher queries for exploration

## 📁 File Structure

```
scripts/
├── TechnologyNeo4jLoader.py          # Main loader class
└── run_technology_loader.py          # Simple runner script

data/
└── extracted_triples/
    ├── ontology_enhanced_triples_python.json
    ├── ontology_enhanced_triples_java.json
    ├── ontology_enhanced_triples_js.json
    ├── ontology_enhanced_triples_html.json
    ├── ontology_enhanced_triples_css.json
    ├── ontology_enhanced_triples_sql.json
    └── ontology_enhanced_triples_react.json
```

## 🛠️ Installation

1. **Install Neo4j Python Driver**:
   ```bash
   pip install neo4j
   ```

2. **Start Neo4j Database**:
   - Start Neo4j Desktop
   - Create/start a database (default: `trainee-kg`)
   - Note your connection details (URI, username, password)

## 🔧 Configuration

### Environment Variables (Optional)
```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password"
export NEO4J_DATABASE="trainee-kg"
```

### Default Configuration
- **URI**: `bolt://localhost:7687`
- **User**: `neo4j`
- **Password**: `Ragav@2000` (change this!)
- **Database**: `trainee-kg`

## 📖 Usage

### Method 1: Simple Runner Script

```bash
cd scripts
python run_technology_loader.py
```

Choose from the menu:
1. Load all available technologies
2. Load specific technologies (Python, Java)
3. Load with custom configuration
4. Exit

### Method 2: Direct Python Usage

```python
from TechnologyNeo4jLoader import TechnologyNeo4jLoader

# Create loader
loader = TechnologyNeo4jLoader(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="your_password",
    database="trainee-kg"
)

# Load all technologies
success = loader.load_all_technologies(clear_existing=True)

# Load specific technologies
success = loader.load_all_technologies(
    clear_existing=True,
    technologies=["python", "java", "javascript"]
)
```

### Method 3: Command Line

```bash
cd scripts
python TechnologyNeo4jLoader.py
```

## 📊 Supported Data Formats

The loader supports multiple triple formats:

### Format 1: Subject-Predicate-Object
```json
[
  {
    "subject": "Python Class",
    "predicate": "inherits_from",
    "object": "Object",
    "confidence": 0.9,
    "source_url": "https://example.com"
  }
]
```

### Format 2: Nodes and Edges
```json
{
  "nodes": [
    {"name": "Python Class", "type": "Entity", "technology": "PYTHON"},
    {"name": "Object", "type": "Entity", "technology": "PYTHON"}
  ],
  "edges": [
    {
      "from": "Python Class",
      "to": "Object",
      "relation": "inherits_from",
      "technology": "PYTHON"
    }
  ]
}
```

### Format 3: Nested Triples
```json
{
  "triples": [
    {
      "subject": "Python Class",
      "predicate": "inherits_from",
      "object": "Object"
    }
  ]
}
```

## 🔍 Sample Queries

After loading, you can explore your knowledge graph with these Cypher queries:

### 1. View All Technologies
```cypher
MATCH (n:Entity)
UNWIND split(n.technology, ',') as tech
RETURN DISTINCT tech as technology
ORDER BY technology
```

### 2. Technology-Specific Nodes
```cypher
MATCH (n:Entity)
WHERE n.technology CONTAINS 'PYTHON'
RETURN n.name, n.type, n.technology
LIMIT 20
```

### 3. Cross-Technology Relationships
```cypher
MATCH (n1:Entity)-[r]-(n2:Entity)
WHERE n1.technology <> n2.technology
RETURN n1.name, type(r), n2.name
LIMIT 15
```

### 4. Technology Distribution
```cypher
MATCH (n:Entity)
UNWIND split(n.technology, ',') as tech
RETURN tech as technology, count(n) as node_count
ORDER BY node_count DESC
```

### 5. Multi-Technology Concepts
```cypher
MATCH (n:Entity)
WHERE size(split(n.technology, ',')) > 1
RETURN n.name, n.type, n.technology
ORDER BY size(split(n.technology, ',')) DESC
LIMIT 15
```

## 📈 Statistics and Analytics

The loader provides comprehensive statistics:

- **Total nodes and relationships by technology**
- **Technology distribution across the graph**
- **Most connected entities by technology**
- **Relationship types by technology**
- **Cross-technology concept analysis**

## 🔧 Advanced Usage

### Custom Technology File Pattern
```python
loader = TechnologyNeo4jLoader()
loader.tech_file_pattern = "my_triples_*.json"
```

### Batch Size Configuration
The loader uses optimized batch sizes:
- **Nodes**: 100 per batch
- **Relationships**: 50 per batch (grouped by type)

### Error Handling
- Automatic retry for failed batches
- Detailed error logging
- Graceful handling of malformed data

## 🚨 Troubleshooting

### Connection Issues
1. **Neo4j not running**: Start Neo4j Desktop and your database
2. **Wrong credentials**: Check username/password in configuration
3. **Wrong port**: Verify Neo4j is running on port 7687

### Data Loading Issues
1. **No technology files found**: Check `data/extracted_triples/` directory
2. **File format errors**: Ensure files follow expected JSON structure
3. **Memory issues**: Reduce batch sizes for large datasets

### Performance Issues
1. **Slow loading**: Check Neo4j memory settings
2. **Index creation errors**: Clear database and retry
3. **Batch failures**: Check for malformed data in triple files

## 📋 Requirements

- Python 3.7+
- Neo4j Python Driver
- Neo4j Database (Desktop or Server)
- Technology triple files in JSON format

## 🎯 Expected Technology Files

The loader expects files with the pattern: `ontology_enhanced_triples_*.json`

Available technologies:
- `ontology_enhanced_triples_python.json`
- `ontology_enhanced_triples_java.json`
- `ontology_enhanced_triples_js.json`
- `ontology_enhanced_triples_html.json`
- `ontology_enhanced_triples_css.json`
- `ontology_enhanced_triples_sql.json`
- `ontology_enhanced_triples_react.json`

## 🔄 Next Steps

After loading your technology knowledge graph:

1. **Explore in Neo4j Browser**: http://localhost:7474
2. **Run sample queries** to understand your data
3. **Analyze cross-technology patterns**
4. **Build applications** that leverage the knowledge graph
5. **Add more technologies** by creating new triple files

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify your Neo4j connection settings
3. Ensure your triple files follow the expected format
4. Check the logs for detailed error messages

---

**Happy Knowledge Graph Building! 🚀** 