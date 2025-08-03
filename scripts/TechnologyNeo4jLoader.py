"""
Technology-Specific Neo4j Loader
Loads triples from separated technology files into Neo4j database
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TechnologyNeo4jLoader:
    def __init__(self,
                 triples_dir="scripts/data/extracted_triples",
                 uri="bolt://localhost:7687",
                 user="neo4j",
                 password="Ragav@2000",
                 database="trainee-kg"):
        self.triples_dir = Path(triples_dir)
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None
        
        # Technology file patterns
        self.tech_file_pattern = "ontology_enhanced_triples_*.json"
        
        # Statistics tracking
        self.stats = {
            "technologies_loaded": [],
            "total_nodes": 0,
            "total_relationships": 0,
            "nodes_by_tech": {},
            "relationships_by_tech": {},
            "errors": []
        }

    def connect_to_neo4j(self):
        """Connect to Neo4j database"""
        try:
            from neo4j import GraphDatabase

            logger.info(f"🔌 Connecting to Neo4j at {self.uri}...")

            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )

            # Test connection
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]

                if test_value == 1:
                    # Check existing data
                    count_result = session.run("MATCH (n) RETURN count(n) as count")
                    existing_count = count_result.single()["count"]

                    logger.info(f"✅ Connected successfully!")
                    logger.info(f"📊 Database currently has {existing_count} nodes")
                    return True

        except ImportError:
            logger.error("❌ neo4j package not installed. Run: pip install neo4j")
            return False
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            logger.info("💡 Make sure Neo4j is running and credentials are correct")
            return False

    def discover_technology_files(self) -> List[Path]:
        """Discover all technology-specific triple files"""
        try:
            tech_files = list(self.triples_dir.glob(self.tech_file_pattern))
            
            if not tech_files:
                logger.warning(f"⚠️ No technology files found in {self.triples_dir}")
                logger.info(f"💡 Expected pattern: {self.tech_file_pattern}")
                return []
            
            logger.info(f"📁 Found {len(tech_files)} technology files:")
            for file in tech_files:
                tech_name = file.stem.replace("ontology_enhanced_triples_", "").upper()
                logger.info(f"   • {tech_name}: {file.name}")
            
            return tech_files
            
        except Exception as e:
            logger.error(f"❌ Error discovering technology files: {e}")
            return []

    def load_technology_triples(self, file_path: Path) -> Tuple[List[Dict], List[Dict], str]:
        """Load triples from a specific technology file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract technology name from filename
            tech_name = file_path.stem.replace("ontology_enhanced_triples_", "").upper()
            
            # Process triples based on expected structure
            nodes = []
            edges = []
            
            if isinstance(data, list):
                # Direct triple list
                for triple in data:
                    if isinstance(triple, dict) and 'subject' in triple and 'predicate' in triple and 'object' in triple:
                        # Create nodes for subject and object
                        nodes.extend([
                            {'name': triple['subject'], 'type': 'Entity', 'technology': tech_name},
                            {'name': triple['object'], 'type': 'Entity', 'technology': tech_name}
                        ])
                        
                        # Create edge
                        edges.append({
                            'from': triple['subject'],
                            'to': triple['object'],
                            'relation': triple['predicate'],
                            'technology': tech_name,
                            'confidence': triple.get('confidence', 0.8),
                            'source_url': triple.get('source_url', ''),
                            'source_type': 'technology_triple'
                        })
            
            elif isinstance(data, dict):
                # Structured data with nodes and edges
                if 'nodes' in data and 'edges' in data:
                    for node in data['nodes']:
                        node['technology'] = tech_name
                        nodes.append(node)
                    
                    for edge in data['edges']:
                        edge['technology'] = tech_name
                        edges.append(edge)
                
                elif 'triples' in data:
                    # Nested triples structure
                    for triple in data['triples']:
                        if isinstance(triple, dict) and 'subject' in triple and 'predicate' in triple and 'object' in triple:
                            nodes.extend([
                                {'name': triple['subject'], 'type': 'Entity', 'technology': tech_name},
                                {'name': triple['object'], 'type': 'Entity', 'technology': tech_name}
                            ])
                            
                            edges.append({
                                'from': triple['subject'],
                                'to': triple['object'],
                                'relation': triple['predicate'],
                                'technology': tech_name,
                                'confidence': triple.get('confidence', 0.8),
                                'source_url': triple.get('source_url', ''),
                                'source_type': 'technology_triple'
                            })
            
            # Remove duplicate nodes
            unique_nodes = {}
            for node in nodes:
                name = node['name']
                if name not in unique_nodes:
                    unique_nodes[name] = node
                else:
                    # Merge technologies if node already exists
                    existing_tech = unique_nodes[name].get('technology', '')
                    new_tech = node.get('technology', '')
                    if new_tech and new_tech not in existing_tech:
                        unique_nodes[name]['technology'] = f"{existing_tech},{new_tech}"
            
            nodes = list(unique_nodes.values())
            
            logger.info(f"📊 {tech_name}: {len(nodes)} unique nodes, {len(edges)} edges")
            return nodes, edges, tech_name
            
        except Exception as e:
            logger.error(f"❌ Error loading {file_path}: {e}")
            return [], [], ""

    def create_technology_constraints_and_indexes(self):
        """Create constraints and indexes optimized for technology-specific data"""
        try:
            with self.driver.session(database=self.database) as session:
                logger.info("🏗️ Creating technology-specific constraints and indexes...")

                operations = [
                    ("CONSTRAINT", "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE"),
                    ("INDEX", "CREATE INDEX entity_technology_index IF NOT EXISTS FOR (e:Entity) ON (e.technology)"),
                    ("INDEX", "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)"),
                    ("INDEX", "CREATE INDEX relationship_technology_index IF NOT EXISTS FOR ()-[r]-() ON (r.technology)"),
                    ("INDEX", "CREATE INDEX relationship_type_index IF NOT EXISTS FOR ()-[r]-() ON (type(r))")
                ]

                for op_type, query in operations:
                    try:
                        session.run(query)
                        logger.info(f"✅ Created {op_type.lower()}")
                    except Exception as e:
                        if "already exists" in str(e).lower() or "equivalent" in str(e).lower():
                            logger.info(f"ℹ️ {op_type} already exists")
                        else:
                            logger.warning(f"⚠️ Warning creating {op_type.lower()}: {e}")

                logger.info("✅ Database optimization completed")
                return True

        except Exception as e:
            logger.error(f"⚠️ Error optimizing database: {e}")
            return False

    def load_technology_nodes(self, nodes: List[Dict], technology: str):
        """Load nodes for a specific technology with batching"""
        try:
            logger.info(f"📤 Loading {len(nodes)} nodes for {technology}...")

            with self.driver.session(database=self.database) as session:
                batch_size = 100
                loaded_count = 0

                for i in range(0, len(nodes), batch_size):
                    batch = nodes[i:i + batch_size]

                    # Prepare batch data with validation
                    node_data = []
                    for node in batch:
                        name = str(node.get('name', '')).strip()
                        if name and len(name) > 0:
                            node_data.append({
                                'name': name,
                                'type': str(node.get('type', 'Entity')).strip(),
                                'technology': str(node.get('technology', technology)).strip(),
                                'source_count': int(node.get('source_count', 1))
                            })

                    if node_data:
                        # Batch insert with MERGE for duplicate handling
                        session.run("""
                            UNWIND $nodes AS node
                            MERGE (e:Entity {name: node.name})
                            SET e.type = COALESCE(e.type, node.type),
                                e.technology = CASE 
                                    WHEN e.technology IS NULL THEN node.technology
                                    WHEN node.technology NOT IN e.technology THEN e.technology + ',' + node.technology
                                    ELSE e.technology
                                END,
                                e.source_count = COALESCE(e.source_count, 0) + node.source_count,
                                e.created_at = COALESCE(e.created_at, datetime()),
                                e.updated_at = datetime()
                        """, nodes=node_data)

                        loaded_count += len(node_data)

                    # Progress reporting
                    if (i + batch_size) % 500 == 0 or i + batch_size >= len(nodes):
                        logger.info(f"📈 {technology} Nodes: {min(i + batch_size, len(nodes))}/{len(nodes)} processed, {loaded_count} loaded")

                logger.info(f"✅ Successfully loaded {loaded_count} nodes for {technology}")
                return loaded_count

        except Exception as e:
            logger.error(f"❌ Error loading {technology} nodes: {e}")
            return 0

    def load_technology_edges(self, edges: List[Dict], technology: str):
        """Load relationships for a specific technology with batching"""
        try:
            logger.info(f"📤 Loading {len(edges)} relationships for {technology}...")

            with self.driver.session(database=self.database) as session:
                # Group edges by relation type for efficiency
                edges_by_relation = {}
                for edge in edges:
                    relation = str(edge.get('relation', 'RELATED_TO')).strip()
                    relation = self.clean_relation_name(relation)

                    if relation:
                        if relation not in edges_by_relation:
                            edges_by_relation[relation] = []
                        edges_by_relation[relation].append(edge)

                total_loaded = 0

                for relation_type, relation_edges in edges_by_relation.items():
                    logger.info(f"🔗 Loading {len(relation_edges)} '{relation_type}' relationships for {technology}...")

                    batch_size = 50
                    loaded_in_relation = 0

                    for i in range(0, len(relation_edges), batch_size):
                        batch = relation_edges[i:i + batch_size]

                        # Prepare batch data with validation
                        edge_data = []
                        for edge in batch:
                            from_name = str(edge.get('from', '')).strip()
                            to_name = str(edge.get('to', '')).strip()

                            if from_name and to_name and from_name != to_name:
                                edge_data.append({
                                    'from_name': from_name,
                                    'to_name': to_name,
                                    'confidence': float(edge.get('confidence', 0.5)),
                                    'source_url': str(edge.get('source_url', ''))[:500],
                                    'source_type': str(edge.get('source_type', 'technology_triple')),
                                    'technology': str(edge.get('technology', technology))
                                })

                        if edge_data:
                            query = f"""
                            UNWIND $edges AS edge
                            MATCH (from:Entity {{name: edge.from_name}})
                            MATCH (to:Entity {{name: edge.to_name}})
                            MERGE (from)-[r:`{relation_type}`]->(to)
                            SET r.confidence = edge.confidence,
                                r.source_url = edge.source_url,
                                r.source_type = edge.source_type,
                                r.technology = CASE 
                                    WHEN r.technology IS NULL THEN edge.technology
                                    WHEN edge.technology NOT IN r.technology THEN r.technology + ',' + edge.technology
                                    ELSE r.technology
                                END,
                                r.created_at = COALESCE(r.created_at, datetime()),
                                r.updated_at = datetime()
                            """

                            try:
                                session.run(query, edges=edge_data)
                                loaded_in_relation += len(edge_data)
                                total_loaded += len(edge_data)
                            except Exception as e:
                                logger.warning(f"⚠️ Warning loading {relation_type} batch for {technology}: {str(e)[:80]}...")

                    logger.info(f"   ✅ {technology} {relation_type}: {loaded_in_relation}/{len(relation_edges)} loaded")

                logger.info(f"✅ Successfully loaded {total_loaded} relationships for {technology}")
                return total_loaded

        except Exception as e:
            logger.error(f"❌ Error loading {technology} relationships: {e}")
            return 0

    def clean_relation_name(self, relation):
        """Clean relation name for Cypher compatibility"""
        if not relation:
            return "RELATED_TO"

        # Replace problematic characters
        cleaned = relation.replace(' ', '_').replace('-', '_').replace('.', '_')
        cleaned = ''.join(c for c in cleaned if c.isalnum() or c == '_')

        # Ensure it starts with a letter
        if cleaned and not cleaned[0].isalpha():
            cleaned = 'REL_' + cleaned

        return cleaned if cleaned else "RELATED_TO"

    def create_technology_statistics(self):
        """Generate comprehensive statistics for technology-specific data"""
        try:
            with self.driver.session(database=self.database) as session:
                logger.info("📊 Generating technology-specific statistics...")

                # Basic counts
                node_count = session.run("MATCH (n:Entity) RETURN count(n) as count").single()["count"]
                rel_count = session.run("MATCH ()-[r]-() RETURN count(r) as count").single()["count"]

                # Technology distribution
                tech_distribution = session.run("""
                    MATCH (n:Entity)
                    UNWIND split(n.technology, ',') as tech
                    RETURN tech as technology, count(n) as count
                    ORDER BY count DESC
                """).data()

                # Relationship types by technology
                rel_by_tech = session.run("""
                    MATCH ()-[r]-()
                    UNWIND split(r.technology, ',') as tech
                    RETURN tech as technology, type(r) as rel_type, count(r) as count
                    ORDER BY tech, count DESC
                """).data()

                # Most connected entities by technology
                top_nodes_by_tech = session.run("""
                    MATCH (n:Entity)
                    UNWIND split(n.technology, ',') as tech
                    OPTIONAL MATCH (n)-[r]-()
                    RETURN tech as technology, n.name as name, n.type as type, count(r) as connections
                    ORDER BY tech, connections DESC
                """).data()

                # Display statistics
                print(f"\n📈 TECHNOLOGY-SPECIFIC KNOWLEDGE GRAPH STATISTICS")
                print("=" * 60)
                print(f"📊 Total Nodes: {node_count:,}")
                print(f"🔗 Total Relationships: {rel_count:,}")

                if tech_distribution:
                    print(f"\n🏷️  TECHNOLOGY DISTRIBUTION:")
                    for tech in tech_distribution[:10]:
                        print(f"   • {tech['technology']}: {tech['count']:,} nodes")

                if rel_by_tech:
                    print(f"\n🔀 RELATIONSHIP TYPES BY TECHNOLOGY:")
                    current_tech = ""
                    for rel in rel_by_tech[:15]:
                        if rel['technology'] != current_tech:
                            current_tech = rel['technology']
                            print(f"\n   📌 {current_tech}:")
                        print(f"      • {rel['rel_type']}: {rel['count']:,} relationships")

                if top_nodes_by_tech:
                    print(f"\n🌟 MOST CONNECTED ENTITIES BY TECHNOLOGY:")
                    current_tech = ""
                    for node in top_nodes_by_tech[:15]:
                        if node['technology'] != current_tech:
                            current_tech = node['technology']
                            print(f"\n   📌 {current_tech}:")
                        print(f"      • {node['name']} ({node['type']}): {node['connections']} connections")

                return {
                    "total_nodes": node_count,
                    "total_relationships": rel_count,
                    "technology_distribution": tech_distribution,
                    "relationships_by_technology": rel_by_tech,
                    "top_nodes_by_technology": top_nodes_by_tech,
                    "created_at": datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"⚠️ Error generating statistics: {e}")
            return {}

    def load_all_technologies(self, clear_existing=False, technologies=None):
        """Load all technology files or specific technologies"""
        print("=" * 70)
        print("🚀 Technology-Specific Neo4j Knowledge Graph Loader")
        print("=" * 70)

        try:
            # Connect to Neo4j
            if not self.connect_to_neo4j():
                return False

            # Discover technology files
            tech_files = self.discover_technology_files()
            if not tech_files:
                return False

            # Filter by specific technologies if provided
            if technologies:
                filtered_files = []
                for file in tech_files:
                    tech_name = file.stem.replace("ontology_enhanced_triples_", "").upper()
                    if tech_name.lower() in [t.lower() for t in technologies]:
                        filtered_files.append(file)
                tech_files = filtered_files
                logger.info(f"🎯 Loading specific technologies: {[f.stem.replace('ontology_enhanced_triples_', '').upper() for f in tech_files]}")

            # Clear existing data if requested
            if clear_existing:
                with self.driver.session(database=self.database) as session:
                    logger.info("🧹 Clearing existing data...")
                    session.run("MATCH ()-[r]-() DELETE r")
                    session.run("MATCH (n) DELETE n")
                    logger.info("✅ Existing data cleared")

            # Setup database optimization
            self.create_technology_constraints_and_indexes()

            # Load each technology
            print(f"\n📦 LOADING TECHNOLOGIES")
            print("-" * 30)

            for tech_file in tech_files:
                tech_name = tech_file.stem.replace("ontology_enhanced_triples_", "").upper()
                logger.info(f"\n🔄 Processing {tech_name}...")

                # Load triples for this technology
                nodes, edges, loaded_tech = self.load_technology_triples(tech_file)
                
                if nodes or edges:
                    # Load nodes
                    nodes_loaded = self.load_technology_nodes(nodes, loaded_tech)
                    
                    # Load edges
                    edges_loaded = self.load_technology_edges(edges, loaded_tech)
                    
                    # Update statistics
                    self.stats["technologies_loaded"].append(loaded_tech)
                    self.stats["nodes_by_tech"][loaded_tech] = nodes_loaded
                    self.stats["relationships_by_tech"][loaded_tech] = edges_loaded
                    self.stats["total_nodes"] += nodes_loaded
                    self.stats["total_relationships"] += edges_loaded
                    
                    logger.info(f"✅ {loaded_tech}: {nodes_loaded} nodes, {edges_loaded} relationships loaded")
                else:
                    logger.warning(f"⚠️ No data found in {tech_file}")

            # Generate comprehensive statistics
            print(f"\n📊 ANALYSIS PHASE")
            print("-" * 20)
            stats = self.create_technology_statistics()

            # Success summary
            print(f"\n🎉 SUCCESS! Technology Knowledge Graph loaded successfully")
            print("=" * 60)
            print(f"✅ Technologies loaded: {len(self.stats['technologies_loaded'])}")
            print(f"✅ Total nodes: {self.stats['total_nodes']:,}")
            print(f"✅ Total relationships: {self.stats['total_relationships']:,}")
            print(f"✅ Neo4j URI: {self.uri}")
            print(f"✅ Database: {self.database}")
            print(f"✅ Browser: http://localhost:7474")
            print(f"✅ Loaded at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            return True

        except Exception as e:
            logger.error(f"❌ Error in loading process: {e}")
            return False
        finally:
            if self.driver:
                self.driver.close()

    def get_technology_sample_queries(self):
        """Sample Cypher queries for technology-specific exploration"""
        return [
            "// 1. Overview: Show all technologies in the graph\nMATCH (n:Entity)\nUNWIND split(n.technology, ',') as tech\nRETURN DISTINCT tech as technology\nORDER BY technology",

            "// 2. Technology-specific nodes\nMATCH (n:Entity)\nWHERE n.technology CONTAINS 'JAVA'\nRETURN n.name, n.type, n.technology\nLIMIT 20",

            "// 3. Cross-technology relationships\nMATCH (n1:Entity)-[r]-(n2:Entity)\nWHERE n1.technology <> n2.technology\nRETURN n1.name, type(r), n2.name\nLIMIT 15",

            "// 4. Technology distribution\nMATCH (n:Entity)\nUNWIND split(n.technology, ',') as tech\nRETURN tech as technology, count(n) as node_count\nORDER BY node_count DESC",

            "// 5. Most connected entities by technology\nMATCH (n:Entity)\nUNWIND split(n.technology, ',') as tech\nOPTIONAL MATCH (n)-[r]-()\nRETURN tech as technology, n.name as entity, count(r) as connections\nORDER BY tech, connections DESC\nLIMIT 10",

            "// 6. Relationship types by technology\nMATCH ()-[r]-()\nUNWIND split(r.technology, ',') as tech\nRETURN tech as technology, type(r) as relationship_type, count(r) as count\nORDER BY tech, count DESC",

            "// 7. Find concepts that appear in multiple technologies\nMATCH (n:Entity)\nWHERE size(split(n.technology, ',')) > 1\nRETURN n.name, n.type, n.technology\nORDER BY size(split(n.technology, ',')) DESC\nLIMIT 15",

            "// 8. Technology-specific concept clusters\nMATCH (n:Entity)-[r]-(connected)\nWHERE n.technology CONTAINS 'PYTHON'\nRETURN n.name, type(r), connected.name\nLIMIT 20"
        ]


def main():
    """Run technology-specific Neo4j data loading"""
    print("🎯 Technology-Specific Knowledge Graph Loader")
    print("=" * 50)

    # Configuration
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "Ragav@2000")
    database = os.getenv("NEO4J_DATABASE", "trainee-kg")

    print(f"🔧 Configuration:")
    print(f"   URI: {uri}")
    print(f"   User: {user}")
    print(f"   Database: {database}")
    print()

    # Create loader
    loader = TechnologyNeo4jLoader(uri=uri, user=user, password=password, database=database)
    
    # Load all technologies
    success = loader.load_all_technologies(clear_existing=True)

    if success:
        print(f"\n🔍 SAMPLE QUERIES FOR TECHNOLOGY EXPLORATION")
        print("=" * 50)

        queries = loader.get_technology_sample_queries()
        for i, query in enumerate(queries[:5], 1):
            print(f"\n{i}. {query}")

        print(f"\n🌐 NEXT STEPS:")
        print("=" * 15)
        print("1. Open Neo4j Browser: http://localhost:7474")
        print("2. Login with your credentials")
        print("3. Try the technology-specific queries above")
        print("4. Explore cross-technology relationships")
        print("5. Analyze technology-specific patterns")

        return True
    else:
        print("\n❌ Loading failed")
        print("💡 Troubleshooting:")
        print("   • Start Neo4j and ensure it's running")
        print("   • Check connection credentials")
        print("   • Verify technology files exist in scripts/data/extracted_triples/")
        print("   • Ensure technology files follow the pattern: ontology_enhanced_triples_*.json")
        return False


if __name__ == "__main__":
    main() 