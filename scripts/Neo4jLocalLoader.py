"""
Neo4j Loader for Local Neo4j Desktop Instance
"""

import json
import os
from datetime import datetime
from pathlib import Path


class Neo4jLocalLoader:
    def __init__(self,
                 validated_data_file="data/validated_data/validated_data.json",
                 uri="bolt://localhost:7687",
                 user="neo4j",
                 password="password123",
                 database="neo4j"):
        self.validated_data_file = Path(validated_data_file)
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None

    def connect_to_neo4j(self):
        """Connect to local Neo4j instance"""
        try:
            from neo4j import GraphDatabase

            print(f"🔌 Connecting to local Neo4j at {self.uri}...")

            # Simple connection for local Neo4j
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

                    print(f"✅ Connected successfully!")
                    print(f"📊 Database currently has {existing_count} nodes")
                    return True

        except ImportError:
            print("❌ neo4j package not installed. Run: pip install neo4j")
            return False
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            print("💡 Make sure Neo4j Desktop is running and database is started")
            print("💡 Check connection details:")
            print(f"   URI: {self.uri}")
            print(f"   Username: {self.user}")
            print(f"   Database: {self.database}")
            return False

    def load_validated_data(self):
        """Load validated data from previous step"""
        try:
            if not self.validated_data_file.exists():
                # Try alternative locations
                alternative_paths = [
                    Path("data/validated_data/nodes.json"),
                    Path("data/validated_data/edges.json")
                ]

                if all(p.exists() for p in alternative_paths):
                    return self.load_from_separate_files()
                else:
                    raise FileNotFoundError(f"Validated data file not found: {self.validated_data_file}")

            with open(self.validated_data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            nodes = data.get('nodes', [])
            edges = data.get('edges', [])
            metadata = data.get('metadata', {})

            print(f"📁 Loaded validated data: {len(nodes)} nodes, {len(edges)} edges")
            return nodes, edges, metadata

        except Exception as e:
            print(f"❌ Error loading validated data: {e}")
            return [], [], {}

    def load_from_separate_files(self):
        """Load from separate nodes.json and edges.json files"""
        try:
            nodes_file = Path("data/validated_data/nodes.json")
            edges_file = Path("data/validated_data/edges.json")

            with open(nodes_file, 'r', encoding='utf-8') as f:
                nodes = json.load(f)

            with open(edges_file, 'r', encoding='utf-8') as f:
                edges = json.load(f)

            metadata = {
                "loaded_from": "separate_files",
                "total_nodes": len(nodes),
                "total_edges": len(edges)
            }

            print(f"📁 Loaded from separate files: {len(nodes)} nodes, {len(edges)} edges")
            return nodes, edges, metadata

        except Exception as e:
            print(f"❌ Error loading from separate files: {e}")
            return [], [], {}

    def clear_existing_data(self):
        """Clear existing data in Neo4j"""
        try:
            with self.driver.session(database=self.database) as session:
                print("🧹 Clearing existing data...")

                # Delete all relationships first, then nodes
                session.run("MATCH ()-[r]-() DELETE r")
                session.run("MATCH (n) DELETE n")

                print("✅ Existing data cleared")
                return True

        except Exception as e:
            print(f"⚠️ Error clearing existing data: {e}")
            return False

    def create_constraints_and_indexes(self):
        """Create constraints and indexes for performance"""
        try:
            with self.driver.session(database=self.database) as session:
                print("🏗️ Creating constraints and indexes...")

                # Constraints and indexes for local Neo4j
                operations = [
                    ("CONSTRAINT", "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE"),
                    ("INDEX", "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)"),
                    ("INDEX", "CREATE INDEX entity_source_count_index IF NOT EXISTS FOR (e:Entity) ON (e.source_count)")
                ]

                for op_type, query in operations:
                    try:
                        session.run(query)
                        print(f"✅ Created {op_type.lower()}")
                    except Exception as e:
                        if "already exists" in str(e).lower() or "equivalent" in str(e).lower():
                            print(f"ℹ️ {op_type} already exists")
                        else:
                            print(f"⚠️ Warning creating {op_type.lower()}: {e}")

                print("✅ Database optimization completed")
                return True

        except Exception as e:
            print(f"⚠️ Error optimizing database: {e}")
            return False

    def load_nodes(self, nodes):
        """Load nodes into local Neo4j with optimized batching"""
        try:
            print(f"📤 Loading {len(nodes)} nodes...")

            with self.driver.session(database=self.database) as session:
                # Larger batch size for local Neo4j (better performance)
                batch_size = 100
                loaded_count = 0

                for i in range(0, len(nodes), batch_size):
                    batch = nodes[i:i + batch_size]

                    # Prepare batch data with validation
                    node_data = []
                    for node in batch:
                        name = str(node.get('name', '')).strip()
                        if name and len(name) > 0:  # Valid name check
                            node_data.append({
                                'name': name,
                                'type': str(node.get('type', 'Entity')).strip(),
                                'source_count': int(node.get('source_count', 1))
                            })

                    if node_data:
                        # Batch insert with MERGE for duplicate handling
                        session.run("""
                            UNWIND $nodes AS node
                            MERGE (e:Entity {name: node.name})
                            SET e.type = node.type,
                                e.source_count = node.source_count,
                                e.created_at = datetime(),
                                e.updated_at = datetime()
                        """, nodes=node_data)

                        loaded_count += len(node_data)

                    # Progress reporting
                    if (i + batch_size) % 500 == 0 or i + batch_size >= len(nodes):
                        print(f"📈 Nodes: {min(i + batch_size, len(nodes))}/{len(nodes)} processed, {loaded_count} loaded")

                print(f"✅ Successfully loaded {loaded_count} nodes")
                return True

        except Exception as e:
            print(f"❌ Error loading nodes: {e}")
            return False

    def load_edges(self, edges):
        """Load relationships into local Neo4j with optimized batching"""
        try:
            print(f"📤 Loading {len(edges)} relationships...")

            with self.driver.session(database=self.database) as session:
                # Group edges by relation type for efficiency
                edges_by_relation = {}
                for edge in edges:
                    relation = str(edge.get('relation', 'RELATED_TO')).strip()
                    # Clean relation name for Cypher
                    relation = self.clean_relation_name(relation)

                    if relation:
                        if relation not in edges_by_relation:
                            edges_by_relation[relation] = []
                        edges_by_relation[relation].append(edge)

                total_loaded = 0

                for relation_type, relation_edges in edges_by_relation.items():
                    print(f"🔗 Loading {len(relation_edges)} '{relation_type}' relationships...")

                    # Larger batch size for local Neo4j
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
                                    'source_url': str(edge.get('source_url', ''))[:500],  # Limit URL length
                                    'source_type': str(edge.get('source_type', 'text'))
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
                                r.created_at = datetime(),
                                r.updated_at = datetime()
                            """

                            try:
                                session.run(query, edges=edge_data)
                                loaded_in_relation += len(edge_data)
                                total_loaded += len(edge_data)
                            except Exception as e:
                                print(f"⚠️ Warning loading {relation_type} batch: {str(e)[:80]}...")

                    print(f"   ✅ {relation_type}: {loaded_in_relation}/{len(relation_edges)} loaded")

                print(f"✅ Successfully loaded {total_loaded} relationships across {len(edges_by_relation)} types")
                return True

        except Exception as e:
            print(f"❌ Error loading relationships: {e}")
            return False

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

    def create_summary_statistics(self):
        """Generate comprehensive database statistics"""
        try:
            with self.driver.session(database=self.database) as session:
                print("📊 Generating comprehensive statistics...")

                # Basic counts
                node_count = session.run("MATCH (n:Entity) RETURN count(n) as count").single()["count"]
                rel_count = session.run("MATCH ()-[r]-() RETURN count(r) as count").single()["count"]

                # Node types distribution
                node_types = session.run("""
                    MATCH (n:Entity)
                    RETURN n.type as type, count(n) as count
                    ORDER BY count DESC
                    LIMIT 10
                """).data()

                # Relationship types
                rel_types = session.run("""
                    MATCH ()-[r]-() 
                    RETURN type(r) as rel_type, count(r) as count 
                    ORDER BY count DESC
                    LIMIT 15
                """).data()

                # Most connected nodes
                top_nodes = session.run("""
                    MATCH (n:Entity)
                    OPTIONAL MATCH (n)-[r]-()
                    RETURN n.name as name, n.type as type, count(r) as connections
                    ORDER BY connections DESC
                    LIMIT 10
                """).data()

                # Display statistics
                print(f"\n📈 KNOWLEDGE GRAPH STATISTICS")
                print("=" * 45)
                print(f"📊 Total Nodes: {node_count:,}")
                print(f"🔗 Total Relationships: {rel_count:,}")
                print(f"🏷️  Node Types: {len(node_types)}")
                print(f"🔀 Relationship Types: {len(rel_types)}")

                if node_types:
                    print(f"\n🏷️  TOP NODE TYPES:")
                    for nt in node_types[:5]:
                        print(f"   • {nt['type']}: {nt['count']:,} nodes")

                if rel_types:
                    print(f"\n🔀 TOP RELATIONSHIP TYPES:")
                    for rt in rel_types[:5]:
                        print(f"   • {rt['rel_type']}: {rt['count']:,} relationships")

                if top_nodes:
                    print(f"\n🌟 MOST CONNECTED ENTITIES:")
                    for node in top_nodes[:5]:
                        print(f"   • {node['name']} ({node['type']}): {node['connections']} connections")

                return {
                    "total_nodes": node_count,
                    "total_relationships": rel_count,
                    "node_types": node_types,
                    "relationship_types": rel_types,
                    "top_connected_nodes": top_nodes,
                    "created_at": datetime.now().isoformat()
                }

        except Exception as e:
            print(f"⚠️ Error generating statistics: {e}")
            return {}

    def load_data_to_neo4j(self, clear_existing=False):
        """Main method to load all data into local Neo4j"""
        print("=" * 60)
        print("📊 Step 6: Loading Data into Local Neo4j")
        print("=" * 60)

        try:
            # Connect to Neo4j
            if not self.connect_to_neo4j():
                return False

            # Load validated data
            nodes, edges, metadata = self.load_validated_data()
            if not nodes and not edges:
                print("❌ No validated data to load")
                print("💡 Make sure you've run steps 1-5 first")
                return False

            # Clear existing data if requested
            if clear_existing:
                self.clear_existing_data()

            # Setup database optimization
            self.create_constraints_and_indexes()

            # Load data with progress tracking
            print(f"\n📦 DATA LOADING PHASE")
            print("-" * 30)

            if not self.load_nodes(nodes):
                return False

            if not self.load_edges(edges):
                return False

            # Generate comprehensive statistics
            print(f"\n📊 ANALYSIS PHASE")
            print("-" * 20)
            stats = self.create_summary_statistics()

            # Success summary
            print(f"\n🎉 SUCCESS! Knowledge Graph loaded successfully")
            print("=" * 50)
            print(f"✅ Local Neo4j: {self.uri}")
            print(f"✅ Database: {self.database}")
            print(f"✅ Browser: http://localhost:7474")
            print(f"✅ Data loaded at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            return True

        except Exception as e:
            print(f"❌ Error in loading process: {e}")
            return False
        finally:
            if self.driver:
                self.driver.close()

    def get_sample_queries(self):
        """Sample Cypher queries optimized for local exploration"""
        return [
            "// 1. Overview: Show random sample of nodes\nMATCH (n) RETURN n LIMIT 25",

            "// 2. Schema: Show all labels and relationship types\nCALL db.labels() YIELD label RETURN label\nUNION\nCALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType as label",

            "// 3. Analysis: Node and relationship counts\nMATCH (n) RETURN count(n) as nodes\nUNION\nMATCH ()-[r]-() RETURN count(r) as relationships",

            "// 4. Discovery: Most connected entities\nMATCH (n:Entity)\nOPTIONAL MATCH (n)-[r]-()\nRETURN n.name, n.type, count(r) as connections\nORDER BY connections DESC LIMIT 10",

            "// 5. Exploration: Relationship type distribution\nMATCH ()-[r]-()\nRETURN type(r) as relationship_type, count(r) as count\nORDER BY count DESC",

            "// 6. Pattern: Find inheritance chains\nMATCH (child)-[:extends]->(parent)\nRETURN child.name as child, parent.name as parent",

            "// 7. Search: Java-related concepts and their connections\nMATCH (n:Entity)-[r]-(connected)\nWHERE n.name CONTAINS 'Java'\nRETURN n, r, connected LIMIT 20",

            "// 8. Advanced: Find concept clusters (densely connected groups)\nMATCH (center:Entity)-[r1]-(connected1)-[r2]-(connected2)-[r3]-(center)\nWHERE center <> connected1 AND center <> connected2 AND connected1 <> connected2\nRETURN center.name, count(*) as cluster_size\nORDER BY cluster_size DESC LIMIT 5"
        ]


def main():
    """Run Neo4j data loading for local instance"""
    print("🎯 Knowledge Graph Loader - Local Neo4j")
    print("=" * 45)

    # Default local Neo4j connection settings
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "password123"  # Change this to your Neo4j password
    database = "neo4j"

    # Allow override via environment variables
    uri = os.getenv("NEO4J_LOCAL_URI", uri)
    user = os.getenv("NEO4J_LOCAL_USER", user)
    password = os.getenv("NEO4J_LOCAL_PASSWORD", password)
    database = os.getenv("NEO4J_LOCAL_DATABASE", database)

    print(f"🔧 Configuration:")
    print(f"   URI: {uri}")
    print(f"   User: {user}")
    print(f"   Database: {database}")
    print()

    loader = Neo4jLocalLoader(uri=uri, user=user, password=password, database=database)
    success = loader.load_data_to_neo4j(clear_existing=True)

    if success:
        print(f"\n🔍 SAMPLE QUERIES FOR EXPLORATION")
        print("=" * 40)

        queries = loader.get_sample_queries()
        for i, query in enumerate(queries[:4], 1):
            print(f"\n{i}. {query}")

        print(f"\n🌐 NEXT STEPS:")
        print("=" * 15)
        print("1. Open Neo4j Browser: http://localhost:7474")
        print("2. Login with your credentials")
        print("3. Try the sample queries above")
        print("4. Explore your knowledge graph!")
        print("5. Use graph visualization to see relationships")

        return True
    else:
        print("\n❌ Loading failed")
        print("💡 Troubleshooting:")
        print("   • Start Neo4j Desktop and your database")
        print("   • Check if Neo4j is running on port 7687")
        print("   • Verify username/password are correct")
        print("   • Make sure you've completed steps 1-5 first")
        return False


if __name__ == "__main__":
    main()