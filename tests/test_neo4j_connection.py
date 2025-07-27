"""
Simple Neo4j Aura Connection Test
"""

def test_neo4j_connection():
    try:
        from neo4j import GraphDatabase

        # Your actual Aura credentials
        uri = "neo4j+s://6e7e8fbf.databases.neo4j.io"
        user = "neo4j"
        password = "8s3xntRX-IYB2dG4qlOK2J5gDGhHMa4AFFXmrr8ks6U"

        print("🔌 Testing Neo4j Aura connection...")
        print(f"URI: {uri}")

        # Create driver with minimal configuration (like your working code)
        driver = GraphDatabase.driver(uri, auth=(user, password))

        # Test connection
        with driver.session(database="neo4j") as session:
            result = session.run("RETURN 'Hello Neo4j!' AS message")
            message = result.single()["message"]
            print(f"✅ Connection successful! Message: {message}")

            # Test a simple query
            result = session.run("MATCH (n) RETURN count(n) AS node_count")
            count = result.single()["node_count"]
            print(f"📊 Current nodes in database: {count}")

        driver.close()
        return True

    except ImportError:
        print("❌ neo4j package not installed")
        print("💡 Install with: pip install neo4j")
        return False

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print(f"💡 Error type: {type(e).__name__}")

        # Check specific error types
        if "routing" in str(e).lower():
            print("💡 Try installing latest driver: pip install --upgrade neo4j")
        elif "authentication" in str(e).lower():
            print("💡 Check your username and password")
        elif "certificate" in str(e).lower() or "ssl" in str(e).lower():
            print("💡 SSL/Certificate issue - trying alternative connection...")
            return test_alternative_connection(uri, user, password)

        return False

def test_alternative_connection(uri, user, password):
    """Try alternative connection methods"""
    try:
        from neo4j import GraphDatabase

        # Try without +s (less secure but might work)
        alt_uri = uri.replace("neo4j+s://", "neo4j://")
        print(f"🔄 Trying alternative URI: {alt_uri}")

        driver = GraphDatabase.driver(alt_uri, auth=(user, password))

        with driver.session() as session:
            result = session.run("RETURN 'Alternative connection works!' AS message")
            message = result.single()["message"]
            print(f"✅ Alternative connection successful! Message: {message}")

        driver.close()
        print("💡 Use this URI in your Neo4jLocalLoader.py:", alt_uri)
        return True

    except Exception as e:
        print(f"❌ Alternative connection also failed: {e}")
        return False

if __name__ == "__main__":
    print("🎯 Neo4j Aura Connection Test")
    print("-" * 35)

    success = test_neo4j_connection()

    if success:
        print("\n✅ Connection test passed! You can now run the Neo4j loader.")
    else:
        print("\n❌ Connection test failed. Please check:")
        print("   1. Neo4j Aura instance is running")
        print("   2. Credentials are correct")
        print("   3. Internet connection is stable")
        print("   4. Try: pip install --upgrade neo4j")