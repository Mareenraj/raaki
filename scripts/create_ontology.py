"""
Generalized Entity-Based Ontology Creator for Knowledge Graph
Creates abstract entities that can represent concepts across multiple technologies
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class GeneralizedOntologyCreator:
    def __init__(self, model_name="microsoft/DialoGPT-medium", output_dir="data/ontology", approach="generalized"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.approach = approach
        self.model = None
        self.tokenizer = None

        # Target technologies for context
        self.technologies = [
            "html", "css", "js", "python", "java", "sql",
            "bootstrap", "jquery", "json", "ajax", "xml", "api",
            "php", "csharp", "nodejs", "react", "typescript"
        ]

    def get_generalized_ontology(self) -> Dict:
        """Create generalized entity-based ontology that applies across technologies"""

        # High-level abstract entities that can represent concepts across any technology
        entities = {
            # Core Programming Entities
            "programming_constructs": [
                "Language", "Syntax", "Grammar", "Keyword", "Operator", "Literal",
                "Identifier", "Comment", "Statement", "Expression", "Block", "Scope"
            ],

            # Data & Type Entities
            "data_entities": [
                "DataType", "PrimitiveType", "CompositeType", "Structure", "Collection",
                "Container", "Element", "Item", "Entry", "Record", "Field", "Attribute",
                "Property", "Value", "Reference", "Pointer", "Index", "Key"
            ],

            # Behavioral Entities
            "behavioral_entities": [
                "Function", "Method", "Procedure", "Routine", "Operation", "Action",
                "Behavior", "Process", "Algorithm", "Logic", "Rule", "Condition",
                "Loop", "Iteration", "Recursion", "Callback", "Handler", "Listener"
            ],

            # Structural Entities
            "structural_entities": [
                "Component", "Module", "Package", "Library", "Framework", "System",
                "Architecture", "Pattern", "Structure", "Hierarchy", "Tree", "Graph",
                "Network", "Layer", "Interface", "Contract", "Protocol", "Standard"
            ],

            # Object-Oriented Entities
            "oop_entities": [
                "Class", "Object", "Instance", "Entity", "Model", "Type", "Template",
                "Blueprint", "Schema", "Definition", "Declaration", "Implementation",
                "Inheritance", "Polymorphism", "Encapsulation", "Abstraction"
            ],

            # Control Flow Entities
            "control_entities": [
                "ControlFlow", "Branch", "Decision", "Selection", "Switch", "Case",
                "Jump", "Return", "Break", "Continue", "Exception", "Error", "Fault",
                "Interrupt", "Signal", "Event", "Trigger", "Response"
            ],

            # Communication Entities
            "communication_entities": [
                "Message", "Request", "Response", "Call", "Invocation", "Communication",
                "Channel", "Connection", "Link", "Bridge", "Gateway", "Proxy",
                "Adapter", "Wrapper", "Decorator", "Interceptor", "Middleware"
            ],

            # Storage & Persistence Entities
            "storage_entities": [
                "Storage", "Database", "Repository", "Store", "Cache", "Buffer",
                "Memory", "Register", "Variable", "Constant", "Configuration",
                "Setting", "Parameter", "Argument", "Option", "Flag", "State"
            ],

            # Resource Management Entities
            "resource_entities": [
                "Resource", "Asset", "File", "Document", "Media", "Content",
                "Source", "Target", "Destination", "Path", "Location", "Address",
                "URI", "URL", "Endpoint", "Route", "Namespace", "Context"
            ],

            # Processing Entities
            "processing_entities": [
                "Processor", "Parser", "Compiler", "Interpreter", "Transpiler",
                "Transformer", "Converter", "Serializer", "Formatter", "Validator",
                "Filter", "Mapper", "Reducer", "Aggregator", "Analyzer", "Generator"
            ],

            # Security & Access Entities
            "security_entities": [
                "Security", "Authentication", "Authorization", "Permission", "Role",
                "User", "Session", "Token", "Credential", "Certificate", "Key",
                "Encryption", "Hash", "Signature", "Verification", "Trust", "Policy"
            ],

            # Lifecycle & Management Entities
            "lifecycle_entities": [
                "Lifecycle", "Creation", "Initialization", "Configuration", "Setup",
                "Startup", "Runtime", "Execution", "Shutdown", "Cleanup", "Destruction",
                "Management", "Monitor", "Controller", "Scheduler", "Timer", "Clock"
            ],

            # Quality & Testing Entities
            "quality_entities": [
                "Quality", "Test", "Validation", "Verification", "Assertion", "Mock",
                "Stub", "Spy", "Benchmark", "Metric", "Measurement", "Analysis",
                "Report", "Log", "Trace", "Debug", "Profile", "Optimization"
            ],

            # User Interface Entities
            "ui_entities": [
                "Interface", "View", "Display", "Screen", "Page", "Form", "Control",
                "Widget", "Component", "Element", "Container", "Layout", "Style",
                "Theme", "Template", "Renderer", "Presentation", "Interaction"
            ],

            # Network & Web Entities
            "network_entities": [
                "Network", "Web", "Internet", "Protocol", "Service", "Server", "Client",
                "Peer", "Node", "Host", "Domain", "Port", "Socket", "Stream",
                "Packet", "Frame", "Header", "Payload", "Transport", "Delivery"
            ],

            # Development & Tools Entities
            "development_entities": [
                "Development", "Tool", "Utility", "Helper", "Builder", "Bundler",
                "Packager", "Deployer", "Installer", "Manager", "Environment",
                "Workspace", "Project", "Solution", "Build", "Release", "Version"
            ],

            # Documentation & Metadata Entities
            "metadata_entities": [
                "Documentation", "Specification", "Description", "Annotation", "Tag",
                "Label", "Marker", "Flag", "Attribute", "Metadata", "Information",
                "Data", "Content", "Text", "String", "Character", "Symbol", "Token"
            ]
        }

        # Flatten all entities
        all_entities = []
        for category_entities in entities.values():
            all_entities.extend(category_entities)

        # Generalized relations that apply across domains
        relations = [
            # Fundamental Relations
            "isA", "hasA", "partOf", "memberOf", "elementOf", "instanceOf", "typeOf",
            "kindOf", "categoryOf", "classOf", "subclassOf", "superclassOf",

            # Structural Relations
            "contains", "includes", "comprises", "consistsOf", "composedOf", "madeOf",
            "belongsTo", "ownedBy", "childOf", "parentOf", "siblingOf", "relatedTo",

            # Behavioral Relations
            "performs", "executes", "processes", "handles", "manages", "controls",
            "operates", "functions", "behaves", "acts", "responds", "reacts",

            # Dependency Relations
            "dependsOn", "requires", "needs", "uses", "utilizes", "employs",
            "relies", "based On", "buildsOn", "extends", "inherits", "derives",

            # Communication Relations
            "communicates", "interacts", "connects", "links", "associates", "relates",
            "sends", "receives", "transmits", "delivers", "exchanges", "shares",

            # Transformation Relations
            "transforms", "converts", "changes", "modifies", "updates", "alters",
            "maps", "translates", "interprets", "compiles", "generates", "produces",

            # Control Relations
            "controls", "manages", "governs", "regulates", "coordinates", "orchestrates",
            "schedules", "triggers", "initiates", "starts", "stops", "pauses",

            # Access Relations
            "accesses", "reads", "writes", "modifies", "creates", "deletes",
            "inserts", "updates", "retrieves", "queries", "searches", "finds",

            # Validation Relations
            "validates", "verifies", "checks", "tests", "evaluates", "assesses",
            "measures", "monitors", "observes", "tracks", "records", "logs",

            # Configuration Relations
            "configures", "sets", "defines", "specifies", "declares", "establishes",
            "initializes", "setupps", "customizes", "adapts", "adjusts", "tunes",

            # Security Relations
            "authenticates", "authorizes", "permits", "denies", "grants", "revokes",
            "protects", "secures", "encrypts", "decrypts", "signs", "verifies",

            # Lifecycle Relations
            "creates", "instantiates", "constructs", "builds", "assembles", "composes",
            "destroys", "disposes", "releases", "frees", "cleans", "removes",

            # Quality Relations
            "improves", "optimizes", "enhances", "refines", "polishes", "debugs",
            "fixes", "repairs", "maintains", "supports", "sustains", "preserves",

            # Documentation Relations
            "documents", "describes", "explains", "annotates", "comments", "marks",
            "labels", "tags", "categorizes", "classifies", "indexes", "catalogs",

            # Temporal Relations
            "precedes", "follows", "succeeds", "before", "after", "during", "while",
            "when", "triggers", "causes", "results", "leads", "produces", "yields"
        ]

        return {
            "entities": sorted(list(set(all_entities))),
            "relations": sorted(list(set(relations))),
            "entity_categories": entities,
            "total_categories": len(entities),
            "applicability": "universal across programming languages and technologies"
        }

    def create_technology_mapping(self, ontology: Dict) -> Dict:
        """Map generalized entities to specific technology concepts"""

        # Example mappings showing how generalized entities apply to specific technologies
        technology_mappings = {
            "html": {
                "Component": ["HTMLElement", "Tag", "Attribute"],
                "Structure": ["Document", "DOM", "Tree"],
                "Container": ["Div", "Section", "Article"],
                "Element": ["Input", "Button", "Image"],
                "Property": ["Attribute", "Value", "Content"],
                "Template": ["HTML Template", "Document Structure"]
            },

            "css": {
                "Rule": ["CSS Rule", "Style Declaration"],
                "Property": ["CSS Property", "Style Attribute"],
                "Value": ["Property Value", "Color", "Size"],
                "Selector": ["Element Selector", "Class Selector"],
                "Container": ["Box Model", "Layout Container"],
                "Layout": ["Flexbox", "Grid", "Float"]
            },

            "javascript": {
                "Function": ["Function Declaration", "Method", "Arrow Function"],
                "Object": ["JavaScript Object", "Instance", "Prototype"],
                "Variable": ["Variable Declaration", "Identifier"],
                "Event": ["DOM Event", "Custom Event", "Handler"],
                "Module": ["ES Module", "CommonJS Module"],
                "Promise": ["Async Operation", "Future Value"]
            },

            "python": {
                "Class": ["Python Class", "Type Definition"],
                "Function": ["Function Definition", "Method", "Lambda"],
                "Module": ["Python Module", "Package"],
                "Collection": ["List", "Dictionary", "Set", "Tuple"],
                "Framework": ["Django", "Flask", "FastAPI"],
                "Library": ["NumPy", "Pandas", "Requests"]
            },

            "java": {
                "Class": ["Java Class", "Interface", "Abstract Class"],
                "Method": ["Instance Method", "Static Method", "Constructor"],
                "Package": ["Java Package", "Namespace"],
                "Framework": ["Spring", "Hibernate", "Maven"],
                "Container": ["Collection", "List", "Map", "Set"],
                "Exception": ["Checked Exception", "Runtime Exception"]
            },

            "sql": {
                "Structure": ["Table", "View", "Index"],
                "Element": ["Row", "Column", "Field"],
                "Key": ["Primary Key", "Foreign Key", "Unique Key"],
                "Operation": ["Select", "Insert", "Update", "Delete"],
                "Function": ["Aggregate Function", "Scalar Function"],
                "Relationship": ["Join", "Foreign Key Constraint"]
            },

            "react": {
                "Component": ["React Component", "Functional Component"],
                "Function": ["Hook", "Event Handler", "Lifecycle Method"],
                "State": ["Component State", "Global State"],
                "Property": ["Props", "Attributes"],
                "Event": ["React Event", "Synthetic Event"],
                "Context": ["React Context", "Provider"]
            },

            "api": {
                "Endpoint": ["API Endpoint", "Resource URL"],
                "Request": ["HTTP Request", "API Call"],
                "Response": ["API Response", "Data Payload"],
                "Method": ["HTTP Method", "Operation"],
                "Authentication": ["API Key", "OAuth", "JWT"],
                "Resource": ["API Resource", "Data Entity"]
            }
        }

        return technology_mappings

    def load_model(self):
        """Load model and tokenizer with error handling"""
        try:
            print(f"Loading model: {self.model_name}")
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            print("✅ Model loaded successfully!")
            return True

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def generate_ontology_with_llm(self) -> Dict:
        """Generate generalized ontology using LLM"""
        tech_list = ", ".join(self.technologies)
        prompt = f"""Create a generalized ontology with abstract entities that can represent concepts across multiple programming technologies: {tech_list}

Focus on creating GENERALIZED entities that are applicable across different technologies rather than technology-specific classes.

For example:
- Instead of "HTMLElement", "ReactComponent", "JavaClass" use "Component"
- Instead of "CSSProperty", "JavaField", "PythonAttribute" use "Property" 
- Instead of "JSFunction", "PythonFunction", "JavaMethod" use "Function"

Create abstract entities for:
- Programming constructs (Language, Syntax, Statement, Expression)
- Data types and structures (DataType, Collection, Container, Element)
- Behavioral concepts (Function, Method, Operation, Process)
- Structural concepts (Component, Module, System, Architecture)
- Communication concepts (Message, Request, Response, Protocol)
- Control flow concepts (Branch, Loop, Exception, Event)

Return valid JSON:
{{
  "entities": ["Entity1", "Entity2", ...],
  "relations": ["relation1", "relation2", ...]
}}

JSON ontology:"""

        try:
            import torch

            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=1200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    repetition_penalty=1.1
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                ontology_text = response[json_start:json_end]
                ontology = json.loads(ontology_text)

                if "entities" in ontology and "relations" in ontology:
                    print("✅ Generated generalized ontology using LLM")
                    return {
                        "entities": ontology["entities"],
                        "relations": ontology["relations"],
                        "source": "llm_generated"
                    }

            print("⚠️ LLM output invalid, falling back to curated ontology")
            return self.get_generalized_ontology()

        except Exception as e:
            print(f"⚠️ Error with LLM generation: {e}")
            return self.get_generalized_ontology()

    def create_hybrid_ontology(self) -> Dict:
        """Create hybrid ontology combining curated and LLM-generated content"""
        base_ontology = self.get_generalized_ontology()

        if self.load_model():
            try:
                llm_ontology = self.generate_ontology_with_llm()

                # Merge ontologies
                combined_entities = list(set(base_ontology["entities"] + llm_ontology.get("entities", [])))
                combined_relations = list(set(base_ontology["relations"] + llm_ontology.get("relations", [])))

                print("✅ Created hybrid generalized ontology")
                return {
                    "entities": sorted(combined_entities),
                    "relations": sorted(combined_relations),
                    "entity_categories": base_ontology.get("entity_categories", {}),
                    "source": "hybrid"
                }
            except Exception as e:
                print(f"⚠️ Hybrid creation failed: {e}, using curated ontology")

        return base_ontology

    def generate_ontology(self) -> Dict:
        """Generate ontology based on selected approach"""
        print(f"🎯 Using approach: {self.approach}")

        if self.approach == "llm":
            if self.load_model():
                return self.generate_ontology_with_llm()
            else:
                print("⚠️ LLM approach failed, falling back to generalized")
                return self.get_generalized_ontology()

        elif self.approach == "hybrid":
            return self.create_hybrid_ontology()

        else:  # generalized (default)
            print("🎯 Using curated generalized ontology")
            return self.get_generalized_ontology()

    def validate_ontology(self, ontology: Dict) -> Tuple[bool, List[str]]:
        """Validate ontology structure and content"""
        errors = []

        # Check required fields
        if "entities" not in ontology:
            errors.append("Missing 'entities' field")
        if "relations" not in ontology:
            errors.append("Missing 'relations' field")

        # Check data types
        if not isinstance(ontology.get("entities", []), list):
            errors.append("'entities' must be a list")
        if not isinstance(ontology.get("relations", []), list):
            errors.append("'relations' must be a list")

        # Check minimum content
        if len(ontology.get("entities", [])) < 20:
            errors.append("Too few entities (minimum 20 required)")
        if len(ontology.get("relations", [])) < 10:
            errors.append("Too few relations (minimum 10 required)")

        # Check for duplicates
        entities = ontology.get("entities", [])
        if len(entities) != len(set(entities)):
            errors.append("Duplicate entities found")

        relations = ontology.get("relations", [])
        if len(relations) != len(set(relations)):
            errors.append("Duplicate relations found")

        return len(errors) == 0, errors

    def save_ontology(self, ontology: Dict, filename: str = "generalized_ontology.json") -> Optional[str]:
        """Save ontology with comprehensive metadata"""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            filepath = self.output_dir / filename

            # Validate before saving
            is_valid, errors = self.validate_ontology(ontology)
            if not is_valid:
                print(f"❌ Ontology validation failed: {', '.join(errors)}")
                return None

            # Add technology mappings
            tech_mappings = self.create_technology_mapping(ontology)

            # Create metadata
            metadata = {
                "created_at": datetime.now().isoformat(),
                "approach": self.approach,
                "model_used": self.model_name if self.approach in ["llm", "hybrid"] else "curated",
                "version": "4.0",
                "ontology_type": "generalized_entities",
                "target_technologies": self.technologies,
                "total_entities": len(ontology["entities"]),
                "total_relations": len(ontology["relations"]),
                "total_categories": ontology.get("total_categories", 0),
                "validation_passed": is_valid,
                "description": "Generalized entity-based ontology applicable across programming technologies",
                "applicability": ontology.get("applicability", "universal")
            }

            ontology_with_metadata = {
                "metadata": metadata,
                "ontology": ontology,
                "technology_mappings": tech_mappings
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(ontology_with_metadata, f, indent=2, ensure_ascii=False)

            print(f"💾 Generalized ontology saved to: {filepath}")
            return str(filepath)

        except Exception as e:
            print(f"❌ Error saving ontology: {e}")
            return None

    def create_ontology(self) -> Optional[Dict]:
        """Main method to create generalized ontology"""
        print("=" * 70)
        print("🌐 Generalized Entity-Based Ontology Creator")
        print("=" * 70)
        print(f"🎯 Creating abstract entities applicable across technologies")
        print(f"📋 Target Technologies ({len(self.technologies)}):")
        for i, tech in enumerate(self.technologies, 1):
            print(f"   {i:2d}. {tech.upper()}")
        print(f"🔧 Approach: {self.approach.replace('_', ' ').title()}")
        print("-" * 70)

        try:
            # Generate ontology
            ontology = self.generate_ontology()

            # Save ontology
            filepath = self.save_ontology(ontology)

            if filepath:
                print(f"\n✅ Generalized Ontology Created Successfully!")
                print(f"📊 Statistics:")
                print(f"   • Total Entities: {len(ontology['entities'])}")
                print(f"   • Total Relations: {len(ontology['relations'])}")
                print(f"   • Entity Categories: {ontology.get('total_categories', 'N/A')}")
                print(f"   • Technologies Covered: {len(self.technologies)}")
                print(f"📁 Location: {filepath}")

                return ontology
            else:
                raise Exception("Failed to save ontology")

        except Exception as e:
            print(f"❌ Failed to create ontology: {e}")
            return None

    def demonstrate_mappings(self, ontology: Dict):
        """Demonstrate how generalized entities map to specific technologies"""
        print(f"\n🔗 ENTITY MAPPING EXAMPLES")
        print("-" * 50)

        tech_mappings = self.create_technology_mapping(ontology)

        # Show how same entity applies across technologies
        sample_entities = ["Component", "Function", "Property", "Structure", "Event"]

        for entity in sample_entities:
            print(f"\n📌 Entity: '{entity}' maps to:")
            for tech, mappings in tech_mappings.items():
                if entity in mappings:
                    examples = mappings[entity][:3]  # Show first 3 examples
                    examples_str = ", ".join(examples)
                    print(f"   • {tech.upper()}: {examples_str}")

        print(f"\n💡 This shows how generalized entities provide universal")
        print(f"   concepts that can represent technology-specific implementations")


def main():
    """Run generalized ontology creation"""
    print("🌐 Generalized Entity-Based Knowledge Graph Ontology Creator")
    print("=" * 65)

    # Configuration options
    approaches = {
        "1": ("generalized", "Curated generalized entities (Recommended)"),
        "2": ("llm", "LLM-generated generalized entities"),
        "3": ("hybrid", "Hybrid approach (Curated + LLM)")
    }

    print("📋 Available Approaches:")
    for key, (approach, description) in approaches.items():
        print(f"   {key}. {description}")

    # Use default approach
    selected_approach = "generalized"  # Default for reliability

    print(f"\n🎯 Using approach: {selected_approach}")
    print("-" * 65)

    # Create ontology
    creator = GeneralizedOntologyCreator(approach=selected_approach)
    ontology = creator.create_ontology()

    if ontology:
        print(f"\n📋 GENERALIZED ONTOLOGY OVERVIEW")
        print("-" * 45)

        # Show entity categories
        if "entity_categories" in ontology:
            print(f"\n🏷️  Entity Categories:")
            for category, entities in list(ontology["entity_categories"].items())[:6]:  # Show first 6 categories
                category_name = category.replace("_", " ").title()
                print(f"\n   📂 {category_name} ({len(entities)} entities):")
                for i, entity in enumerate(entities[:5], 1):  # Show first 5 entities
                    print(f"      {i}. {entity}")
                if len(entities) > 5:
                    print(f"      ... and {len(entities) - 5} more")

        print(f"\n🔗 Sample Relations:")
        for i, rel in enumerate(ontology['relations'][:12], 1):  # Show first 12 relations
            print(f"   {i:2d}. {rel}")
        if len(ontology['relations']) > 12:
            print(f"   ... and {len(ontology['relations']) - 12} more")

        # Demonstrate mappings
        creator.demonstrate_mappings(ontology)

        print(f"\n✅ Generalized Ontology Ready for Universal Knowledge Graph!")
        print(f"🔄 Benefits:")
        print(f"   • Universal applicability across all {len(creator.technologies)} technologies")
        print(f"   • Consistent entity representation")
        print(f"   • Scalable to new technologies")
        print(f"   • Language-agnostic knowledge modeling")

        print(f"\n🔄 Next Steps:")
        print(f"   1. Use entities as universal concept extractors")
        print(f"   2. Map technology-specific content to generalized entities")
        print(f"   3. Build unified knowledge graph with consistent vocabulary")
        print(f"   4. Enable cross-technology semantic search and analysis")

        return ontology
    else:
        print("❌ Generalized ontology creation failed!")
        return None


if __name__ == "__main__":
    main()