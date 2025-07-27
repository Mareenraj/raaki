"""
Step 1: Create Initial Ontology using LLM or Enhanced Fallback
"""

import json
import os
from datetime import datetime
from pathlib import Path


class OntologyCreator:
    def __init__(self, model_name="microsoft/DialoGPT-medium", output_dir="data/ontology", use_fallback=True):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.use_fallback = use_fallback
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load model and tokenizer with error handling"""
        try:
            print(f"Loading model: {self.model_name}")

            # Import here to avoid dependency issues if not needed
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            print("Model loaded successfully!")
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to enhanced ontology...")
            return False

    def get_enhanced_fallback_ontology(self):
        """Return comprehensive software engineering ontology"""
        return {
            "classes": [
                # Core Programming Concepts
                "Concept", "Technology", "Framework", "Library", "Language",

                # Software Architecture
                "Architecture", "Pattern", "Component", "Module", "Service",
                "Interface", "API", "Middleware", "Protocol",

                # Development Tools
                "Tool", "BuildTool", "PackageManager", "IDE", "Debugger",
                "Profiler", "TestingFramework", "VersionControl",

                # Data & Storage
                "Database", "DataStructure", "Collection", "Cache",
                "FileSystem", "Storage", "Repository",

                # Web & Network
                "WebFramework", "HTTPMethod", "Request", "Response",
                "Authentication", "Authorization", "Security", "Encryption",

                # Frontend Technologies
                "Frontend", "UI", "Component", "Hook", "State", "Event",
                "DOM", "CSS", "HTML", "JavaScript",

                # Backend Technologies
                "Backend", "Server", "Microservice", "Container", "Cloud",
                "Deployment", "Environment", "Configuration",

                # Development Process
                "Process", "Methodology", "Testing", "Documentation",
                "Performance", "Optimization", "Monitoring", "Logging",

                # Programming Elements
                "Object", "Class", "Method", "Function", "Variable",
                "Parameter", "ReturnType", "Exception", "Error"
            ],
            "relations": [
                # Basic Relations
                "hasConcept", "isTypeOf", "uses", "implements", "extends",
                "inherits", "contains", "belongsTo", "partOf",

                # Framework Relations
                "hasFramework", "hasLibrary", "hasFeature", "hasComponent",
                "hasModule", "hasHook", "hasMethod", "hasProperty",

                # Development Relations
                "developedBy", "usedFor", "basedOn", "buildsOn", "dependsOn",
                "requires", "supports", "manages", "configures",

                # Interaction Relations
                "calls", "invokes", "triggers", "handles", "processes",
                "exposes", "consumes", "produces", "returns", "throws",

                # Network Relations
                "communicatesWith", "connectsTo", "authenticates", "authorizes",
                "serves", "requests", "responds", "routes", "proxies",

                # Architectural Relations
                "orchestrates", "monitors", "controls", "initializes",
                "deploys", "scales", "balances", "caches", "stores",

                # Data Relations
                "stores", "retrieves", "queries", "updates", "deletes",
                "serializes", "deserializes", "validates", "transforms",

                # Development Process Relations
                "tests", "debugs", "profiles", "documents", "versions",
                "compiles", "builds", "packages", "releases"
            ]
        }

    def generate_ontology_with_model(self):
        """Generate ontology using loaded model"""
        prompt = """Create a comprehensive JSON ontology for software engineering and web development concepts.

Include classes for: programming languages, frameworks, tools, architectural patterns, data structures, and development processes.
Include relations for: inheritance, composition, usage, implementation, and interactions.

Return only valid JSON in this format:
{
  "classes": ["Class1", "Class2", ...],
  "relations": ["relation1", "relation2", ...]
}

JSON ontology:"""

        try:
            import torch

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=800,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                ontology_text = response[json_start:json_end]
                ontology = json.loads(ontology_text)

                # Validate ontology structure
                if "classes" in ontology and "relations" in ontology:
                    if isinstance(ontology["classes"], list) and isinstance(ontology["relations"], list):
                        print("✅ Generated ontology using model")
                        return ontology

            print("⚠️ Model output invalid, using fallback")
            return self.get_enhanced_fallback_ontology()

        except Exception as e:
            print(f"⚠️ Error generating with model: {e}")
            return self.get_enhanced_fallback_ontology()

    def generate_initial_ontology(self):
        """Generate initial ontology using model or fallback"""

        if self.use_fallback:
            print("🎯 Using enhanced fallback ontology")
            return self.get_enhanced_fallback_ontology()

        # Try to load and use model
        if self.load_model():
            return self.generate_ontology_with_model()
        else:
            return self.get_enhanced_fallback_ontology()

    def save_ontology(self, ontology, filename="initial_ontology.json"):
        """Save ontology to JSON file with metadata"""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            filepath = self.output_dir / filename

            ontology_with_metadata = {
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "model_used": self.model_name if not self.use_fallback else "enhanced_fallback",
                    "version": "1.1",
                    "total_classes": len(ontology["classes"]),
                    "total_relations": len(ontology["relations"]),
                    "use_fallback": self.use_fallback
                },
                "ontology": ontology
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(ontology_with_metadata, f, indent=2, ensure_ascii=False)

            print(f"💾 Ontology saved to: {filepath}")
            return filepath

        except Exception as e:
            print(f"❌ Error saving ontology: {e}")
            return None

    def create_ontology(self):
        """Main method to create and save initial ontology"""
        print("=" * 50)
        print("🚀 Step 1: Creating Initial Ontology")
        print("=" * 50)

        try:
            ontology = self.generate_initial_ontology()
            filepath = self.save_ontology(ontology)

            if filepath:
                print(f"\n✅ Initial ontology created successfully!")
                print(f"📊 Classes: {len(ontology['classes'])}")
                print(f"🔗 Relations: {len(ontology['relations'])}")
                print(f"📁 Location: {filepath}")
                return ontology
            else:
                raise Exception("Failed to save ontology")

        except Exception as e:
            print(f"❌ Failed to create ontology: {e}")
            return None

    def load_existing_ontology(self, filename="initial_ontology.json"):
        """Load existing ontology from file"""
        try:
            filepath = self.output_dir / filename
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data.get("ontology", data)  # Handle both formats
            return None
        except Exception as e:
            print(f"Error loading existing ontology: {e}")
            return None


def main():
    """Run ontology creation with enhanced output"""
    print("🎯 Knowledge Graph Ontology Creator")
    print("-" * 40)

    creator = OntologyCreator(use_fallback=True)  # Use fallback by default
    ontology = creator.create_ontology()

    if ontology:
        print(f"\n📋 ONTOLOGY SUMMARY")
        print("-" * 30)

        print(f"\n🏷️  Classes ({len(ontology['classes'])}):")
        for i, cls in enumerate(ontology['classes'][:10], 1):  # Show first 10
            print(f"   {i:2d}. {cls}")
        if len(ontology['classes']) > 10:
            print(f"   ... and {len(ontology['classes']) - 10} more")

        print(f"\n🔗 Relations ({len(ontology['relations'])}):")
        for i, rel in enumerate(ontology['relations'][:10], 1):  # Show first 10
            print(f"   {i:2d}. {rel}")
        if len(ontology['relations']) > 10:
            print(f"   ... and {len(ontology['relations']) - 10} more")

        print(f"\n✅ Ready for Step 2: Web Scraping!")
        return ontology
    else:
        print("❌ Ontology creation failed!")
        return None


if __name__ == "__main__":
    main()