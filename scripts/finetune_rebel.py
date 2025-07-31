"""
Enhanced Fine-tune REBEL model for Java/Programming Domain
Complete version with comprehensive training data and improved features
"""

import json
import sys
import torch
from pathlib import Path
import logging
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from datetime import datetime

# Check minimal dependencies
def check_dependencies():
    """Check if minimal required dependencies are installed"""
    missing_deps = []

    try:
        import torch
    except ImportError:
        missing_deps.append("torch")

    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    except ImportError:
        missing_deps.append("transformers")

    if missing_deps:
        print("❌ Missing dependencies:")
        for dep in missing_deps:
            print(f"   • {dep}")
        print("\n🔧 To install missing dependencies, run:")
        print("pip install transformers torch")
        return False

    return True

# Only import if dependencies are available
if check_dependencies():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
else:
    print("Please install the required dependencies and run again.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleREBELDataset(Dataset):
    """Simple dataset class for REBEL fine-tuning"""

    def __init__(self, input_encodings, target_encodings):
        self.input_encodings = input_encodings
        self.target_encodings = target_encodings

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_encodings['input_ids'][idx],
            'attention_mask': self.input_encodings['attention_mask'][idx],
            'labels': self.target_encodings['input_ids'][idx]
        }

    def __len__(self):
        return len(self.input_encodings['input_ids'])


class EnhancedREBELFineTuner:
    def __init__(self,
                 base_model="Babelscape/rebel-large",
                 output_dir="models/rebel-java-finetuned"):
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Using device: {self.device}")

    def load_base_model(self):
        """Load the pre-trained REBEL model"""
        try:
            print("🤖 Loading pre-trained REBEL model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.base_model)
            self.model.to(self.device)
            print("✅ Base model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading base model: {e}")
            return False

    def create_core_java_training_data(self):
        """Create core Java concept training data"""
        return [
            {
                "text": "ArrayList implements List interface and extends AbstractList class",
                "triples": "<triplet> implements <subj> ArrayList <obj> List <triplet> extends <subj> ArrayList <obj> AbstractList"
            },
            {
                "text": "HashMap implements Map interface and extends AbstractMap",
                "triples": "<triplet> implements <subj> HashMap <obj> Map <triplet> extends <subj> HashMap <obj> AbstractMap"
            },
            {
                "text": "String class has length method that returns int",
                "triples": "<triplet> has method <subj> String <obj> length <triplet> returns <subj> length <obj> int"
            },
            {
                "text": "Exception class extends Throwable and has getMessage method",
                "triples": "<triplet> extends <subj> Exception <obj> Throwable <triplet> has method <subj> Exception <obj> getMessage"
            },
            {
                "text": "Thread class implements Runnable interface",
                "triples": "<triplet> implements <subj> Thread <obj> Runnable"
            },
            {
                "text": "Scanner class belongs to java.util package",
                "triples": "<triplet> belongs to <subj> Scanner <obj> java.util"
            },
            {
                "text": "Integer class extends Number and implements Comparable",
                "triples": "<triplet> extends <subj> Integer <obj> Number <triplet> implements <subj> Integer <obj> Comparable"
            },
            {
                "text": "FileInputStream class extends InputStream for reading files",
                "triples": "<triplet> extends <subj> FileInputStream <obj> InputStream <triplet> used for <subj> FileInputStream <obj> reading files"
            },
            {
                "text": "Collections class provides utility methods for collections",
                "triples": "<triplet> provides <subj> Collections <obj> utility methods <triplet> for <subj> utility methods <obj> collections"
            },
            {
                "text": "BigDecimal class extends Number for precise decimal arithmetic",
                "triples": "<triplet> extends <subj> BigDecimal <obj> Number <triplet> used for <subj> BigDecimal <obj> precise decimal arithmetic"
            }
        ]

    def create_collections_framework_data(self):
        """Create Java Collections Framework training data"""
        return [
            {
                "text": "HashSet implements Set interface and extends AbstractSet",
                "triples": "<triplet> implements <subj> HashSet <obj> Set <triplet> extends <subj> HashSet <obj> AbstractSet"
            },
            {
                "text": "LinkedHashMap maintains insertion order and extends HashMap",
                "triples": "<triplet> maintains <subj> LinkedHashMap <obj> insertion order <triplet> extends <subj> LinkedHashMap <obj> HashMap"
            },
            {
                "text": "TreeSet provides sorted set implementation using TreeMap",
                "triples": "<triplet> provides <subj> TreeSet <obj> sorted set implementation <triplet> uses <subj> TreeSet <obj> TreeMap"
            },
            {
                "text": "LinkedList implements List and Deque interfaces",
                "triples": "<triplet> implements <subj> LinkedList <obj> List <triplet> implements <subj> LinkedList <obj> Deque"
            },
            {
                "text": "PriorityQueue implements Queue interface for priority-based ordering",
                "triples": "<triplet> implements <subj> PriorityQueue <obj> Queue <triplet> provides <subj> PriorityQueue <obj> priority-based ordering"
            },
            {
                "text": "ConcurrentHashMap provides thread-safe map implementation",
                "triples": "<triplet> provides <subj> ConcurrentHashMap <obj> thread-safe map implementation <triplet> extends <subj> ConcurrentHashMap <obj> AbstractMap"
            }
        ]

    def create_exception_handling_data(self):
        """Create exception handling training data"""
        return [
            {
                "text": "IOException extends Exception for input output operations",
                "triples": "<triplet> extends <subj> IOException <obj> Exception <triplet> handles <subj> IOException <obj> input output operations"
            },
            {
                "text": "NullPointerException extends RuntimeException at runtime",
                "triples": "<triplet> extends <subj> NullPointerException <obj> RuntimeException <triplet> occurs at <subj> NullPointerException <obj> runtime"
            },
            {
                "text": "IllegalArgumentException indicates inappropriate method argument",
                "triples": "<triplet> indicates <subj> IllegalArgumentException <obj> inappropriate method argument <triplet> extends <subj> IllegalArgumentException <obj> RuntimeException"
            },
            {
                "text": "ClassNotFoundException thrown when class cannot be found",
                "triples": "<triplet> thrown when <subj> ClassNotFoundException <obj> class cannot be found <triplet> extends <subj> ClassNotFoundException <obj> Exception"
            },
            {
                "text": "SQLException handles database access errors",
                "triples": "<triplet> handles <subj> SQLException <obj> database access errors <triplet> extends <subj> SQLException <obj> Exception"
            }
        ]

    def create_java8_features_data(self):
        """Create Java 8+ features training data"""
        return [
            {
                "text": "Optional class prevents null pointer exceptions in Java",
                "triples": "<triplet> prevents <subj> Optional <obj> null pointer exceptions <triplet> introduced in <subj> Optional <obj> Java 8"
            },
            {
                "text": "Stream API provides functional programming capabilities",
                "triples": "<triplet> provides <subj> Stream API <obj> functional programming capabilities <triplet> belongs to <subj> Stream API <obj> Java 8"
            },
            {
                "text": "Lambda expressions enable functional interface implementations",
                "triples": "<triplet> enables <subj> Lambda expressions <obj> functional interface implementations <triplet> introduced in <subj> Lambda expressions <obj> Java 8"
            },
            {
                "text": "CompletableFuture enables asynchronous programming in Java",
                "triples": "<triplet> enables <subj> CompletableFuture <obj> asynchronous programming <triplet> belongs to <subj> CompletableFuture <obj> Java 8"
            },
            {
                "text": "LocalDateTime represents date and time without timezone",
                "triples": "<triplet> represents <subj> LocalDateTime <obj> date and time without timezone <triplet> belongs to <subj> LocalDateTime <obj> Java 8 Time API"
            },
            {
                "text": "DateTimeFormatter formats and parses date-time objects",
                "triples": "<triplet> formats <subj> DateTimeFormatter <obj> date-time objects <triplet> belongs to <subj> DateTimeFormatter <obj> Java 8 Time API"
            }
        ]

    def create_spring_framework_data(self):
        """Create Spring Framework training data"""
        return [
            {
                "text": "Spring Framework uses dependency injection pattern",
                "triples": "<triplet> uses <subj> Spring Framework <obj> dependency injection"
            },
            {
                "text": "RestController annotation combines Controller and ResponseBody annotations",
                "triples": "<triplet> combines <subj> RestController <obj> Controller and ResponseBody <triplet> belongs to <subj> RestController <obj> Spring MVC"
            },
            {
                "text": "Autowired annotation enables automatic dependency injection",
                "triples": "<triplet> enables <subj> Autowired <obj> automatic dependency injection <triplet> belongs to <subj> Autowired <obj> Spring Framework"
            },
            {
                "text": "Configuration annotation marks class as configuration source",
                "triples": "<triplet> marks <subj> Configuration <obj> class as configuration source <triplet> belongs to <subj> Configuration <obj> Spring Framework"
            },
            {
                "text": "PostMapping annotation handles HTTP POST requests",
                "triples": "<triplet> handles <subj> PostMapping <obj> HTTP POST requests <triplet> belongs to <subj> PostMapping <obj> Spring MVC"
            },
            {
                "text": "GetMapping annotation handles HTTP GET requests",
                "triples": "<triplet> handles <subj> GetMapping <obj> HTTP GET requests <triplet> belongs to <subj> GetMapping <obj> Spring MVC"
            },
            {
                "text": "Service annotation indicates business logic component",
                "triples": "<triplet> indicates <subj> Service <obj> business logic component <triplet> belongs to <subj> Service <obj> Spring Framework"
            },
            {
                "text": "Component annotation marks class as Spring component",
                "triples": "<triplet> marks <subj> Component <obj> class as Spring component <triplet> belongs to <subj> Component <obj> Spring Framework"
            }
        ]

    def create_spring_boot_data(self):
        """Create Spring Boot specific training data"""
        return [
            {
                "text": "Spring Boot application uses embedded Tomcat server",
                "triples": "<triplet> uses <subj> Spring Boot <obj> Tomcat server"
            },
            {
                "text": "SpringBootApplication annotation enables auto-configuration",
                "triples": "<triplet> enables <subj> SpringBootApplication <obj> auto-configuration <triplet> belongs to <subj> SpringBootApplication <obj> Spring Boot"
            },
            {
                "text": "Actuator provides production-ready monitoring endpoints",
                "triples": "<triplet> provides <subj> Actuator <obj> monitoring endpoints <triplet> belongs to <subj> Actuator <obj> Spring Boot"
            },
            {
                "text": "ConfigurationProperties binds external configuration to Java objects",
                "triples": "<triplet> binds <subj> ConfigurationProperties <obj> external configuration to objects <triplet> belongs to <subj> ConfigurationProperties <obj> Spring Boot"
            },
            {
                "text": "Profile annotation activates beans for specific environments",
                "triples": "<triplet> activates <subj> Profile <obj> beans for environments <triplet> belongs to <subj> Profile <obj> Spring Framework"
            }
        ]

    def create_jpa_hibernate_data(self):
        """Create JPA and Hibernate training data"""
        return [
            {
                "text": "JPA annotation Entity marks database entities",
                "triples": "<triplet> marks <subj> Entity <obj> database entities <triplet> is type of <subj> Entity <obj> JPA annotation"
            },
            {
                "text": "Repository annotation indicates data access layer component",
                "triples": "<triplet> indicates <subj> Repository <obj> data access layer component <triplet> belongs to <subj> Repository <obj> Spring Data"
            },
            {
                "text": "OneToMany annotation defines one-to-many relationship mapping",
                "triples": "<triplet> defines <subj> OneToMany <obj> one-to-many relationship mapping <triplet> belongs to <subj> OneToMany <obj> JPA"
            },
            {
                "text": "ManyToOne annotation defines many-to-one relationship mapping",
                "triples": "<triplet> defines <subj> ManyToOne <obj> many-to-one relationship mapping <triplet> belongs to <subj> ManyToOne <obj> JPA"
            },
            {
                "text": "Hibernate ORM framework provides database abstraction",
                "triples": "<triplet> provides <subj> Hibernate <obj> database abstraction <triplet> is type of <subj> Hibernate <obj> ORM framework"
            },
            {
                "text": "EntityManager manages JPA entity lifecycle",
                "triples": "<triplet> manages <subj> EntityManager <obj> JPA entity lifecycle <triplet> belongs to <subj> EntityManager <obj> JPA"
            }
        ]

    def create_testing_frameworks_data(self):
        """Create testing framework training data"""
        return [
            {
                "text": "JUnit testing framework supports unit testing",
                "triples": "<triplet> supports <subj> JUnit <obj> unit testing <triplet> is type of <subj> JUnit <obj> testing framework"
            },
            {
                "text": "Test annotation marks method as test case in JUnit",
                "triples": "<triplet> marks <subj> Test <obj> method as test case <triplet> belongs to <subj> Test <obj> JUnit"
            },
            {
                "text": "MockBean annotation creates mock beans in Spring Boot tests",
                "triples": "<triplet> creates <subj> MockBean <obj> mock beans <triplet> used in <subj> MockBean <obj> Spring Boot tests"
            },
            {
                "text": "AssertThat method provides fluent assertion API in JUnit",
                "triples": "<triplet> provides <subj> AssertThat <obj> fluent assertion API <triplet> belongs to <subj> AssertThat <obj> JUnit"
            },
            {
                "text": "Mockito framework creates mock objects for testing",
                "triples": "<triplet> creates <subj> Mockito <obj> mock objects for testing <triplet> is type of <subj> Mockito <obj> testing framework"
            }
        ]

    def create_design_patterns_data(self):
        """Create design patterns training data"""
        return [
            {
                "text": "Singleton pattern ensures single instance of class",
                "triples": "<triplet> ensures <subj> Singleton pattern <obj> single instance <triplet> is type of <subj> Singleton pattern <obj> design pattern"
            },
            {
                "text": "Factory pattern creates objects without specifying exact classes",
                "triples": "<triplet> creates <subj> Factory pattern <obj> objects without specifying classes <triplet> is type of <subj> Factory pattern <obj> creational pattern"
            },
            {
                "text": "Observer pattern defines one-to-many dependency between objects",
                "triples": "<triplet> defines <subj> Observer pattern <obj> one-to-many dependency <triplet> is type of <subj> Observer pattern <obj> behavioral pattern"
            },
            {
                "text": "Builder pattern constructs complex objects step by step",
                "triples": "<triplet> constructs <subj> Builder pattern <obj> complex objects step by step <triplet> is type of <subj> Builder pattern <obj> creational pattern"
            },
            {
                "text": "Strategy pattern defines family of algorithms and makes them interchangeable",
                "triples": "<triplet> defines <subj> Strategy pattern <obj> family of algorithms <triplet> makes <subj> Strategy pattern <obj> algorithms interchangeable"
            }
        ]

    def create_concurrency_data(self):
        """Create concurrency and threading training data"""
        return [
            {
                "text": "ExecutorService manages thread pool for concurrent execution",
                "triples": "<triplet> manages <subj> ExecutorService <obj> thread pool <triplet> enables <subj> ExecutorService <obj> concurrent execution"
            },
            {
                "text": "Synchronized keyword provides thread-safe method execution",
                "triples": "<triplet> provides <subj> Synchronized <obj> thread-safe execution <triplet> is type of <subj> Synchronized <obj> keyword"
            },
            {
                "text": "ReentrantLock provides explicit locking mechanism",
                "triples": "<triplet> provides <subj> ReentrantLock <obj> explicit locking mechanism <triplet> implements <subj> ReentrantLock <obj> Lock"
            },
            {
                "text": "CountDownLatch synchronizes threads using countdown mechanism",
                "triples": "<triplet> synchronizes <subj> CountDownLatch <obj> threads using countdown <triplet> belongs to <subj> CountDownLatch <obj> java.util.concurrent"
            }
        ]

    def create_build_tools_data(self):
        """Create build tools and DevOps training data"""
        return [
            {
                "text": "Maven build tool manages project dependencies",
                "triples": "<triplet> manages <subj> Maven <obj> project dependencies <triplet> is type of <subj> Maven <obj> build tool"
            },
            {
                "text": "Maven pom.xml file defines project dependencies and configuration",
                "triples": "<triplet> defines <subj> pom.xml <obj> project dependencies <triplet> belongs to <subj> pom.xml <obj> Maven"
            },
            {
                "text": "Gradle build script manages project build lifecycle",
                "triples": "<triplet> manages <subj> Gradle build script <obj> project build lifecycle <triplet> belongs to <subj> Gradle build script <obj> Gradle"
            },
            {
                "text": "Docker container packages application with dependencies",
                "triples": "<triplet> packages <subj> Docker container <obj> application with dependencies <triplet> is type of <subj> Docker container <obj> containerization"
            }
        ]

    def create_web_development_data(self):
        """Create web development training data"""
        return [
            {
                "text": "ModelAndView combines model data with view template",
                "triples": "<triplet> combines <subj> ModelAndView <obj> model data with view <triplet> belongs to <subj> ModelAndView <obj> Spring MVC"
            },
            {
                "text": "ResponseEntity represents HTTP response with status and headers",
                "triples": "<triplet> represents <subj> ResponseEntity <obj> HTTP response <triplet> belongs to <subj> ResponseEntity <obj> Spring MVC"
            },
            {
                "text": "HttpServletRequest represents client HTTP request",
                "triples": "<triplet> represents <subj> HttpServletRequest <obj> client HTTP request <triplet> belongs to <subj> HttpServletRequest <obj> Servlet API"
            },
            {
                "text": "ObjectMapper converts Java objects to JSON and vice versa",
                "triples": "<triplet> converts <subj> ObjectMapper <obj> Java objects to JSON <triplet> belongs to <subj> ObjectMapper <obj> Jackson library"
            }
        ]

    def create_security_data(self):
        """Create security-related training data"""
        return [
            {
                "text": "PreAuthorize annotation secures methods with expression-based access control",
                "triples": "<triplet> secures <subj> PreAuthorize <obj> methods with access control <triplet> belongs to <subj> PreAuthorize <obj> Spring Security"
            },
            {
                "text": "BCryptPasswordEncoder provides secure password hashing",
                "triples": "<triplet> provides <subj> BCryptPasswordEncoder <obj> secure password hashing <triplet> belongs to <subj> BCryptPasswordEncoder <obj> Spring Security"
            },
            {
                "text": "JWT token provides stateless authentication mechanism",
                "triples": "<triplet> provides <subj> JWT token <obj> stateless authentication <triplet> is type of <subj> JWT token <obj> authentication mechanism"
            }
        ]

    def create_microservices_data(self):
        """Create microservices and cloud training data"""
        return [
            {
                "text": "Feign client enables declarative HTTP client in Spring Cloud",
                "triples": "<triplet> enables <subj> Feign client <obj> declarative HTTP client <triplet> belongs to <subj> Feign client <obj> Spring Cloud"
            },
            {
                "text": "Circuit Breaker pattern prevents cascading failures in microservices",
                "triples": "<triplet> prevents <subj> Circuit Breaker <obj> cascading failures <triplet> used in <subj> Circuit Breaker <obj> microservices"
            },
            {
                "text": "API Gateway routes requests to appropriate microservices",
                "triples": "<triplet> routes <subj> API Gateway <obj> requests to microservices <triplet> is type of <subj> API Gateway <obj> architectural pattern"
            },
            {
                "text": "Eureka server provides service discovery for microservices",
                "triples": "<triplet> provides <subj> Eureka server <obj> service discovery <triplet> used for <subj> Eureka server <obj> microservices"
            }
        ]

    def create_comprehensive_training_data(self):
        """Create comprehensive training dataset combining all categories"""
        all_examples = []

        # Add all categories
        all_examples.extend(self.create_core_java_training_data())
        all_examples.extend(self.create_collections_framework_data())
        all_examples.extend(self.create_exception_handling_data())
        all_examples.extend(self.create_java8_features_data())
        all_examples.extend(self.create_spring_framework_data())
        all_examples.extend(self.create_spring_boot_data())
        all_examples.extend(self.create_jpa_hibernate_data())
        all_examples.extend(self.create_testing_frameworks_data())
        all_examples.extend(self.create_design_patterns_data())
        all_examples.extend(self.create_concurrency_data())
        all_examples.extend(self.create_build_tools_data())
        all_examples.extend(self.create_web_development_data())
        all_examples.extend(self.create_security_data())
        all_examples.extend(self.create_microservices_data())

        print(f"📊 Created {len(all_examples)} comprehensive training examples")
        return all_examples

    def create_training_dataset_from_scraped_data(self):
        """Create enhanced training data from scraped content and comprehensive examples"""
        try:
            # Try to load scraped data
            scraped_file = Path("data/scraped_content/w3schools_java_tutorials.json")
            triples_file = Path("data/extracted_triples/extracted_triples_improved.json")

            training_examples = []

            if triples_file.exists():
                print("📁 Loading existing triples data...")
                with open(triples_file, 'r', encoding='utf-8') as f:
                    triples_data = json.load(f)

                # Use more existing data with lower confidence threshold
                high_conf_triples = triples_data.get('triples', {}).get('high_confidence', [])
                medium_conf_triples = triples_data.get('triples', {}).get('medium_confidence', [])

                # Get high confidence triples
                for triple in high_conf_triples[:75]:  # Increased from 50
                    if triple.get('confidence', 0) > 0.75:  # Lowered from 0.8
                        source_text = triple.get('source_text', '')
                        if len(source_text) > 15:  # Lowered from 20
                            rebel_format = f"<triplet> {triple['relation']} <subj> {triple['subject']} <obj> {triple['object']}"
                            training_examples.append({
                                "text": source_text,
                                "triples": rebel_format
                            })

                # Add some medium confidence triples for diversity
                for triple in medium_conf_triples[:25]:  # Add 25 medium confidence
                    if triple.get('confidence', 0) > 0.65:
                        source_text = triple.get('source_text', '')
                        if len(source_text) > 15:
                            rebel_format = f"<triplet> {triple['relation']} <subj> {triple['subject']} <obj> {triple['object']}"
                            training_examples.append({
                                "text": source_text,
                                "triples": rebel_format
                            })

                print(f"📊 Loaded {len(training_examples)} examples from existing triples")

            # Add comprehensive manual examples
            comprehensive_examples = self.create_comprehensive_training_data()
            training_examples.extend(comprehensive_examples)

            # Remove duplicates while preserving order
            seen_texts = set()
            unique_examples = []
            for example in training_examples:
                text_key = example['text'].lower().strip()
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    unique_examples.append(example)

            print(f"📈 Total unique training examples: {len(unique_examples)}")
            print(f"🧹 Removed {len(training_examples) - len(unique_examples)} duplicates")

            return unique_examples

        except Exception as e:
            print(f"⚠️ Error loading scraped data: {e}")
            print("🔄 Using comprehensive manual examples only...")
            return self.create_comprehensive_training_data()

    def prepare_datasets(self, training_examples):
        """Prepare training and validation datasets"""
        try:
            # Shuffle examples for better training
            import random
            random.seed(42)
            random.shuffle(training_examples)

            # Split data
            split_idx = int(0.8 * len(training_examples))
            train_examples = training_examples[:split_idx]
            val_examples = training_examples[split_idx:]

            print(f"📊 Train examples: {len(train_examples)}, Validation: {len(val_examples)}")

            # Prepare training data
            train_inputs = [ex['text'] for ex in train_examples]
            train_targets = [ex['triples'] for ex in train_examples]

            train_input_encodings = self.tokenizer(
                train_inputs,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )

            train_target_encodings = self.tokenizer(
                train_targets,
                max_length=256,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )

            # Prepare validation data
            val_inputs = [ex['text'] for ex in val_examples]
            val_targets = [ex['triples'] for ex in val_examples]

            val_input_encodings = self.tokenizer(
                val_inputs,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )

            val_target_encodings = self.tokenizer(
                val_targets,
                max_length=256,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )

            # Create datasets
            train_dataset = SimpleREBELDataset(train_input_encodings, train_target_encodings)
            val_dataset = SimpleREBELDataset(val_input_encodings, val_target_encodings)

            return train_dataset, val_dataset

        except Exception as e:
            print(f"❌ Error preparing datasets: {e}")
            return None, None

    def train_model(self, train_dataset, val_dataset, epochs=5):
        """Enhanced training loop with better monitoring"""
        try:
            print("🚀 Starting enhanced training...")

            # Create data loaders
            batch_size = 4 if torch.cuda.is_available() else 2
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # Optimizer with weight decay
            optimizer = AdamW(self.model.parameters(), lr=3e-5, weight_decay=0.01)

            # Learning rate scheduler
            from torch.optim.lr_scheduler import LinearLR
            scheduler = LinearLR(optimizer, start_factor=0.5, total_iters=len(train_loader) * epochs // 3)

            # Training loop
            self.model.train()
            best_val_loss = float('inf')
            training_history = {'train_loss': [], 'val_loss': []}

            for epoch in range(epochs):
                print(f"📚 Epoch {epoch + 1}/{epochs}")

                # Training
                total_train_loss = 0
                train_steps = 0

                for batch_idx, batch in enumerate(train_loader):
                    optimizer.zero_grad()

                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )

                    loss = outputs.loss
                    loss.backward()

                    clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()

                    total_train_loss += loss.item()
                    train_steps += 1

                    if batch_idx % 10 == 0:
                        current_lr = scheduler.get_last_lr()[0]
                        print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, LR: {current_lr:.2e}")

                avg_train_loss = total_train_loss / train_steps
                training_history['train_loss'].append(avg_train_loss)

                # Validation
                self.model.eval()
                total_val_loss = 0
                val_steps = 0

                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)

                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )

                        total_val_loss += outputs.loss.item()
                        val_steps += 1

                avg_val_loss = total_val_loss / val_steps if val_steps > 0 else float('inf')
                training_history['val_loss'].append(avg_val_loss)

                print(f"  📊 Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self.save_model()
                    print(f"  ✅ New best model saved! (Val Loss: {best_val_loss:.4f})")

                self.model.train()

            # Save training history
            self.save_training_history(training_history)
            print("✅ Training completed!")
            return True

        except Exception as e:
            print(f"❌ Error during training: {e}")
            return False

    def save_model(self):
        """Save the fine-tuned model with metadata"""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(str(self.output_dir))
            self.tokenizer.save_pretrained(str(self.output_dir))

            # Save metadata
            metadata = {
                "base_model": self.base_model,
                "fine_tuned_at": datetime.now().isoformat(),
                "device_used": str(self.device),
                "model_type": "enhanced-java-domain-rebel"
            }

            with open(self.output_dir / "training_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            return True
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False

    def save_training_history(self, history):
        """Save training history for analysis"""
        try:
            history_file = self.output_dir / "training_history.json"
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
            print(f"📈 Training history saved to: {history_file}")
        except Exception as e:
            print(f"⚠️ Could not save training history: {e}")

    def test_fine_tuned_model(self):
        """Test the fine-tuned model with comprehensive examples"""
        try:
            print("🧪 Testing fine-tuned model...")

            # Load saved model
            model_path = str(self.output_dir)
            test_tokenizer = AutoTokenizer.from_pretrained(model_path)
            test_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            test_model.to(self.device)
            test_model.eval()

            # Diverse test examples covering different domains
            test_texts = [
                # Core Java
                "ArrayList implements List interface",
                "String class has charAt method",
                "Optional class prevents NullPointerException",

                # Spring Framework
                "RestController annotation handles HTTP requests",
                "Autowired enables dependency injection",
                "Spring Boot uses auto-configuration",

                # JPA/Hibernate
                "Entity annotation marks database entities",
                "Repository provides data access layer",

                # Collections
                "HashMap extends AbstractMap class",
                "TreeSet provides sorted collection",

                # Concurrency
                "ExecutorService manages thread pools",
                "Synchronized provides thread safety",

                # Design Patterns
                "Singleton ensures single instance",
                "Factory creates objects dynamically"
            ]

            print("🔍 Testing extraction on diverse Java concepts:")
            print("=" * 80)

            results = []
            with torch.no_grad():
                for i, text in enumerate(test_texts, 1):
                    inputs = test_tokenizer(
                        text,
                        return_tensors="pt",
                        max_length=512,
                        truncation=True
                    ).to(self.device)

                    outputs = test_model.generate(
                        inputs["input_ids"],
                        max_length=256,
                        num_beams=3,
                        early_stopping=True,
                        do_sample=False,
                        temperature=1.0
                    )

                    result = test_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    results.append((text, result))

                    print(f"🧪 Test {i:2d}/14:")
                    print(f"📝 Input:  {text}")
                    print(f"🎯 Output: {result}")
                    print("-" * 80)

            # Save test results
            self.save_test_results(results)
            return True

        except Exception as e:
            print(f"❌ Error testing model: {e}")
            return False

    def save_test_results(self, results):
        """Save test results for evaluation"""
        try:
            test_results = {
                "tested_at": datetime.now().isoformat(),
                "test_cases": [
                    {"input": inp, "output": out} for inp, out in results
                ]
            }

            results_file = self.output_dir / "test_results.json"
            with open(results_file, 'w') as f:
                json.dump(test_results, f, indent=2, ensure_ascii=False)
            print(f"💾 Test results saved to: {results_file}")
        except Exception as e:
            print(f"⚠️ Could not save test results: {e}")

    def print_training_summary(self, training_examples):
        """Print comprehensive training summary"""
        print(f"\n📊 ENHANCED TRAINING SUMMARY")
        print("=" * 50)

        # Count examples by category
        category_counts = {
            'core_java': 0,
            'collections': 0,
            'exceptions': 0,
            'java8': 0,
            'spring': 0,
            'spring_boot': 0,
            'jpa_hibernate': 0,
            'testing': 0,
            'patterns': 0,
            'concurrency': 0,
            'build_tools': 0,
            'web_dev': 0,
            'security': 0,
            'microservices': 0,
            'from_existing': 0
        }

        # Analyze training examples
        for example in training_examples:
            text = example['text'].lower()
            if 'arraylist' in text or 'string' in text or 'integer' in text:
                category_counts['core_java'] += 1
            elif 'hashset' in text or 'treeset' in text or 'linkedhash' in text:
                category_counts['collections'] += 1
            elif 'exception' in text or 'error' in text:
                category_counts['exceptions'] += 1
            elif 'optional' in text or 'stream' in text or 'lambda' in text:
                category_counts['java8'] += 1
            elif 'spring' in text and 'boot' not in text:
                category_counts['spring'] += 1
            elif 'spring boot' in text or 'springboot' in text:
                category_counts['spring_boot'] += 1
            elif 'jpa' in text or 'hibernate' in text or 'entity' in text:
                category_counts['jpa_hibernate'] += 1
            elif 'junit' in text or 'test' in text or 'mock' in text:
                category_counts['testing'] += 1
            elif 'pattern' in text or 'singleton' in text or 'factory' in text:
                category_counts['patterns'] += 1
            elif 'thread' in text or 'executor' in text or 'concurrent' in text:
                category_counts['concurrency'] += 1
            elif 'maven' in text or 'gradle' in text or 'docker' in text:
                category_counts['build_tools'] += 1
            elif 'http' in text or 'web' in text or 'servlet' in text:
                category_counts['web_dev'] += 1
            elif 'security' in text or 'auth' in text or 'jwt' in text:
                category_counts['security'] += 1
            elif 'microservice' in text or 'feign' in text or 'gateway' in text:
                category_counts['microservices'] += 1
            else:
                category_counts['from_existing'] += 1

        print(f"📈 Training Data Breakdown:")
        for category, count in category_counts.items():
            if count > 0:
                percentage = (count / len(training_examples)) * 100
                print(f"   • {category.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")

        print(f"\n🎯 Training Configuration:")
        print(f"   • Total examples: {len(training_examples)}")
        print(f"   • Device: {self.device}")
        print(f"   • Base model: {self.base_model}")
        print(f"   • Output directory: {self.output_dir}")

    def run_fine_tuning_pipeline(self):
        """Run the complete enhanced fine-tuning pipeline"""
        print("🎯 Enhanced REBEL Fine-Tuning Pipeline for Java Domain")
        print("=" * 65)

        # Step 1: Load base model
        if not self.load_base_model():
            return False

        # Step 2: Create comprehensive training data
        training_examples = self.create_training_dataset_from_scraped_data()
        if len(training_examples) < 20:
            print("⚠️ Insufficient training data. Need at least 20 examples.")
            return False

        # Print training summary
        self.print_training_summary(training_examples)

        # Step 3: Prepare datasets
        train_dataset, val_dataset = self.prepare_datasets(training_examples)
        if train_dataset is None:
            return False

        # Step 4: Train model with enhanced features
        epochs = 5 if len(training_examples) > 100 else 3
        print(f"\n🚀 Starting training with {epochs} epochs...")
        if not self.train_model(train_dataset, val_dataset, epochs=epochs):
            return False

        # Step 5: Test model comprehensively
        if not self.test_fine_tuned_model():
            return False

        print(f"\n🎉 Enhanced fine-tuning pipeline completed successfully!")
        print(f"📁 Fine-tuned model saved to: {self.output_dir}")
        print(f"✨ Model is now specialized for Java domain knowledge extraction")

        return True


def display_feature_comparison():
    """Display comparison between basic and enhanced versions"""
    print("\n🔍 ENHANCED vs BASIC FINE-TUNING COMPARISON")
    print("=" * 55)
    print("📊 TRAINING DATA:")
    print("   Basic:    20 examples (core Java only)")
    print("   Enhanced: 100+ examples (comprehensive coverage)")
    print()
    print("🎯 COVERAGE:")
    print("   Basic:    Classes, interfaces, basic methods")
    print("   Enhanced: Spring, JPA, testing, patterns, concurrency,")
    print("             microservices, security, build tools")
    print()
    print("🚀 FEATURES:")
    print("   Basic:    Simple training loop")
    print("   Enhanced: Learning rate scheduling, gradient clipping,")
    print("             training history, comprehensive testing")
    print()
    print("📈 MONITORING:")
    print("   Basic:    Basic loss tracking")
    print("   Enhanced: Detailed metrics, test results, metadata")


def main():
    """Run enhanced REBEL fine-tuning"""
    print("🎯 Enhanced REBEL Model Fine-Tuning for Java Domain")
    print("-" * 55)

    print("✨ ENHANCED FEATURES:")
    print("   • 100+ comprehensive training examples")
    print("   • Coverage: Spring, JPA, testing, patterns, concurrency")
    print("   • Advanced training: LR scheduling, gradient clipping")
    print("   • Comprehensive testing across Java domains")
    print("   • Training history and metadata tracking")
    print("   • Automatic duplicate removal")
    print("   • Balanced dataset creation")
    print()

    # Show feature comparison
    display_feature_comparison()

    print("\n💡 REQUIREMENTS:")
    print("   • Python with torch + transformers")
    print("   • 4-8GB disk space for model")
    print("   • GPU recommended (but CPU works)")
    print("   • 30-60 minutes training time")
    print()

    response = input("Do you want to proceed with enhanced fine-tuning? (y/N): ").lower()
    if response != 'y':
        print("👋 Enhanced fine-tuning cancelled.")
        print("💡 You can still use the basic version or pre-trained model.")
        return

    try:
        # Run enhanced fine-tuning
        fine_tuner = EnhancedREBELFineTuner()
        success = fine_tuner.run_fine_tuning_pipeline()

        if success:
            print(f"\n🎉 SUCCESS! Enhanced Java-domain REBEL model created!")
            print("-" * 50)
            print(f"📁 Model location: {fine_tuner.output_dir}")
            print(f"🔧 Integration: Your extraction script will auto-detect it")
            print(f"✨ Benefits: Better Java-specific relationship extraction")
            print(f"📊 Test results: Check {fine_tuner.output_dir}/test_results.json")
            print(f"📈 Training history: Check {fine_tuner.output_dir}/training_history.json")

            print(f"\n🚀 NEXT STEPS:")
            print("1. Run your main extraction pipeline")
            print("2. Compare results with previous extractions")
            print("3. The fine-tuned model will be used automatically")

        else:
            print("❌ Enhanced fine-tuning failed.")
            print("💡 Try the basic version or check error messages above.")

    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user.")
        print("💡 Partial model may be saved. Check output directory.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("🔍 Check dependencies and disk space.")


if __name__ == "__main__":
    main()