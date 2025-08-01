import requests
from bs4 import BeautifulSoup
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from collections import defaultdict
from create_ontology import GeneralizedOntologyCreator  # Assuming this is in a separate file

class EnhancedWebScraper:
    def __init__(
        self,
        technologies: List[str] = [
            "html", "css", "js", "python", "java", "sql",
            "bootstrap", "jquery", "json", "ajax", "xml", "api",
            "php", "cs", "nodejs", "react", "typescript"
        ],
        output_dir: str = "data/scraped_content",
        ontology_approach: str = "generalized"
    ):
        self.base_url = "https://www.w3schools.com/"
        self.output_dir = Path(output_dir)
        self.technologies = technologies
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

        # Initialize ontology creator
        self.ontology_creator = GeneralizedOntologyCreator(approach=ontology_approach)
        self.ontology = self.ontology_creator.create_ontology()
        self.tech_mappings = self.ontology_creator.create_technology_mapping(self.ontology)

        # Track scraped URLs per technology
        self.scraped_urls = {tech: self._load_scraped_urls(tech) for tech in technologies}

    def _load_scraped_urls(self, technology: str) -> set:
        """Load previously scraped URLs for a specific technology"""
        try:
            url_file = self.output_dir / f"{technology}_scraped_urls.json"
            if url_file.exists():
                with open(url_file, 'r', encoding='utf-8') as f:
                    return set(json.load(f))
            return set()
        except Exception as e:
            print(f"Error loading scraped URLs for {technology}: {e}")
            return set()

    def _save_scraped_urls(self, technology: str):
        """Save scraped URLs for a specific technology"""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            url_file = self.output_dir / f"{technology}_scraped_urls.json"
            with open(url_file, 'w', encoding='utf-8') as f:
                json.dump(list(self.scraped_urls[technology]), f, indent=2)
        except Exception as e:
            print(f"Error saving scraped URLs for {technology}: {e}")

    def get_tutorial_urls(self, technology: str) -> List[Dict]:
        """Extract tutorial URLs for a specific technology"""
        try:
            main_url = f"{self.base_url}{technology}/default.asp"
            print(f"Fetching {technology} tutorial URLs from {main_url}")
            response = self.session.get(main_url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            tutorial_links = []

            # Flexible URL pattern matching
            prefix = f"{technology}_"
            suffixes = [".asp", ".php"]  # Support both .asp and .php

            # Find tutorial links
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if href.startswith(prefix) and any(href.endswith(suffix) for suffix in suffixes):
                    full_url = f"{self.base_url}{technology}/{href}"
                    if full_url not in self.scraped_urls[technology]:  # Skip already scraped
                        title = link.get_text().strip()
                        tutorial_links.append({
                            "url": full_url,
                            "title": title,
                            "filename": href,
                            "technology": technology
                        })

            # Remove duplicates
            seen_urls = set()
            unique_links = []
            for link in tutorial_links:
                if link["url"] not in seen_urls:
                    seen_urls.add(link["url"])
                    unique_links.append(link)

            print(f"Found {len(unique_links)} new {technology} tutorial URLs")
            return unique_links

        except Exception as e:
            print(f"Error getting {technology} tutorial URLs: {e}")
            return []

    def map_content_to_ontology(self, content: str, technology: str) -> Dict[str, List[str]]:
        """Map content to ontology entities"""
        entity_matches = defaultdict(list)

        # Get relevant mappings for the technology
        tech_mapping = self.tech_mappings.get(technology, {})

        # Simple keyword-based matching
        for entity in self.ontology["entities"]:
            if entity.lower() in content.lower():
                entity_matches[entity].append(content[:100] + "...")

            if entity in tech_mapping:
                for tech_term in tech_mapping[entity]:
                    if tech_term.lower() in content.lower():
                        entity_matches[entity].append(f"{tech_term}: {content[:100]}...")

        return dict(entity_matches)

    def scrape_page_content(self, url: str, title: str, technology: str) -> Dict:
        """Scrape and process content from a single tutorial page"""
        try:
            response = self.session.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            main_content = soup.find("div", {"id": "main"})

            if not main_content:
                return {
                    "title": title,
                    "url": url,
                    "technology": technology,
                    "error": "No main content found",
                    "scraped_at": datetime.now().isoformat()
                }

            # Extract text content
            content_elements = main_content.find_all(
                ["p", "h1", "h2", "h3", "h4", "li", "div"],
                class_=lambda x: x not in ["tryit", "w3-btn", "w3-bar"]
            )

            text_content = []
            for elem in content_elements:
                text = elem.get_text().strip()
                if text and len(text) > 10:
                    text_content.append(text)

            # Extract code examples
            code_examples = []
            for code_elem in main_content.find_all(["pre", "code"]):
                code = code_elem.get_text().strip()
                if code and len(code) > 20:
                    code_examples.append(code)

            # Map content to ontology entities
            combined_content = " ".join(text_content + code_examples)
            ontology_mappings = self.map_content_to_ontology(combined_content, technology)

            # Mark URL as scraped
            self.scraped_urls[technology].add(url)

            return {
                "title": title,
                "url": url,
                "technology": technology,
                "text_content": text_content,
                "code_examples": code_examples,
                "ontology_mappings": ontology_mappings,
                "scraped_at": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "title": title,
                "url": url,
                "technology": technology,
                "error": str(e),
                "scraped_at": datetime.now().isoformat()
            }

    def scrape_all_tutorials(self) -> Dict[str, Dict]:
        """Scrape tutorials for all specified technologies"""
        print("=== Enhanced Web Scraping ===")
        scraped_data = defaultdict(dict)

        for tech in self.technologies:
            print(f"\nProcessing {tech.upper()} tutorials...")
            tutorial_urls = self.get_tutorial_urls(tech)

            if not tutorial_urls:
                print(f"No new {tech} tutorial URLs found")
                continue

            total_urls = len(tutorial_urls)
            print(f"Starting to scrape {total_urls} {tech} tutorials...")

            for i, link_info in enumerate(tutorial_urls):
                url = link_info["url"]
                title = link_info["title"]
                technology = link_info["technology"]

                print(f"Scraping {i + 1}/{total_urls}: {title[:50]}...")
                content = self.scrape_page_content(url, title, technology)
                scraped_data[tech][url] = content

                # Save URLs after each page
                self._save_scraped_urls(technology)

                # Respectful delay
                time.sleep(1)

                # Progress update
                if (i + 1) % 10 == 0:
                    print(f"Progress: {i + 1}/{total_urls} completed")

        return dict(scraped_data)

    def save_scraped_data(self, data: Dict[str, Dict], base_filename: str = "w3schools_") -> List[str]:
        """Save scraped data to separate files per technology"""
        filepaths = []
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)

            for tech, tech_data in data.items():
                if not tech_data:
                    continue

                filename = f"{base_filename}{tech}_tutorials.json"
                filepath = self.output_dir / filename

                # Add metadata
                metadata = {
                    "scraped_at": datetime.now().isoformat(),
                    "technology": tech,
                    "total_pages": len(tech_data),
                    "ontology_approach": self.ontology_creator.approach,
                    "ontology_stats": {
                        "entities": len(self.ontology["entities"]),
                        "relations": len(self.ontology["relations"]),
                        "categories": self.ontology.get("total_categories", 0)
                    }
                }

                output_data = {
                    "metadata": metadata,
                    "content": tech_data
                }

                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)

                file_size = os.path.getsize(filepath) / (1024 * 1024)
                print(f"Data saved to: {filepath}")
                print(f"File size: {file_size:.2f} MB")
                filepaths.append(str(filepath))

            return filepaths

        except Exception as e:
            print(f"Error saving data: {e}")
            return []

    def scrape_and_save(self) -> Optional[Dict]:
        """Main method to scrape and save data"""
        try:
            scraped_data = self.scrape_all_tutorials()

            if not scraped_data:
                print("No new data scraped")
                return None

            filepaths = self.save_scraped_data(scraped_data)

            if filepaths:
                # Print summary
                total_pages = sum(len(tech_data) for tech_data in scraped_data.values())
                successful = sum(
                    sum(1 for data in tech_data.values() if "error" not in data)
                    for tech_data in scraped_data.values()
                )
                failed = total_pages - successful
                tech_counts = {tech: len(data) for tech, data in scraped_data.items()}

                print(f"\n=== SCRAPING SUMMARY ===")
                print(f"Total pages: {total_pages}")
                print(f"Successful: {successful}")
                print(f"Failed: {failed}")
                print(f"Success rate: {(successful / total_pages) * 100:.1f}%")
                print(f"\nTechnology Breakdown:")
                for tech, count in tech_counts.items():
                    print(f"   • {tech.upper()}: {count} pages")

                print(f"\nOntology Integration Summary:")
                total_mappings = sum(
                    sum(len(data.get("ontology_mappings", {})) for data in tech_data.values())
                    for tech_data in scraped_data.values()
                )
                print(f"   • Total ontology entity mappings: {total_mappings}")
                print(f"   • Average mappings per page: {total_mappings / total_pages:.1f}")
                print(f"\nOutput Files:")
                for fp in filepaths:
                    print(f"   • {fp}")

                return scraped_data

            return None

        except Exception as e:
            print(f"Error in scraping process: {e}")
            return None


def main():
    """Run enhanced web scraping"""
    print("🌐 Enhanced W3Schools Multi-Technology Scraper")
    print("=" * 50)

    scraper = EnhancedWebScraper()
    data = scraper.scrape_and_save()

    if data:
        print(f"\nWeb scraping completed successfully!")

        # Show preview of first page per technology
        for tech, tech_data in data.items():
            if tech_data:
                first_content = list(tech_data.values())[0]
                print(f"\n{tech.upper()} Preview:")
                print(f"Title: {first_content['title']}")
                print(f"Text sections: {len(first_content.get('text_content', []))}")
                print(f"Code examples: {len(first_content.get('code_examples', []))}")
                print(f"Ontology mappings: {len(first_content.get('ontology_mappings', {}))}")
                if first_content.get("ontology_mappings"):
                    print("Sample mappings:")
                    for entity, snippets in list(first_content["ontology_mappings"].items())[:3]:
                        print(f"   • {entity}: {snippets[0][:50]}...")


if __name__ == "__main__":
    main()