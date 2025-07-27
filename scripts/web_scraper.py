"""
Step 2: Web Scraping from W3Schools Java tutorials
"""

import requests
from bs4 import BeautifulSoup
import time
import json
import os
from datetime import datetime


class WebScraper:
    def __init__(self, output_dir="data/scraped_content"):
        self.base_url = "https://www.w3schools.com/java/"
        self.main_url = "https://www.w3schools.com/java/default.asp"
        self.output_dir = output_dir
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def get_tutorial_urls(self):
        """Extract all Java tutorial URLs from main page"""
        try:
            print("Fetching tutorial URLs...")
            response = self.session.get(self.main_url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            tutorial_links = []

            # Find Java tutorial links
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if href.startswith("java_") and href.endswith(".asp"):
                    full_url = self.base_url + href
                    title = link.get_text().strip()
                    tutorial_links.append({
                        "url": full_url,
                        "title": title,
                        "filename": href
                    })

            # Remove duplicates
            seen_urls = set()
            unique_links = []
            for link in tutorial_links:
                if link["url"] not in seen_urls:
                    seen_urls.add(link["url"])
                    unique_links.append(link)

            print(f"Found {len(unique_links)} tutorial URLs")
            return unique_links

        except Exception as e:
            print(f"Error getting tutorial URLs: {e}")
            return []

    def scrape_page_content(self, url, title):
        """Scrape content from a single tutorial page"""
        try:
            response = self.session.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            main_content = soup.find("div", {"id": "main"})

            if not main_content:
                return {
                    "title": title,
                    "url": url,
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

            return {
                "title": title,
                "url": url,
                "text_content": text_content,
                "code_examples": code_examples,
                "scraped_at": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "title": title,
                "url": url,
                "error": str(e),
                "scraped_at": datetime.now().isoformat()
            }

    def scrape_all_tutorials(self):
        """Scrape all tutorial pages"""
        print("=== Step 2: Web Scraping ===")

        tutorial_urls = self.get_tutorial_urls()
        if not tutorial_urls:
            print("No tutorial URLs found")
            return {}

        scraped_data = {}
        total_urls = len(tutorial_urls)

        print(f"Starting to scrape {total_urls} tutorials...")

        for i, link_info in enumerate(tutorial_urls):
            url = link_info["url"]
            title = link_info["title"]

            print(f"Scraping {i + 1}/{total_urls}: {title[:50]}...")

            content = self.scrape_page_content(url, title)
            scraped_data[url] = content

            # Respectful delay
            time.sleep(1)

            # Progress update
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{total_urls} completed")

        return scraped_data

    def save_scraped_data(self, data, filename="w3schools_java_tutorials.json"):
        """Save scraped data to JSON file"""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            filepath = os.path.join(self.output_dir, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            file_size = os.path.getsize(filepath) / (1024 * 1024)
            print(f"Data saved to: {filepath}")
            print(f"File size: {file_size:.2f} MB")
            return filepath

        except Exception as e:
            print(f"Error saving data: {e}")
            return None

    def scrape_and_save(self):
        """Main method to scrape tutorials and save data"""
        try:
            scraped_data = self.scrape_all_tutorials()

            if not scraped_data:
                print("No data scraped")
                return None

            filepath = self.save_scraped_data(scraped_data)

            if filepath:
                # Print summary
                total_pages = len(scraped_data)
                successful = sum(1 for data in scraped_data.values() if "error" not in data)
                failed = total_pages - successful

                print(f"\n=== SCRAPING SUMMARY ===")
                print(f"Total pages: {total_pages}")
                print(f"Successful: {successful}")
                print(f"Failed: {failed}")
                print(f"Success rate: {(successful / total_pages) * 100:.1f}%")

                return scraped_data

            return None

        except Exception as e:
            print(f"Error in scraping process: {e}")
            return None


def main():
    """Run web scraping"""
    scraper = WebScraper()
    data = scraper.scrape_and_save()

    if data:
        print(f"\nWeb scraping completed successfully!")

        # Show preview of first page
        first_url, first_content = list(data.items())[0]
        print(f"\nFirst page preview:")
        print(f"Title: {first_content['title']}")
        print(f"Text sections: {len(first_content.get('text_content', []))}")
        print(f"Code examples: {len(first_content.get('code_examples', []))}")


if __name__ == "__main__":
    main()