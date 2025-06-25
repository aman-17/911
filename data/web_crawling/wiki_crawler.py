import argparse
import os
import re

import requests
from bs4 import BeautifulSoup


def parse_wiki_page(url):
    """Fetch and parse a Wikipedia page, extracting only main paragraph text."""
    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        content_div = soup.find("div", {"id": "mw-content-text"})
        if not content_div:
            raise ValueError("Could not find main content div")

        paragraphs = content_div.find_all("p")

        clean_text = []
        for p in paragraphs:
            if len(p.text.strip()) < 20:
                continue
            text = re.sub(r"\[\d+\]", "", p.text)
            text = re.sub(r"\[[^\]]*\]", "", text)
            text = re.sub(r"\s+", " ", text).strip()

            if text:
                clean_text.append(text)

        return "\n\n".join(clean_text)

    except Exception as e:
        return e


def save_to_file(text, url, output_dir="."):
    if not text:
        return False

    page_name = url.split("/")[-1]
    filename = f"{page_name}.txt"
    filepath = os.path.join(output_dir, filename)

    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
        return True
    except Exception as e:
        return e


def main():
    parser = argparse.ArgumentParser(
        description="Extract clean text from Wikipedia pages"
    )
    parser.add_argument("url", help="Wikipedia URL to parse")
    parser.add_argument(
        "--output",
        "-o",
        default="wiki_text",
        help="Directory to save output files (default: wiki_text)",
    )
    args = parser.parse_args()
    text = parse_wiki_page(args.url)

    if text:
        save_to_file(text, args.url, args.output)
    else:
        print("Failed to extract text from the provided URL")


if __name__ == "__main__":
    main()
