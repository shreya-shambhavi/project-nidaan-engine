import requests
import xml.etree.ElementTree as ET

def fetch_pubmed_abstracts(query: str, max_results: int = 3) -> list[str]:

    print(f"Retriever Agent: Searching PubMed for '{query}'...")

    # Step 1: Search for article IDs (unchanged)
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max_results
    }

    try:
        search_response = requests.get(search_url, params=search_params, timeout=10)
        search_response.raise_for_status()
        search_data = search_response.json()
    except requests.RequestException as e:
        print(f"Retriever Agent: Search failed — {e}")
        return []

    article_ids = search_data.get("esearchresult", {}).get("idlist", [])

    if not article_ids:
        print("Retriever Agent: No articles found.")
        return []

    print(f"Retriever Agent: Found {len(article_ids)} articles. Fetching XML...")

    # Step 2: Fetch in XML mode instead of plain text
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    fetch_params = {
        "db": "pubmed",
        "id": ",".join(article_ids),
        "retmode": "xml"       # was: retmode=text, rettype=abstract
    }

    try:
        fetch_response = requests.get(fetch_url, params=fetch_params, timeout=15)
        fetch_response.raise_for_status()
    except requests.RequestException as e:
        print(f"Retriever Agent: Fetch failed — {e}")
        return []

    return _parse_pubmed_xml(fetch_response.text)


def _parse_pubmed_xml(xml_text: str) -> list[str]:

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print(f"Retriever Agent: XML parse error — {e}")
        return []

    abstracts = []

    for article in root.findall(".//PubmedArticle"):

        # Extract metadata
        pmid_el   = article.find(".//PMID")
        title_el  = article.find(".//ArticleTitle")
        journal_el = article.find(".//Journal/Title")
        year_el   = article.find(".//PubDate/Year")

        pmid    = pmid_el.text    if pmid_el    is not None else "Unknown"
        title   = title_el.text   if title_el   is not None else "No title"
        journal = journal_el.text if journal_el is not None else "Unknown journal"
        year    = year_el.text    if year_el    is not None else "Unknown year"

        # Handle both simple and structured (sectioned) abstracts
        abstract_sections = article.findall(".//AbstractText")

        if not abstract_sections:
            continue   # skip articles with no abstract

        if len(abstract_sections) == 1:
            # Simple abstract: one block of text
            body = abstract_sections[0].text or ""
        else:
            # Structured abstract: BACKGROUND / METHODS / RESULTS / CONCLUSIONS
            parts = []
            for section in abstract_sections:
                label = section.get("Label", "")
                text  = section.text or ""
                parts.append(f"{label}: {text}" if label else text)
            body = "\n".join(parts)

        formatted = (
            f"PMID: {pmid}\n"
            f"Title: {title}\n"
            f"Journal: {journal} ({year})\n"
            f"Abstract:\n{body}"
        )

        abstracts.append(formatted)

    print(f"Retriever Agent: Parsed {len(abstracts)} abstracts successfully.")
    return abstracts
