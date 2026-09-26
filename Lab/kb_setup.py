"""
kb_setup.py - Load the ISDO knowledge base into ChromaDB and test retrieval.
 
Steps:
  1. Read every .md file in data/kb/
  2. Split each article into chunks at '## ' headings
  3. Store all chunks in the ChromaDB collection 'isdo_kb'
  4. Run 4 sample queries and print the best-matching article + confidence
 
Usage (from the project folder, with labenv activated):
    python Lab\kb_setup.py
 
Only dependency: chromadb (uses its built-in default embedding model,
all-MiniLM-L6-v2, which is downloaded once on first run).
"""
 
import re
from pathlib import Path
 
import chromadb
 
# --- Paths & settings -------------------------------------------------------
# Find the project folder that contains data/kb (works whether this script
# sits in the project root or in a subfolder such as Lab/)
_HERE = Path(__file__).resolve().parent
BASE_DIR = next((p for p in (_HERE, *_HERE.parents) if (p / "data" / "kb").is_dir()), _HERE)
KB_DIR = BASE_DIR / "data" / "kb"
DB_DIR = BASE_DIR / "data" / "chroma_db"   # persistent store on disk
COLLECTION_NAME = "isdo_kb"
 
 
# --- 1. Read markdown files -------------------------------------------------
def load_articles(kb_dir: Path) -> dict[str, str]:
    files = sorted(kb_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No .md files found in {kb_dir}")
    return {f.stem: f.read_text(encoding="utf-8") for f in files}
 
 
# --- 2. Split at '## ' headings ---------------------------------------------
def split_into_chunks(article_name: str, text: str) -> list[dict]:
    """Split one article at level-2 headings ('## ').
 
    - The text before the first '## ' (title + category/tags) becomes an
      'Overview' chunk.
    - '### ' sub-steps stay inside their parent '## ' section.
    - Each chunk is prefixed with the article title so it carries context.
    """
    title_match = re.search(r"^# (.+)$", text, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else article_name
 
    # Split right before each line that starts with exactly '## '
    parts = re.split(r"(?m)^(?=## )", text)
 
    chunks = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("## "):
            section = part.splitlines()[0][3:].strip()
        else:
            section = "Overview"
        chunks.append({
            "id": f"{article_name}::{len(chunks):02d}",
            "text": f"{title}\n\n{part}" if section != "Overview" else part,
            "metadata": {
                "article": article_name,
                "title": title,
                "section": section,
                "chunk_index": len(chunks),
            },
        })
    return chunks
 
 
# --- 3. Store in ChromaDB ---------------------------------------------------
def build_collection(all_chunks: list[dict]):
    client = chromadb.PersistentClient(path=str(DB_DIR))
 
    # Start clean on every run so edited/removed articles don't leave stale chunks
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
 
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # distance = 1 - cosine similarity
    )
    collection.add(
        ids=[c["id"] for c in all_chunks],
        documents=[c["text"] for c in all_chunks],
        metadatas=[c["metadata"] for c in all_chunks],
    )
    return collection
 
 
# --- 4. Query & score -------------------------------------------------------
def best_match(collection, query: str, n_results: int = 5) -> dict:
    """Return the best article for a query.
 
    Confidence = cosine similarity of the closest chunk (1 - cosine distance),
    shown as a percentage. The article owning that chunk is the answer.
    """
    res = collection.query(query_texts=[query], n_results=n_results)
    metas, dists = res["metadatas"][0], res["distances"][0]
 
    top_meta, top_dist = metas[0], dists[0]
    return {
        "article": top_meta["article"],
        "title": top_meta["title"],
        "section": top_meta["section"],
        "confidence": round((1 - top_dist) * 100, 1),
    }
 
 
SAMPLE_QUERIES = [
    ("I changed my password and now Cisco AnyConnect won't connect", "vpn_troubleshooting"),
    ("My account is locked after too many wrong login attempts", "password_reset"),
    ("Whole finance team getting DBCON_FAIL error when logging into SAP", "erp_connectivity"),
    ("Outlook on my iPhone stopped syncing new emails", "email_troubleshooting"),
]
 
 
def main():
    articles = load_articles(KB_DIR)
    print(f"Loaded {len(articles)} articles from {KB_DIR}")
 
    all_chunks = []
    for name, text in articles.items():
        chunks = split_into_chunks(name, text)
        all_chunks.extend(chunks)
        print(f"  - {name}: {len(chunks)} chunks")
 
    collection = build_collection(all_chunks)
    print(f"\nStored {collection.count()} chunks in collection '{COLLECTION_NAME}' ({DB_DIR})\n")
 
    print("=" * 78)
    print("Sample queries")
    print("=" * 78)
    passed = 0
    for i, (query, expected) in enumerate(SAMPLE_QUERIES, 1):
        m = best_match(collection, query)
        ok = m["article"] == expected
        passed += ok
        print(f"\nQ{i}: {query}")
        print(f"    Best match : {m['article']}.md  ({m['title']})")
        print(f"    Section    : {m['section']}")
        print(f"    Confidence : {m['confidence']}%")
        print(f"    Expected   : {expected}.md  -> {'PASS' if ok else 'FAIL'}")
 
    print(f"\n{passed}/{len(SAMPLE_QUERIES)} queries matched the expected article.")
 
 
if __name__ == "__main__":
    main()