import csv
import json
import logging
import os
import time
from typing import IO, Any, Dict, List, Tuple
import python_multipart

import chromadb
import ollama
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

base_dir = os.path.dirname(os.path.abspath(__file__))

load_dotenv(dotenv_path=os.path.join(base_dir, ".env"), override=False)

# Configure logging based on LOG_LEVEL from the environment (default: INFO).
log_level_name = os.getenv("LOG_LEVEL", "info").upper()
log_level = getattr(logging, log_level_name, logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

# Reduce noise from HTTP client libraries used by ollama.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embeddinggemma:300m")
CHROMA_PATH = os.getenv("CHROMA_PATH", os.path.join(base_dir, "chroma_storage"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "products")

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

logger = logging.getLogger(__name__)


def product_json_to_text(p: Dict[str, Any]) -> str:
    parts: List[str] = []
    if "title" in p:
        parts.append(f"Title: {p['title']}")
    if "product_category" in p:
        parts.append(f"Category: {p['product_category']}")
    if "brand" in p:
        parts.append(f"Brand: {p['brand']}")
    if "description" in p:
        parts.append(f"Description: {p['description']}")
    if "benefits" in p:
        parts.append(f"Benefits: {p['benefits']}")
    if "colors" in p:
        parts.append(f"Colors: {p['colors']}")
    return "\n".join(parts)


def product_csv_to_text(p: Dict[str, Any]) -> str:
    parts: List[str] = []
    if "product_name" in p:
        parts.append(f"Title: {p['product_name']}")
    if "product_category_tree" in p:
        parts.append(f"Category: {p['product_category_tree']}")
    if "brand" in p:
        parts.append(f"Brand: {p['brand']}")
    if "description" in p:
        parts.append(f"Description: {p['description']}")
    return "\n".join(parts)


def load_products_from_file(file_obj: IO[str], ext: str) -> List[Dict[str, Any]]:
    if ext == ".json":
        data = json.load(file_obj)
        if isinstance(data, dict):
            raise ValueError("JSON file must contain a list of products, not an object.")
        products = list(data)
        logger.info("Loaded %d products from JSON file.", len(products))
        return products
    if ext == ".csv":
        reader = csv.DictReader(file_obj)
        products = list(reader)
        logger.info("Loaded %d products from CSV file.", len(products))
        return products
    raise ValueError(f"Unsupported products file type: {ext}")


def embed_products(products: List[Dict[str, Any]], ext: str) -> int:
    if not products:
        logger.info("No products to embed.")
        return 0

    if ext == ".json":
        product_to_text = product_json_to_text
    else:
        product_to_text = product_csv_to_text

    logger.info("Starting embedding for %d products (format=%s).", len(products), ext)
    start_time = time.perf_counter()

    for idx, product in enumerate(products):
        text = product_to_text(product)
        response = ollama.embed(model=EMBEDDING_MODEL, input=text)
        embeddings = response["embeddings"]
        product_id = (
            product.get("id")
            or product.get("uniq_id")
            or str(idx)
        )
        if idx % 100 == 0:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(
                "Embedding progress index=%d id=%s elapsed=%.2fms",
                idx,
                product_id,
                elapsed_ms,
            )
        collection.add(
            ids=[product_id],
            embeddings=embeddings,
            documents=[json.dumps(product)],
        )

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "Completed embedding for %d products in %.2fms.",
        len(products),
        elapsed_ms,
    )

    return len(products)


def embed_products_from_upload(upload: UploadFile) -> int:
    filename = upload.filename or ""
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    if ext not in {".csv", ".json"}:
        raise ValueError("Only .csv and .json files are supported.")

    file_bytes = upload.file.read()
    logger.info(
        "Received upload for embedding: filename=%s, size=%d bytes, ext=%s",
        filename or "<unknown>",
        len(file_bytes),
        ext,
    )
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError("File must be UTF-8 encoded.") from e

    from io import StringIO

    file_obj = StringIO(text)
    products = load_products_from_file(file_obj, ext)
    return embed_products(products, ext)


def search_products(query: str, n_results: int = 10) -> Tuple[List[Dict[str, Any]], List[List[float]]]:
    if not query:
        raise ValueError("Query must not be empty.")

    logger.info("Starting search: query=%s, n_results=%d", query, n_results)
    start_time = time.perf_counter()

    response = ollama.embed(model=EMBEDDING_MODEL, input=query)
    results = collection.query(query_embeddings=response["embeddings"], n_results=n_results)
    documents = results.get("documents") or []
    distances = results.get("distances") or []

    decoded_products: List[Dict[str, Any]] = []
    for group in documents:
        for doc in group:
            try:
                decoded_products.append(json.loads(doc))
            except json.JSONDecodeError:
                decoded_products.append({"raw_document": doc})

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "Search completed: query=%s, requested=%d, returned=%d, elapsed=%.2fms.",
        query,
        n_results,
        len(decoded_products),
        elapsed_ms,
    )

    return decoded_products, distances


app = FastAPI(title="Embedding E-commerce Search API")


@app.post("/embed")
def embed_endpoint(file: UploadFile = File(...)) -> JSONResponse:
    logger.info("HTTP /embed called with file=%s", file.filename)
    try:
        count = embed_products_from_upload(file)
    except ValueError as e:
        logger.warning("Embed request validation failed: %s", e)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # pragma: no cover - unexpected errors
        logger.exception("Unexpected error during embedding.")
        raise HTTPException(status_code=500, detail="Failed to embed products.") from e

    logger.info("HTTP /embed completed successfully: embedded_count=%d", count)
    return JSONResponse({"status": "ok", "embedded_count": count})


@app.post("/search")
def search_endpoint(payload: Dict[str, Any]) -> JSONResponse:
    query = payload.get("query", "")
    n_results = int(payload.get("n_results", 10))

    if n_results <= 0:
        logger.warning("Invalid n_results value on /search: %d", n_results)
        raise HTTPException(status_code=400, detail="n_results must be positive.")

    try:
        logger.info("HTTP /search called: query=%s, n_results=%d", query, n_results)
        products, distances = search_products(query, n_results=n_results)
    except ValueError as e:
        logger.warning("Search request validation failed: %s", e)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # pragma: no cover - unexpected errors
        logger.exception("Unexpected error during search.")
        raise HTTPException(status_code=500, detail="Search failed.") from e

    logger.info(
        "HTTP /search completed successfully: query=%s, n_results=%d, returned=%d",
        query,
        n_results,
        len(products),
    )

    return JSONResponse(
        {
            "query": query,
            "n_results": n_results,
            "products": products,
            "distances": distances,
        }
    )


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "8000"))
    log_level = os.getenv("LOG_LEVEL", "info")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level=log_level,
    )
