import ollama
import chromadb
import os
import json
import csv

EMBEDDING_MODEL = "embeddinggemma:300m"
CHAT_MODEL= "gemma3:4b"

base_dir = os.path.dirname(os.path.abspath(__file__))
products_path = os.path.join(base_dir, "data", "flipkart_products_small_sample.csv")
chroma_path = os.path.join(base_dir, "chroma_storage")

client = chromadb.PersistentClient(path=chroma_path)
collection = client.get_or_create_collection(name="products")


def product_json_to_text(p):
    parts = []
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


def product_csv_to_text(p):
    parts = []
    if "product_name" in p:
        parts.append(f"Title: {p['product_name']}")
    if "product_category_tree" in p:
        parts.append(f"Category: {p['product_category_tree']}")
    if "brand" in p:
        parts.append(f"Brand: {p['brand']}")
    if "description" in p:
        parts.append(f"Description: {p['description']}")
    return "\n".join(parts)


products = None

products_ext = os.path.splitext(products_path)[1].lower()

with open(products_path, "r", encoding="utf-8") as f:
    if products_ext == ".json":
        products = json.load(f)
    elif products_ext == ".csv":
        reader = csv.DictReader(f)
        products = list(reader)
    else:
        raise ValueError(f"Unsupported products file type: {products_ext}")

if products:
    if products_ext == ".json":
        product_to_text = product_json_to_text
    else:
        product_to_text = product_csv_to_text

    for idx, product in enumerate(products):
        text = product_to_text(product)
        response = ollama.embed(model=EMBEDDING_MODEL, input=text)
        embeddings = response["embeddings"]
        product_id = (
            product.get("id")
            or product.get("uniq_id")
            or str(idx)
        )
        collection.add(
            ids=[product_id],
            embeddings=embeddings,
            documents=[json.dumps(product)],
        )

# an example input
prompt = "Can you recommend biking shorts?"

# generate an embedding for the input and retrieve the most relevant doc
response = ollama.embed(model=EMBEDDING_MODEL, input=prompt)
results = collection.query(query_embeddings=response["embeddings"], n_results=10)
data = results["documents"]

# output = ollama.generate(
#   model=CHAT_MODEL,
#   prompt=f"Using this data: {data}. Respond to this prompt: {prompt} in JSON format"
# )

# print(output['response'])

print(data)
