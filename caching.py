from datasets import Dataset, Features, Value, Sequence
from datasets import load_dataset
from huggingface_hub import create_repo, login, HfApi
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import pdfplumber
import torch
import os
import shelve


# Initialize models
text_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define features schema
features = Features({
    "key": Value("string"),
    "value": {
        "text_embedding": Sequence(Value("float32")),
        "vision_embedding": Sequence(Value("float32")),
    }
})

# Function to check and create a repository if it doesn't exist
def ensure_repo_exists(repo_name):
    api = HfApi()
    user = api.whoami()["name"]
    
    # Check if repo_name already includes the namespace 
    if "/" not in repo_name:
        full_repo_name = f"{user}/{repo_name}"
    else:
        full_repo_name = repo_name  # Assume it's already correctly formatted
    
    try:
        # Check if the repository already exists
        api.repo_info(full_repo_name, repo_type="dataset")
        print(f"Repository '{full_repo_name}' already exists.")
    except Exception:
        # Create the repository if it doesn't exist
        create_repo(repo_name, repo_type="dataset", private=False)
        print(f"Repository '{full_repo_name}' created.")
    return full_repo_name

# Load or initialize Hugging Face cache
def load_cache_dataset(repo_name):
    try:
        # Try to load the dataset
        return load_dataset(repo_name, split="train")
    except Exception:
        # If it doesn't exist, create a new dataset
        data = {"key": [], "value": []}
        return Dataset.from_dict(data, features=features)
        
# Embedding functions
def embed_text(pdf_filename, slide_number, text_model):
    with pdfplumber.open(pdf_filename) as pdf:
        text = pdf.pages[slide_number - 1].extract_text() or ""
        return text_model.encode(text)

def embed_visual(pdf_filename, slide_number, clip_processor, clip_model):
    with pdfplumber.open(pdf_filename) as pdf:
        image = pdf.pages[slide_number - 1].to_image().original
        inputs = clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            return clip_model.get_image_features(**inputs).squeeze().tolist()
            
def caching_hf(pdf_path, repo_name="your-hf-repo"):

    # Ensure the repository exists or create a new one
    full_repo_name = ensure_repo_exists(repo_name)

    # Load or initialize the dataset
    cache_dataset = load_cache_dataset(full_repo_name)

    # Process each slide
    slides = range(1, len(pdfplumber.open(pdf_path).pages) + 1)
    for slide_number in slides:
        key = f"{pdf_path}_slide{slide_number}"
        if key in cache_dataset["key"]:
            print(f"Fetching from cache: Slide {slide_number}")
            continue

        result = {
            "text_embedding": embed_text(pdf_path, slide_number, text_model),
            "vision_embedding": embed_visual(pdf_path, slide_number, clip_processor, clip_model)
        }

        new_entry = {
            "key": key,
            "value": {
                "text_embedding": result["text_embedding"],
                "vision_embedding": result["vision_embedding"]
            }
        }
        cache_dataset = cache_dataset.add_item(new_entry)
        print(f"Caching Slide {slide_number}")

    # Push to the Hugging Face Hub
    cache_dataset.push_to_hub(full_repo_name)
    print(f"Finished caching {pdf_path}")


def caching_local(pdf_path):
    with shelve.open("local_cache.db") as cache:
        slides = range(1, len(pdfplumber.open(pdf_path).pages) + 1)

        for slide_number in slides:
            key = f"{pdf_path}_slide{slide_number}"
            if key in cache:
                print(f"Fetching from cache: Slide {slide_number}")
                continue

            print(f"Caching slide {slide_number}")
            text_embedding = embed_text(pdf_path, slide_number, text_model)
            visual_embedding = embed_visual(pdf_path, slide_number, clip_processor, clip_model)
            cache[key] = {"text": text_embedding, "visual": visual_embedding}
