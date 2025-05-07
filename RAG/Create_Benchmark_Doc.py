# File to create a Benchmark Document to compare the two different RAG Approaches

"""
Approaches
1. using byaldi package (https://github.com/AnswerDotAI/byaldi)
2. using CLIP Model to compare query and slide embedding (https://huggingface.co/openai/clip-vit-base-patch32)

20 potential search queries are generated using ChatGPT-4o and processed with both approaches.
Both pipelines output the 10 best matching slides from the NFDI4BIOIMAGE training material data base.
Human raters then examine the fit of each output to see whether one of the approaches outperforms the other one.
Outputs are going to be unlabeled, in terms of the used approach, to prevent introducing any biases to the results.

"""


import sys
import os

# Add parent directory to be able to import modules / files
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
        
import pandas as pd
from natsort import natsorted
from caching import load_full_hf_cache, get_zenodo_pdfs
from pdf_utilities import download_all_pdfs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import torch
from sklearn.metrics.pairwise import cosine_similarity
from caching import load_full_hf_cache
from matplotlib.backends.backend_pdf import PdfPages
import gc

def search_via_byaldi(query, k_results, docs_retrieval_model):

    # Retrieve k best results
    results = docs_retrieval_model.search(query, k=k_results)
    
    # Reconstruct the images
    reconstructed_images = []
    scores = []
    
    for result in results:
        base64_img = result["base64"]  
        image_data = base64.b64decode(base64_img)
        image = Image.open(BytesIO(image_data))
        reconstructed_images.append(image)
        scores.append(result["score"]) 

    del results, docs_retrieval_model
    gc.collect()
    
    return reconstructed_images, scores


def search_via_CLIP_similarity(query, k_results,  clip_model, clip_processor):

    # Tokenize the text 
    inputs = clip_processor(text=[query], return_tensors="pt", padding=True)
    
    # Get the text embedding from the model
    with torch.no_grad():
        query_embedding = clip_model.get_text_features(**inputs)
    
    # Normalize and send to the Device
    query_vector = query_embedding / query_embedding.norm(dim=-1, keepdim=True) 
    query_vector = query_vector.cpu().numpy()

    # Load the DataSet with the stores Embeddings from Huggingface
    repo_name = "ScaDS-AI/SlideInsight_Cache"
    df = load_full_hf_cache(repo_name=repo_name)

    # Extract the visual embeddings
    visual_embeddings = df['visual_embedding'].apply(lambda x: np.array(x))
    visual_matrix = np.stack(visual_embeddings.to_list()) 

    # Compute similarity between the embedding matrix and the query vector
    similarities = cosine_similarity(query_vector, visual_matrix) 
    similarities = similarities.flatten()  
    
    df['similarity_to_query'] = similarities

    # Retrieve k best results
    top_matches = df.sort_values(by='similarity_to_query', ascending=False).head(k_results)

    # Import Image dataset 
    from datasets import load_dataset, Image
    dataset = load_dataset("ScaDS-AI/Slide_Insight_Images")["train"]  
    dataset = dataset.cast_column("image", Image())
    df_image = dataset.to_pandas()

    keys = top_matches['key'].tolist()
    filtered_df = df_image[df_image["key"].isin(keys)]

    del df, dataset, df_image, clip_model, clip_processor
    gc.collect()

    return filtered_df, top_matches



def generate_comparison_pdf(queries, doc_model, clip_model, clip_processor, k_results=5, output_path="benchmark_results.pdf"):
    with PdfPages(output_path) as pdf:
        for query in queries:
            try:
                print(f"Processing query: {query}")

                # --- Approach 1: Byaldi ---
                print('Processing the Byaldi approach...')
                byaldi_images, byaldi_scores = search_via_byaldi(query, k_results, doc_model)

                fig, axs = plt.subplots(1, k_results, figsize=(5 * k_results, 5))
                if k_results == 1:
                    axs = [axs]

                for i in range(k_results):
                    axs[i].imshow(byaldi_images[i])
                    #axs[i].set_title(f"Score: {byaldi_scores[i]:.3f}", fontsize=10)
                    axs[i].axis("off")
        
                fig.suptitle(f"Query: {query}\nApproach A", fontsize=14)
                pdf.savefig(fig)
                plt.close(fig)

                # --- Approach 2: CLIP ---
                print('Processing the CLIP approach...')
                clip_df, top_matches = search_via_CLIP_similarity(query, k_results, clip_model, clip_processor)

                # --- Plotting ---
                print('Plotting...')
                fig, axs = plt.subplots(1, k_results, figsize=(5 * k_results, 5))
                if k_results == 1:
                    axs = [axs]

                for i, (_, row) in enumerate(clip_df.iloc[:k_results].iterrows()):
                    img_bytes = row["image"]["bytes"]
                    img = Image.open(BytesIO(img_bytes)).convert("RGB")
                    key = row["key"]
                    score = top_matches[top_matches["key"] == key].iloc[0]["similarity_to_query"]

                    axs[i].imshow(img)
                    #axs[i].set_title(f"Score: {score:.3f}", fontsize=10)
                    axs[i].axis("off")
                    
                fig.suptitle(f"Query: {query}\nApproach B", fontsize=14)
                pdf.savefig(fig)
                plt.close(fig)
                img.close()

                # Free memory
                del clip_df, top_matches, fig, axs
                gc.collect()

            except Exception as e:
                print(f"Error processing query '{query}': {e}")







































