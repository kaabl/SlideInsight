def load_pdf(pdf_name):
    """
    Load pdf and convert it to a list of its pages as images.

    Parameters
    ----------
    pdf_name: str
        Name of the pdf (must be in the same folder as this notebook)

    Returns
    -------
    slides: list
        List of Images 
    """
    from pdf2image import convert_from_path

    slides = convert_from_path(pdf_name)
    return slides



def save_images(filepath, pdf, new_width=None):
    """
    Save a list of images (from pdf file) to a specified directory , optionally with a resized width.

    Parameters
    ----------
    filepath : str
        The directory path where the images will be saved.
    pdf : str
        Name of the pdf (must be in the same folder as this notebook)
    new_width : int, optional
        The new width for resizing each image while maintaining its aspect ratio.
        If not provided, the original width of each image will be used.

    Returns
    -------
    None
        Saves each image in the specified directory as PNG files named slide1.png, slide2.png, etc.
    """
    from PIL import Image
    import os
    from pdf2image import convert_from_path
    import re
    
    # Convert pdf to images
    slides = convert_from_path(pdf)
    
    for i, img in enumerate(slides, start=1):
        # Use original width if new_width is not specified
        if new_width is None:
            new_width = img.width
        
        # Calculate the new height to maintain the aspect ratio
        aspect_ratio = img.height / img.width
        new_height = int(new_width * aspect_ratio)
        
        # Resize the image
        img_resized = img.resize((new_width, new_height))
        
        # Ensure filepath directory exists
        os.makedirs(filepath, exist_ok=True)

        # Extract the filename without directory and extension
        pdf_name = os.path.basename(pdf)
        pdf_new = re.sub(r'\.pdf$', '', pdf_name, flags=re.IGNORECASE)

        # Saving the image in the specified filepath
        image_filename = os.path.join(filepath, f"{pdf_new}_slide{i}.png")
        img_resized.save(image_filename, "PNG")


def text_extraction(pdf_name, images, slide_dict=None):
    """
    Extract text from a PDF and pair it with corresponding slide images, updating the given dictionary
    or creating a new one if none is provided. The result is saved as a YAML file.

    Parameters
    ----------
    pdf_name : str
        The name or path of the PDF file from which to extract text.
    images : list
        A list of images corresponding to the slides in the PDF.
    slide_dict : dict, optional
        A dictionary mapping slide image file names to the extracted text. 
        If None, a new dictionary is created.

    Returns
    -------
    dict
        The updated slide_dict containing the new text-image pairs.
    """
    import yaml
    import pdfplumber
    import re

    # Initialize slide_dict if not provided
    if slide_dict is None:
        slide_dict = {}

    slide_texts = []

    # Removing the .pdf end in the PDF filename:
    pdf_new = re.sub(r'\.pdf$', '', pdf_name, flags=re.IGNORECASE)

    # Extract the original text from each slide
    with pdfplumber.open(pdf_name) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            slide_texts.append(text)

    # Update the dictionary with new text-image pairs
    for i, (img, text) in enumerate(zip(images, slide_texts), start=1):
        # Define the path for each image file
        image_path = f"{pdf_new}_slide{i}.png"

        # Add the image path and corresponding text to the dictionary only if it doesn't already exist
        if image_path not in slide_dict:
            slide_dict[image_path] = text

    # Save the updated dictionary as a YAML file
    with open("dict_slides_text.yml", "w") as yaml_file:
        yaml.dump(slide_dict, yaml_file, default_flow_style=False)
        
    return slide_dict



def text_extract_from_pdfs(downloads_folder="downloads", yaml_file_path="dict_slides_text.yml"):
    """
    Extract text from PDFs based on their corresponding slide images and save the text-image pairs in a YAML file.
    Processes PDFs and images in sorted order. Skips PDFs whose slides are already processed and stored in the YAML file.

    Parameters
    ----------
    downloads_folder : str, optional
        The path to the folder containing the PDF files and the 'images' subfolder (default is "downloads").
    yaml_file_path : str, optional
        The path to the YAML file where text-image pairs will be stored (default is "dict_slides_text.yml").

    Returns
    -------
    None
    """
    import os
    import yaml
    import re
    from natsort import natsorted

    # Define paths
    images_folder = os.path.join(downloads_folder, "images")

    # Ensure the YAML dictionary exists or initialize an empty one
    if os.path.exists(yaml_file_path):
        with open(yaml_file_path, "r") as yaml_file:
            slide_dict = yaml.safe_load(yaml_file) or {}
    else:
        slide_dict = {}

    # Group images by their corresponding PDFs
    pdf_image_map = {}
    for image_file in os.listdir(images_folder):
        if image_file.lower().endswith('.png'):
            # Extract the base PDF name from the image filename (e.g., `example_slide1.png` -> `example`)
            match = re.match(r"(.+)_slide\d+\.png", image_file)
            if match:
                pdf_name = match.group(1)
                pdf_image_map.setdefault(pdf_name, []).append(os.path.join(images_folder, image_file))

    # Sort the PDFs alphabetically and their corresponding images
    sorted_pdf_image_map = {pdf_name: natsorted(images) for pdf_name, images in natsorted(pdf_image_map.items())}

    # Process each group of images
    for pdf_name, image_list in sorted_pdf_image_map.items():
        # Check if these slides are already in the dictionary
        processed_slides = [key for key in slide_dict if pdf_name in key]
        if processed_slides:
            print(f"Slides for {pdf_name} already processed. Skipping.")
            continue

        # Construct the PDF file path
        pdf_path = os.path.join(downloads_folder, f"{pdf_name}.pdf")
        if not os.path.exists(pdf_path):
            print(f"PDF file {pdf_path} corresponding to the images not found. Skipping.")
            continue

        # Process and extract text
        try:
            print(f"Processing slides for {pdf_name}...")
            slide_dict = text_extraction(pdf_path, image_list, slide_dict)
        except Exception as e:
            print(f"Error processing slides for {pdf_name}: {e}")

    # Save the updated dictionary to the YAML file
    with open(yaml_file_path, "w") as yaml_file:
        yaml.dump(slide_dict, yaml_file, default_flow_style=False, allow_unicode=True, sort_keys=False)



def get_mixed_embedding(client, image_path, text_model):
    """
    Generates a structured description of an image using GPT-4o.

    Parameters
    ----------
    client : ChatCompletionsClient
        The GPT-4o client.
    image_path : str
        Path to the image.
    text_model: str

    Returns
    -------
    mixed_embedding:
        Text Embedding of the models anwser. 
    """
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import (
                SystemMessage,
                UserMessage,
                TextContentItem,
                ImageContentItem,
                ImageUrl,
                ImageDetailLevel,
            )
    from azure.core.credentials import AzureKeyCredential 
    import os
    from PIL import Image

        
    endpoint = "https://models.inference.ai.azure.com"
    token = os.environ["GITHUB_TOKEN"]
    client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(token),
        )

    response = client.complete(
        messages=[
            SystemMessage(
                content="You are a professional Data Scientist. Provide a structured description of the image in 1-2 sentences."
            ),
            UserMessage(
                content=[
                    ImageContentItem(
                        image_url=ImageUrl.load(
                            image_file=image_path,
                            image_format="png",
                            detail=ImageDetailLevel.LOW
                        )
                    ),
                ],
            ),
        ],
        model="gpt-4o",
    )

    # Parse structured description from response
    structured_response = response.choices[0].message.content
    
    # Convert the textual response into an embedding
    mixed_embedding = text_model.encode(structured_response)

    return mixed_embedding



def calculate_text_embeddings(pdf_name, text_model):
    """
    Extracts text from each page of a PDF and computes text embeddings.

    Parameters
    ----------
    pdf_name : str
        The path to the PDF file from which text needs to be extracted.
    text_model: str
        Name of the text embedding model.

    Returns
    -------
    dict
        A dictionary where keys are page numbers (int) and values are text embeddings (array-like),
        representing the encoded textual content of each page.
    """
    import pdfplumber
    
    text_embeddings = {}
    with pdfplumber.open(pdf_name) as pdf:
        for page_number, page in enumerate(pdf.pages):
            text = page.extract_text() or ""  # Handle empty pages gracefully
            text_embeddings[page_number] = text_model.encode(text)
    return text_embeddings


def process_slides(slides, client, clip_processor, clip_model, text_model):
    """
    Processes PDF slides to compute visual and mixed-modal embeddings.

    Parameters
    ----------
    slides : list
        List of images representing the slides.
    client : object
        The initialized client for mixed-modal embedding (GitHub Marketplace ChatGPT 4o).
    clip_processor: str
    clip_model: str
    text_model: str

    Returns
    -------
    list
        A list of dictionaries containing embeddings and slide numbers.
    """
    import torch
    import os
    from PIL import Image
    from transformers import CLIPProcessor, CLIPModel
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import (
            SystemMessage,
            UserMessage,
            TextContentItem,
            ImageContentItem,
            ImageUrl,
            ImageDetailLevel,
        )
    from azure.core.credentials import AzureKeyCredential 
    
    
    endpoint = "https://models.inference.ai.azure.com"
    token = os.environ["GITHUB_TOKEN"]
    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(token),
    )

    
    slide_embeddings = []
    for slide_number, slide_image in enumerate(slides):
        # Save slide image temporarily
        image_path = f"images/slide_{slide_number}.png"
        slide_image.save(image_path)
        
        # Generate visual embedding using CLIP
        inputs = clip_processor(images=slide_image, return_tensors="pt")
        with torch.no_grad():
            visual_embedding = clip_model.get_image_features(**inputs).squeeze().tolist()
        
        # Generate 'mixed-modal' embedding using GPT-4o
        try:
            mixed_embedding = get_mixed_embedding(client, image_path, text_model)
        except Exception as e:
            print(f"Error generating GPT-4o embedding for slide {slide_number}: {e}")
            mixed_embedding = None
        
        # Append embeddings
        slide_embeddings.append({
            "slide_number": slide_number,
            "visual_embedding": visual_embedding,
            "mixed_modal_embedding": mixed_embedding,
        })
    return slide_embeddings
























