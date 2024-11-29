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


def text_extraction(pdf_name, images, slide_dict):
    """
    Extract text from a PDF and pair it with corresponding slide images, updating the given dictionary
    and saving the result as a YAML file.

    Parameters
    ----------
    pdf_name : str
        The name or path of the PDF file from which to extract text.
    images : list
        A list of images corresponding to the slides in the PDF.
    slide_dict : dict
        A dictionary mapping slide image file names to the extracted text.

    Returns
    -------
    dict
        The updated slide_dict containing the new text-image pairs.
    """
    import yaml
    import pdfplumber
    import re

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

    return slide_dict



        
    
def text_extract_from_pdfs(downloads_folder="downloads", yaml_file_path="dict_slides_text.yml"):
    """
    Extract text from PDFs based on their corresponding slide images and save the text-image pairs in a YAML file.
    Skips PDFs whose slides are already processed and stored in the YAML file.

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

    # Process each group of images
    for pdf_name, image_list in pdf_image_map.items():
        # Sort the images to ensure slide order
        image_list.sort()

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



