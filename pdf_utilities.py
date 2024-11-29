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


def text_extraction(pdf_name, images):
    """
    Extract text from a PDF and pair it with corresponding slide images, saving the result as a YAML file.
    If the YAML file already exists, it extends it with new text-image pairs.   
    
    Parameters
    ----------
    pdf_name : str
        The name or path of the PDF file from which to extract text.
    images : list
        A list of images corresponding to the slides in the PDF.
    
    Returns
    -------
    None
        Saves a dictionary mapping slide image file names (e.g., slide1.png) to the extracted text
        in a YAML file named 'dict_slides_text.yml'.
    """
    import yaml
    import pdfplumber
    import os
    import re

    slide_texts = []
    
    # Removing the .pdf end in the PDF filename:
    pdf_new = re.sub(r'\.pdf$', '', pdf_name, flags=re.IGNORECASE)

    # Extract the original text from each Slide
    with pdfplumber.open(pdf_name) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            slide_texts.append(text)

    yaml_file_path = "dict_slides_text.yml"
    slide_dict = {}

    # Load existing dictionary if it exists
    if os.path.exists(yaml_file_path):
        with open(yaml_file_path, "r") as yaml_file:
            slide_dict = yaml.safe_load(yaml_file) or {}
    
    for i, (img, text) in enumerate(zip(images, slide_texts), start=1):
        # Define the path for each image file
        image_path = f"{pdf_new}_slide{i}.png"
        
        # Add the image path and corresponding text to the dictionary only if it doesn't already exist
        if image_path not in slide_dict:
            slide_dict[image_path] = text
    
    # Save the dictionary to a YAML file
    with open(yaml_file_path, "w") as yaml_file:
        yaml.dump(slide_dict, yaml_file, default_flow_style=False, allow_unicode=True, sort_keys=False)


        
    


    














