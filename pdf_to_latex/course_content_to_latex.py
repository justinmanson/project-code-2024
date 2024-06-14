import fitz  # PyMuPDF
import io
import base64
import json
import os
from PIL import Image
import argparse
from openai import OpenAI

MAX_IMAGE_SIZE_MB = 18
INITIAL_QUALITY = 90
QUALITY_DECREMENT = 5

def encode_image(image, quality):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=quality)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_image_size(image, quality):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=quality)
    size_mb = len(buffered.getvalue()) / (1024 * 1024)
    return size_mb

def compress_image_to_limit(image):
    quality = INITIAL_QUALITY
    size_mb = get_image_size(image, quality)

    while size_mb > MAX_IMAGE_SIZE_MB and quality > QUALITY_DECREMENT:
        print(f"Image size snag, current image size is {size_mb}. Not all is lost yet, let's try to compress it again!")
        quality -= QUALITY_DECREMENT
        size_mb = get_image_size(image, quality)
    
    if size_mb > MAX_IMAGE_SIZE_MB:
        raise ValueError("Image cannot be compressed below the size limit.")
    
    return encode_image(image, quality)

def save_image_locally(image, directory, page_number):
    """Save the image locally for inspection."""
    os.makedirs(directory, exist_ok=True)
    image_path = os.path.join(directory, f'page_{page_number}.jpeg')
    image.save(image_path, format='JPEG')

def convert_pdf_to_images(pdf_path):
    document = fitz.open(pdf_path)
    images = []
    for page in document:
        # Render page to an image
        pix = page.get_pixmap(dpi=96)  # Low res
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Resize the image to fit 512x512 as per API's low detail specification
        img.thumbnail((512, 512), Image.LANCZOS)

        try:
            compressed_image = compress_image_to_limit(img)
            # save_image_locally(img, "image_from_pdf_test", 0)
            images.append(compressed_image)
        except ValueError:
            print(f"Skipping image from page {page.number} as it exceeds the size limit even after compression.")
    return images

def query_openai_for_latex(images, client):
    responses = []
    for base64_image in images:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract and return the content (text and math) of this image. Ignore figures, ignore diagrams, ignore captions, ignore page headers and footers. Keep formatting as simple as possible, returning strictly the worded content from the image and equations in math delimiters (i.e., $...$). Any text in French must be translated to English. Your response must only contain the specified content of the image without enclosing your response in a specific wrapper and with no accompanying explanations. "},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"  # input budget 65 tokens
                            }
                        },
                    ],
                }
            ],
            max_tokens=4000,
        )
        responses.append(response.choices[0].message.content)
    return responses

def save_latex_as_json(latex_pages, file_path):
    with open(file_path, 'w') as file:
        json.dump(latex_pages, file, indent=4)

def create_course_material_latex_directory_structure(base_dir, course_dirs):
    os.makedirs(base_dir, exist_ok=True)
    for course_dir in course_dirs:
        os.makedirs(os.path.join(base_dir, course_dir), exist_ok=True)

def process_pdf_files(client, input_dir, output_dir):
    global_latex_content = []
    for root, dirs, files in os.walk(input_dir):
        relative_root = os.path.relpath(root, input_dir)
        if relative_root == ".":
            relative_root = ""
        for file in files:
            if file.endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                course_dir = os.path.join(output_dir, relative_root)
                json_file_name = os.path.splitext(file)[0] + '.json'
                json_file_path = os.path.join(course_dir, json_file_name)

                print(f"Processing PDF document: {file}")

                images = convert_pdf_to_images(pdf_path)
                latex_pages = query_openai_for_latex(images, client)
                save_latex_as_json(latex_pages, json_file_path)

                global_latex_content.extend(latex_pages)

                # Update course specific .json file
                course_json_path = os.path.join(course_dir, relative_root.replace(os.sep, '_') + '.json')
                if os.path.exists(course_json_path):
                    with open(course_json_path, 'r') as course_file:
                        course_latex_content = json.load(course_file)
                else:
                    course_latex_content = []
                
                course_latex_content.extend(latex_pages)
                save_latex_as_json(course_latex_content, course_json_path)

    # Save the global latex content
    global_json_path = os.path.join(output_dir, 'course_material_latex.json')
    save_latex_as_json(global_latex_content, global_json_path)

def main(api_key, organisation, input_dir):
    client = OpenAI(api_key=api_key, organization=organisation)
    output_dir = 'course_material_latex'
    
    # Get the list of course directories
    course_dirs = next(os.walk(input_dir))[1]
    
    # Create the directory structure in the output directory
    create_course_material_latex_directory_structure(output_dir, course_dirs)
    
    # Process all PDF files in the input directory
    process_pdf_files(client, input_dir, output_dir)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Convert PDF to LaTeX using OpenAI API')
    # parser.add_argument('--api_key', required=True, type=str, help='Your OpenAI API key')
    # parser.add_argument('--organisation', required=True, type=str, help='Your OpenAI organisation ID')
    # parser.add_argument('--input_dir', required=True, type=str, help='Path to the input directory containing PDFs')

    # args = parser.parse_args()

    # main(args.api_key, args.organisation, args.input_dir)

    api_key = "..."
    organisation = "..."
    input_dir = "course_material"

    main(api_key, organisation, input_dir)

