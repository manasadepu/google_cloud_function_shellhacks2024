import json
import os, io, re, cv2
from google.cloud import vision
from google.cloud import documentai_v1 as documentai
from google.oauth2 import service_account

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "automatic-hawk-437002-u3-6d369553f0e5.json"


# Instantiates a client
vision_client = vision.ImageAnnotatorClient()

def online_process(
    project_id: str,
    location: str,
    processor_id: str,
    file_path: str,
    mime_type: str,
) -> documentai.Document:
    """
    Processes a document using the Document AI Online Processing API.
    """

    opts = {"api_endpoint": f"{location}-documentai.googleapis.com"}

    # Instantiates a client
    documentai_client = documentai.DocumentProcessorServiceClient(client_options=opts)

    # The full resource name of the processor, e.g.:
    # projects/project-id/locations/location/processor/processor-id
    # You must create new processors in the Cloud Console first
    resource_name = documentai_client.processor_path(project_id, location, processor_id)

    # Read the file into memory
    with open(file_path, "rb") as image:
        image_content = image.read()

        # Load Binary Data into Document AI RawDocument Object
        raw_document = documentai.RawDocument(
            content=image_content, mime_type=mime_type
        )

        # Configure the process request
        request = documentai.ProcessRequest(
            name=resource_name, raw_document=raw_document
        )

        # Use the Document AI client to process the sample form
        result = documentai_client.process_document(request=request)

        return result.document


def trim_text(text: str):
    """
    Remove extra space characters from text (blank, newline, tab, etc.)
    """
    return text.strip().replace("\n", " ")

def nutrition_data_extraction(file_path: str):

    document = online_process(
        project_id="167699591040",
        location="us",
        processor_id="dfdb4140c4c607c6",
        file_path=file_path,
        mime_type="image/jpeg",
    )

# Prepare a list of dictionaries for storing field names, values, and their confidence
    extracted_data = []

    for page in document.pages:

        detected_fields = []
        for defect in page.image_quality_scores.detected_defects:
            detected_fields.append(
                {
                    "defect": defect.type_,
                    "confidence": defect.confidence,
                }
            )

        extracted_data.append(
            {
                "text": trim_text(document.text),
                "imageQualityScore": page.image_quality_scores.quality_score,
                "detectedDefects": detected_fields,
            }
        )

    return extracted_data

def correct_perspective(image):  
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    rect = cv2.boundingRect(largest_contour)
    x, y, w, h = rect
    roi = image[y:y + h, x:x + w]

    return roi

def image_preprocess(image_path):
    img = cv2.imread(image_path)
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise_reduce = cv2.fastNlMeansDenoising(gray_scale, None, 30, 7, 21)

    thresh = cv2.adaptiveThreshold(noise_reduce, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    processed_img = correct_perspective(thresh)

    # Save the preprocessed image in the /tmp directory
    #preprocessImgPath = os.path.join('/tmp', "preprocessedImg.jpg")
    current_dir = os.getcwd()
    preprocessImgPath = os.path.join(current_dir, "preprocessedImg.jpg")
    cv2.imwrite(preprocessImgPath, processed_img)

    return preprocessImgPath

def text_detect(image_path):
    with io.open(image_path, 'rb') as image:
        content = image.read()

    image = vision.Image(content=content)
    response = vision_client.document_text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(f"{response.error.message}")

    return texts[0].description if texts else ''

def extract_ingredients(text):
    ingredient_pattern = re.compile(r'(?i)(?:ingredients?\s*:?\s*)(.*?)(?=\n\s*contains:|$)', re.DOTALL)
    match = ingredient_pattern.search(text)

    if match:
        ingredients_block = match.group(1)

        ingredients_block = ingredients_block.replace('\n', ' ')

        ingredients_list = re.split(r',\s*(?![^()]*\))', ingredients_block.strip())
        ingredients_list = [ingredient.strip() for ingredient in ingredients_list if ingredient.strip()]

        return ingredients_list

    return []
