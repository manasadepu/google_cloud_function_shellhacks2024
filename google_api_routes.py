from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
import os, io, re, cv2
from google.cloud import vision
from google.cloud import documentai_v1 as documentai
from google.oauth2 import service_account

# Google Vision API client
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "automatic-hawk-437002-u3-6d369553f0e5.json"
vision_client = vision.ImageAnnotatorClient()

# Document AI processor
def online_process(project_id: str, location: str, processor_id: str, file_path: str, mime_type: str):
    opts = {"api_endpoint": f"{location}-documentai.googleapis.com"}
    documentai_client = documentai.DocumentProcessorServiceClient(client_options=opts)
    resource_name = documentai_client.processor_path(project_id, location, processor_id)

    with open(file_path, "rb") as image:
        image_content = image.read()
        raw_document = documentai.RawDocument(content=image_content, mime_type=mime_type)
        request = documentai.ProcessRequest(name=resource_name, raw_document=raw_document)
        result = documentai_client.process_document(request=request)

    return result.document

# Utility to trim text
def trim_text(text: str):
    return text.strip().replace("\n", " ")

# Extract nutrition data
def nutrition_data_extraction(file_path: str):
    document = online_process(
        project_id="167699591040",
        location="us",
        processor_id="dfdb4140c4c607c6",
        file_path=file_path,
        mime_type="image/jpeg",
    )

    extracted_data = []
    for page in document.pages:
        detected_fields = []
        for defect in page.image_quality_scores.detected_defects:
            detected_fields.append({
                "defect": defect.type_,
                "confidence": defect.confidence,
            })

        extracted_data.append({
            "text": trim_text(document.text),
            "imageQualityScore": page.image_quality_scores.quality_score,
            "detectedDefects": detected_fields,
        })

    return extracted_data

# Correct perspective in image
def correct_perspective(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.boundingRect(largest_contour)
    x, y, w, h = rect
    roi = image[y:y + h, x:x + w]
    return roi

# Preprocess image for OCR
def image_preprocess(image_path):
    img = cv2.imread(image_path)
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise_reduce = cv2.fastNlMeansDenoising(gray_scale, None, 30, 7, 21)
    thresh = cv2.adaptiveThreshold(noise_reduce, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    processed_img = correct_perspective(thresh)
    
    current_dir = os.getcwd()
    preprocessImgPath = os.path.join(current_dir, "preprocessedImg.jpg")
    cv2.imwrite(preprocessImgPath, processed_img)
    
    return preprocessImgPath

# Detect text using Google Vision OCR
def text_detect(image_path):
    with io.open(image_path, 'rb') as image:
        content = image.read()

    image = vision.Image(content=content)
    response = vision_client.document_text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(f"{response.error.message}")

    return texts[0].description if texts else ''

# Extract ingredients from text
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

# Combined endpoint for both nutrition and ingredient OCR
def google_api_routes(app):

    @app.route('/perform_nutrition_and_ingredient_OCR', methods=['POST'])
    def perform_nutrition_and_ingredient_OCR():
        if 'nutrition_file' not in request.files or 'ingredient_file' not in request.files:
            return jsonify({'error': 'Nutrition file or Ingredient file part is missing in the request'}), 400

        nutrition_file = request.files['nutrition_file']
        ingredient_file = request.files['ingredient_file']

        if nutrition_file.filename == '' or ingredient_file.filename == '':
            return jsonify({'error': 'No selected file for either nutrition or ingredient OCR'}), 400

        try:
            # Save files locally
            current_dir = os.getcwd()
            nutrition_file_path = os.path.join(current_dir, secure_filename(nutrition_file.filename))
            ingredient_file_path = os.path.join(current_dir, secure_filename(ingredient_file.filename))
            nutrition_file.save(nutrition_file_path)
            ingredient_file.save(ingredient_file_path)

            # Perform nutrition data extraction
            nutrition_data = nutrition_data_extraction(nutrition_file_path)

            # Preprocess ingredient image and perform OCR
            processed_image_path = image_preprocess(ingredient_file_path)
            ocr_text = text_detect(processed_image_path)

            # Extract ingredients from OCR text
            ingredients = extract_ingredients(ocr_text)

            # Return combined response
            return jsonify({
                'nutrition_data': nutrition_data,
                'ingredient_data': {
                    'ocr_text': ocr_text,
                    'ingredients': ingredients
                }
            }), 200

        except Exception as e:
            return jsonify({'error': str(e)}), 500
