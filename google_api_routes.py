import os
from flask import jsonify, request
from werkzeug.utils import secure_filename
from helper_functions import nutrition_data_extraction, image_preprocess, text_detect, extract_ingredients

def google_api_routes(app):

    @app.route('/perform_nutrition_label_OCR', methods=['POST'])
    def perform_nutrition_label_OCR():
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file:

            filename = secure_filename(file.filename)
            #file_path = os.path.join('/tmp', filename)
            current_dir = os.getcwd()
            file_path = os.path.join(current_dir, filename)
            file.save(file_path)

            try:
                result = nutrition_data_extraction(file_path)
                return jsonify(result), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500
            
    
    @app.route('/perform_ingredient_ocr', methods=['POST'])
    def perform_ingredient_ocr():
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file:
            filename = secure_filename(file.filename)
            #file_path = os.path.join('/tmp', filename)  # Save to /tmp for Google Cloud Functions
            current_dir = os.getcwd()
            file_path = os.path.join(current_dir, filename)
            file.save(file_path)

            try:
                # Preprocess image
                processed_image_path = image_preprocess(file_path)
                
                # Detect text via OCR
                ocr_text = text_detect(processed_image_path)
                print("OCR Detected Text:\n", ocr_text)

                # Extract ingredients from detected text
                ingredients = extract_ingredients(ocr_text)
                
                # Return extracted ingredients as JSON
                return jsonify({
                    'ocr_text': ocr_text,
                    'ingredients': ingredients
                }), 200

            except Exception as e:
                return jsonify({'error': str(e)}), 500