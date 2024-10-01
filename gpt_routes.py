# main.py
from flask import Flask, jsonify, request
import openai
import json

def gpt_routes(app):
   
    @app.route('/get_structured_nutrition_facts', methods=['POST'])
    def get_structured_nutrition_facts():
        try:

            input_data = request.json
            user_input = input_data.get('text')
            bmi = input_data.get('bmi', 'not given')  # Default to "unknown" if not provided
            age = input_data.get('age', 'not given')  # Default to "unknown" if not provided
            users_health_concerns = input_data.get('users_health_concerns', 'not given')  # Default to "unknown"
            users_ailments = input_data.get('users_ailments', 'not given')  # Default to "unknown"


            if not user_input:
                return jsonify({'error': 'No input text provided'}), 400
            
            prompt = f"""

            Given the following user information:
            bmi: {bmi}
            age: {age}
            users_health_concerns: {users_health_concerns}
            users_ailments: {users_ailments}

            And given the following nutrition label text:

            \"{user_input}\"

            Please extract the nutrition information and then provide a recommendation on whether the given food item is good or bad for the user, taking into account the user's BMI, age, family history, health concerns, and ailments.
            Also consider the ethnicity of the user and provide a recommendation based on that, taking into account common health concerns and dietary habits of that demographic.
            Then convert it into the following JSON format:

            {{
            "recommendation": "<recommendation>",
            "servings_per_container": "<number_of_servings>" // in string format,
            "serving_size": "<serving_size>",
            "calories": "<calories>" // in string format,
            "nutrients": [
                {{
                "name": "Total Fat",
                "amount": "<amount>",
                "daily_value": "<daily_value>"
                }},
                {{
                "name": "Saturated Fat",
                "amount": "<amount>",
                "daily_value": "<daily_value>"
                }},
                ...
            ]
            }}

            Only return valid JSON with all the details extracted from the text. Do not give it in markdown format.
            """
            
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}, {"role": "system", "content": "You are a helpful assistant."}],
                temperature=0.3,
            )

            result = response.choices[0].message.content
            
            # Try to parse `result` as JSON if it's a valid JSON string
            try:
                parsed_result = json.loads(result)  # Convert the string result into a Python dictionary
            except json.JSONDecodeError:
            # If it's not a valid JSON, just treat it as a string
                parsed_result = result

            # Return the parsed result (as JSON if it was valid JSON)
            return jsonify({"result": parsed_result}), 200

        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
    @app.route('/get_ingredient_info', methods=['POST'])
    def get_ingredient_info():
        try:

            input_data = request.json
            text = input_data.get("text")
            allergens = input_data.get("allergens", "unknown")

            if not text:
                return jsonify({'error': 'No input text provided'}), 400
            
            prompt = f"""

            Users allergens: \"{allergens}\"

            Given the following text of ingredients:
            \"{text}\"

            Please extract the ingredients and provide a detailed and helpful description of each ingredient information as well as their benefits and detriments. Make the description as detailed and technical as possible.
            Then provide a conclusion on whether it is a good ingredient or bad, also take into account the users allergies. If the ingredient could be allergic to the user, raise an allergy warning (change to true).
            Then convert it into the following JSON format:

            {{
            "ingredients": [
            {{
                "ingredient_name": "<name>",
                "description": "<description>",
                "benefits": "<benefits>",
                "detriments": "<detriments>",
                "conclusion": "<conclusion>",
                "allergy_warning": false
            }},
            {{
                "ingredient_name": "<name>",
                "description": "<description>",
                "benefits": "<benefits>",
                "detriments": "<detriments>",
                "conclusion": "<conclusion>",
                "allergy_warning": false
            }},
            ...
            ]
            }}


            Only return valid JSON with all the details extracted from the text. If you cannot determine a field, leave it with the value "unknown". Do not give it to me in markdown format.
            """

            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}, {"role": "system", "content": "You are a nutritionist and you have my health as your top concern."}],
                temperature=0.5,
            )

            result = response.choices[0].message.content
            
            # Try to parse `result` as JSON if it's a valid JSON string
            try:
                parsed_result = json.loads(result)  # Convert the string result into a Python dictionary
            except json.JSONDecodeError:
            # If it's not a valid JSON, just treat it as a string
                parsed_result = result
            
            # Return the parsed result (as JSON if it was valid JSON)
            return jsonify({"result": parsed_result}), 200
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
