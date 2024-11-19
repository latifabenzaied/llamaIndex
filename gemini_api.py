from dotenv import load_dotenv
from flask import Flask, request, jsonify
from GeminiService import GeminiService
import os

# Crée une instance de l'application Flask
app = Flask(__name__)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialisation des services
gemini_service = GeminiService(GOOGLE_API_KEY)

@app.route('/check-compatibility', methods=['POST'])
def check_compatibility():
    """Endpoint pour vérifier la compatibilité entre l'offre et le CV"""
    data = request.get_json()
    offer = data.get('offer')
    cv_path = data.get('cv_path')


    if not offer or not cv_path:
        return jsonify({"error": "L'offre et le chemin du CV sont requis"}), 400

    # Extraction du texte du CV et analyse de la compatibilité
    cv_text = gemini_service.get_cv_text_from_pdf(cv_path)
    response = gemini_service.ask_gemini(offer, cv_text)

    return jsonify({"response": response})
# Lancer l'application Flask
if __name__ == '__main__':
    app.run()
