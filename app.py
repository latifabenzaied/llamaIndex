from GeminiService import GeminiService
import os

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Configuration de la clé API
# GOOGLE_API_KEY = "AIzaSyDTIC11PJHkOs6PkpE-yDNsvjddLYrbNpc"

# Exemple d'offre et de CV (le chemin du CV doit être correct)
offer = """Titre : Développeur Full Stack (H/F)
Entreprise : XYZ Tech Solutions
Lieu : Paris, France
Type de contrat : CDI
Salaire : 45 000€ par an + avantages
"""

cv_path = os.path.join("data", "latifa.pdf")
# Initialisation des services
gemini_service = GeminiService(GOOGLE_API_KEY)

# Extraction du texte du CV à partir du fichier PDF
cv_text = gemini_service.get_cv_text_from_pdf(cv_path)

# Analyser la compatibilité entre l'offre et le CV
response = gemini_service.ask_gemini(offer, cv_text)

print(response)
