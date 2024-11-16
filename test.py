from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.readers.file import PDFReader
import os

GOOGLE_API_KEY = "AIzaSyDTIC11PJHkOs6PkpE-yDNsvjddLYrbNpc"
llm= Gemini(
    model="models/gemini-1.5-flash",
    api_key=GOOGLE_API_KEY,
)
Settings.embed_model = GeminiEmbedding(api_key=GOOGLE_API_KEY)
Settings.llm =llm
pdf_path = os.path.join("data", "latifa.pdf")
documents = PDFReader().load_data(file=pdf_path)
index = VectorStoreIndex.from_documents(documents,show_progress=True)



def ask_gemini(query):
    # Interroger LlamaIndex pour obtenir des informations pertinentes

    query_engine = index.as_query_engine(llm=llm)
    response =query_engine.query(query)
    # prompt = f"Voici un résumé du fichier PDF que vous avez chargé : {documents}\nRépondez à la question suivante en utilisant ces informations : {query}"
    prompt = f"""
    Je vais te fournir une description d'offre d'emploi et un CV de candidat. 
    Peux-tu répondre par 'Oui' ou 'Non' à la question suivante, puis fournir une justification breve de ta réponse ?
    Le candidat est-il compatible avec l'offre d'emploi ci-dessous ?

    1. Description de l'offre : 
    Titre : Développeur Full Stack (H/F)
    Entreprise : XYZ Tech Solutions
    Lieu : Paris, France
    Type de contrat : CDI
    Salaire : 45 000€ par an + avantages

    Responsabilités :
    - Développement et maintenance des applications web
    - Collaboration avec les équipes de conception et de gestion de produits
    - Mise en œuvre de solutions backend et développement frontend
    - Revues de code et bonnes pratiques
    - Gestion des performances, sécurité et scalabilité

    Compétences requises :
    - Maîtrise des langages de programmation : JavaScript, TypeScript, HTML, CSS
    - Expérience avec les frameworks frontend : React, Angular ou Vue.js
    - Connaissance des technologies backend : Node.js, Express, MongoDB ou SQL
    - Bonne compréhension des API RESTful
    - Expérience dans l'intégration continue et les tests automatisés

    Avantages :
    - 5 jours de congés supplémentaires
    - Télétravail flexible
    - Formation continue et opportunités d’évolution

    2. CV du candidat : 
    {documents}
    """
    # Utiliser Gemini pour générer une réponse plus avancée
    gemini_response = llm.complete(
       prompt=prompt,
    )
    return gemini_response.text
query = "Quels sont les principaux points abordés dans ce CV ?"
gemini_answer = ask_gemini(query)
print(f"{gemini_answer}")


# print(documents)
# pdf_data = PDFReader().load_data(file=pdf_path)
# documents = (SimpleDirectoryReader(input_dir='C:\\Users\\latifa\\Desktop\\latifa.pdf', recursive=True).
#              load_data())
# Créer un index à partir des documents chargés
