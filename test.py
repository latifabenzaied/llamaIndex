
from llama_index.llms.gemini import Gemini
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader,ServiceContext


GOOGLE_API_KEY = "AIzaSyDGbmlr-CFPwZI9di2MnitQXVMbmKyLYm4"  
gemini_embedding = GeminiEmbedding(model_name="models/embedding-001")
llm = Gemini(
    model="models/gemini-1.5-flash",
    api_key=GOOGLE_API_KEY,  
)

documents = SimpleDirectoryReader('C:\\Users\\latifa\\Desktop\\test').load_data()
# Créer un index à partir des documents chargés
index = VectorStoreIndex.from_documents(documents, embed_model='local')


def ask_gemini(query):
    # Interroger LlamaIndex pour obtenir des informations pertinentes
    response = index.query(query)
    # Utiliser Gemini pour générer une réponse plus avancée
    gemini_response = llm.generate_text(
        model="models/gemini-1.5-flash",  # Remplacez par le modèle de Gemini que vous utilisez
        prompt=f"Répondez à la question suivante : {response['response']}",
        temperature=0.7,
    )
    
    return gemini_response.result


query = "Quel est le meilleur moyen d'indexer des documents en Python?"
gemini_answer = ask_gemini(query)
print(f"Réponse de Gemini : {gemini_answer}")


