from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.settings import Settings
from indexer import DocumentIndexer

class GeminiService:
    def __init__(self, google_api_key, model="models/gemini-1.5-flash"):
        self.google_api_key = google_api_key
        self.llm = Gemini(
            model=model,
            api_key=self.google_api_key
        )
        Settings.embed_model = GeminiEmbedding(api_key=self.google_api_key)
        Settings.llm = self.llm

    def ask_gemini(self, offer, cv_text):
        """Demande au modèle Gemini d'analyser la compatibilité"""
        prompt = f"""
       Je vais te fournir une description d'offre d'emploi et un CV. 
       Réponds brièvement à la question suivante comme si tu étais un responsable RH en t'adressant directement au candidat avec 'vous'.
       Si la candidature est refusée, explique en une phrase la raison du refus en utilisant 'vous'. Si la réponse est positive, réponds simplement 'Oui' sans justification.
       Es-tu compatible avec l'offre d'emploi ci-dessous ?

        1. Description de l'offre : 
        {offer}

        2. CV du candidat : 
        {cv_text}
        """

        gemini_response = self.llm.complete(prompt=prompt)
        return gemini_response.text

    def get_cv_text_from_pdf(self, pdf_path):
        """Extrait le texte du CV à partir du fichier PDF"""
        document_indexer = DocumentIndexer(pdf_path)
        document_indexer.load_documents()
        document_indexer.create_index()
        query_engine = document_indexer.get_index().as_query_engine(llm=self.llm)
        print(query_engine)
        return query_engine.query("Extraire le texte du CV")
