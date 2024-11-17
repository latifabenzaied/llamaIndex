from llama_index.readers.file import PDFReader
from llama_index.core import VectorStoreIndex



class DocumentIndexer:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.documents = None
        self.index = None

    def load_documents(self):
        """Charge les documents à partir du fichier PDF"""
        pdf_reader = PDFReader()
        self.documents = pdf_reader.load_data(file=self.pdf_path)

    def create_index(self):
        """Crée un index à partir des documents chargés"""
        if self.documents is None:
            raise ValueError("Les documents n'ont pas été chargés.")
        self.index = VectorStoreIndex.from_documents(self.documents, show_progress=True)

    def get_index(self):
        """Retourne l'index généré"""
        return self.index
