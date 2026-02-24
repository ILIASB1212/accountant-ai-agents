from langchain_chroma import Chroma
from logger.logger import get_logger
from exception.custom_exception import CustomException
from pathlib import Path

logging = get_logger(__name__)

class VectorStore:
    def __init__(self, embeddings, persist_directory):
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.vectorstore = None
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        logging.info(f"✅ VectorStore initialized with persist_dir: {persist_directory}")

    def create_from_documents(self, documents):
        """Create vectorstore from documents"""
        try:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            logging.info(f"✅ Created vectorstore with {len(documents)} documents")
            return self.vectorstore
        except Exception as e:
            logging.error(f"❌ Error creating vectorstore: {e}")
            raise CustomException("Failed to create vectorstore", e)

    def load_existing(self):
        """Load existing vectorstore"""
        try:
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            logging.info(f"✅ Loaded existing vectorstore from {self.persist_directory}")
            return self.vectorstore
        except Exception as e:
            logging.error(f"❌ Error loading vectorstore: {e}")
            raise CustomException("Failed to load vectorstore", e)

    def get_retriever(self, k: int = 4):
        """Get retriever from vectorstore"""
        if self.vectorstore is None:
            raise CustomException("Vectorstore not created yet. Call create_from_documents first.")
        return self.vectorstore.as_retriever(search_kwargs={"k": k})