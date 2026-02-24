from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from exception.custom_exception import CustomException
from logger.logger import get_logger
logging=get_logger(__name__)

class Embeddings:

    def __init__(self,
        model_name:str="sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize_embeddings: bool = True,
        batch_size: int = 32):

        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self._embeddings = None
        logging.info("embedding get initialized")


    def initializing_embedding(self):
        if self._embeddings is None:
            try:
                self.embeddings = HuggingFaceEmbeddings(
                        model_name=self.model_name,
                        model_kwargs={'device': self.device},
                        encode_kwargs={
                            'normalize_embeddings': self.normalize_embeddings,
                            'batch_size': self.batch_size
                        }
                    )
            except Exception as e:
                    logging.error(f"error during initilizing : {e}")
                    raise CustomException(
                        message=f"Failed to initialize embeddings with model {self.model_name} or their is an error in embeding models",
                        error_detail=e
                    )
        return self.embeddings