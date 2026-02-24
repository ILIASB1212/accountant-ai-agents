from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from logger.logger import get_logger
from exception.custom_exception import CustomException

logging=get_logger(__name__)



class TextSpliter:
    def __init__(self,chunk_size:int=800,chunk_overlap:int=200,persist_directory:str="vectorestore_VDB"):
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap
        self.persist_directory=persist_directory
        logging.info(f"✅ TextSplitter initialized with chunk_size={chunk_size}, overlap={chunk_overlap}")


    
    def split_documents(self,documents):
        # Split
        try:
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
            chunks = text_splitter.split_documents(documents)
            logging.info(f"✅ Split {len(documents)} documents into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logging.error(f"❌ Error splitting documents: {e}")
            raise CustomException("Failed to split documents", e)





