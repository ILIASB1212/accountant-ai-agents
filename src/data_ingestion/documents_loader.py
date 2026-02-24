from langchain_community.document_loaders import PyPDFDirectoryLoader
from logger.logger import get_logger
from exception.custom_exception import CustomException

logging=get_logger(__name__)



class DocumentLoader:
    def __init__(self,directory:str,extract_images:bool=False):
        self.directory=directory
        self.extract_images=extract_images
        logging.info("initialized docuemnt loader")

    def document_loader(self):
        rew_documents=PyPDFDirectoryLoader(self.directory,extract_images=self.extract_images)
        self.docs=rew_documents.load()
        if self.docs:
            try:
                pdf_files = set()
                for doc in self.docs:
                    source = doc.metadata.get('source', 'unknown')
                    pdf_files.add(source)
                # Log each PDF file
                logging.info(f"✅ Successfully loaded {len(self.docs)} documents from {len(pdf_files)} PDF files:")
                for pdf in pdf_files:
                    logging.info(f" 📄 {pdf}")
            except Exception as e:
                    logging.error(f"error in loading documents : {e}")
                    raise CustomException(
                        message=f"Failed to load documents {e}",
                        error_detail=e
                    )
        elif not self.docs:
            logging.warning(f"⚠️ No documents found in {self.directory}")
            print(f"⚠️ No documents found in {self.directory}")
            return []
         
        return self.docs
    
    def get_document_count(self):
        """Get number of documents in directory without loading them"""
        try:
            loader = PyPDFDirectoryLoader(self.directory, extract_images=self.extract_images)
            docs = loader.load()
            return f"documents count : {len(docs)}"
        except Exception as e:
            logging.warning(f"Could not get document count: {e}")
            return 0



