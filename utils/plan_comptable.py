from src.data_ingestion.documents_loader import DocumentLoader
from src.data_ingestion.embedding import  Embeddings
from src.data_ingestion.text_spliter import TextSpliter
from src.data_ingestion.vectorestore import VectorStore
from src.PipeLine.pipeline import RagPipeLine


DATA_DIR = r"C:\Users\VICTUS\Desktop\accountant\data\PLAN_COMPTABLE"  # Change per file
PERSIST_DIR = "vectorestore/db_plan_comptable"  # Change per file
FORCE_REBUILD = False  # Set to True only when you want to rebuild

rag=RagPipeLine(data_dir=DATA_DIR,
            persist_dir=PERSIST_DIR,force_rebuild=False,chunk_size=1000,chunk_overlap=250)
retriever=rag.run()

### Retriever To Retriever Tools
from langchain_classic.tools.retriever import create_retriever_tool
plan_comptable_tool = create_retriever_tool(
    retriever,
    "plan_comptable_tool",  # Simplified name
    """Use this tool to help in accountant operations:
    - Moroccan accounting 
    - les classe de bilan 
    - can work with code general de normalisation comptable to get specefic numbers 
    - use it if you want to comptabilise une facture ou une operation
    - Chart of accounts (Plan comptable marocain)
    - Accounting treatments specific to Morocco
    
    This is your PRIMARY tool for classes comptabilisation questions. ALWAYS use this first to know the classes for comptabilisation des operation queries."""
)
