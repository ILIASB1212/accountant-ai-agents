from src.data_ingestion.documents_loader import DocumentLoader
from src.data_ingestion.embedding import  Embeddings
from src.data_ingestion.text_spliter import TextSpliter
from src.data_ingestion.vectorestore import VectorStore
from src.PipeLine.pipeline import RagPipeLine

DATA_DIR = r"C:\Users\VICTUS\Desktop\accountant\data\LOIS_DE_FINANCE"  # Change per file
PERSIST_DIR = "vectorestore/db_lois_de_finance"  # Change per file
FORCE_REBUILD = False  # Set to True only when you want to rebuild

rag=RagPipeLine(data_dir=DATA_DIR,
            persist_dir=PERSIST_DIR,force_rebuild=False,chunk_size=1000,chunk_overlap=250)
retriever=rag.run()
from langchain_classic.tools.retriever import create_retriever_tool
finance_law_tool = create_retriever_tool(
    retriever,
    "morocco_finance_law_tool",  # Simplified name
    """Use this tool for ANY questions about:
    - Moroccan finance law 
    - Moroccan tax
    - Any question containing words like: law, moroccan finance law, Moroccan lois, 
    
    This is your PRIMARY tool for Moroccan finance law questions. ALWAYS use this first for finance law  queries."""
)
