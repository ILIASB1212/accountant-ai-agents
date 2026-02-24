from src.data_ingestion.documents_loader import DocumentLoader
from src.data_ingestion.embedding import  Embeddings
from src.data_ingestion.text_spliter import TextSpliter
from src.data_ingestion.vectorestore import VectorStore



document_loader=DocumentLoader(directory=r"C:\Users\VICTUS\Desktop\accountant\data\CGNC",extract_images=False)
embeddings_model=Embeddings()
text_spliter=TextSpliter(chunk_overlap=200)

documents=document_loader.document_loader()
print(f"📄 Loaded {len(documents)} documents")
if not documents:
    print("❌ No documents to process. Please add PDF files to data/CGNC/")
    exit()  # or return

embedings=embeddings_model.initializing_embedding()
print("🔤 Embeddings ready")

vectorstore =VectorStore(embeddings=embedings,
                          persist_directory="vectorestore/db_cgnc")

chunks =text_spliter.split_documents(documents)
if not chunks:
    print("❌ No chunks created. Check your PDF files.")
    exit()
    
db = vectorstore.create_from_documents(chunks)
print(f"💾 Vectorstore created with {len(chunks)} documents")
retriver=vectorstore.get_retriever()


### Retriever To Retriever Tools
from langchain_classic.tools.retriever import create_retriever_tool
cgnc_tool=create_retriever_tool(
    retriver,
    "code_general_de_normalisation_comptable",
    "Search accountant information"
)


