#imports

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

#hardcoded variables
DATA_PATH = "E:/LLMS/data"
DB_FAISS_PATH = "E:/LLMS/vectorstores/db_faiss"

#create vector DB
def create_vector_db():
    #load and split docs
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    texts = text_splitter.split_documents(documents)
    
    #embeddding
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', model_kwargs = {'device':'cpu'} )

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == '__main__':
    create_vector_db()


