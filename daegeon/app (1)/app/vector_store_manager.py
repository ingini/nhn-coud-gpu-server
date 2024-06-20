from pathlib import Path
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS

class VectorStoreManager:
    def __init__(self, model_name, model_kwargs, encode_kwargs, cache_dir):
        self.file_store = LocalFileStore(cache_dir)
        huggingface_embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        self.embedder = CacheBackedEmbeddings.from_bytes_store(huggingface_embeddings, self.file_store)
        self.faiss_index_path = Path('faiss-qnaData/index.faiss')
        # self.bm25_index_path = Path('bm25-qnaData/index.faiss')
        self.vector_store = self.load_vector_store()

    def load_vector_store(self):
        if self.faiss_index_path.exists():
            return FAISS.load_local('faiss-qnaData', self.embedder, allow_dangerous_deserialization=True)
        return None

    def create_index(self, data):
        documents = []
        for item in data:
            documents.append(item)
        vectorstore = FAISS.from_documents(documents, embedding=self.embedder)
        vectorstore.save_local('faiss-qnaData')