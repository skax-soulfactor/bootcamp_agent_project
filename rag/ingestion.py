import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_chroma import Chroma

# 모듈화된 설정 가져오기
from config.settings import embeddings, DB_PATH, MOCK_FRAMEWORK_PATH

def build_vector_db():
    if not os.path.exists(MOCK_FRAMEWORK_PATH):
        raise FileNotFoundError(f"경로를 찾을 수 없습니다: {MOCK_FRAMEWORK_PATH}")

    loader = DirectoryLoader(
        MOCK_FRAMEWORK_PATH, 
        glob="**/*.java", 
        loader_cls=TextLoader,
        loader_kwargs={'autodetect_encoding': True}
    )
    docs = loader.load()
    
    java_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JAVA, chunk_size=1500, chunk_overlap=150
    )
    split_docs = java_splitter.split_documents(docs)
    
    Chroma.from_documents(
        documents=split_docs, 
        embedding=embeddings, 
        persist_directory=DB_PATH, 
        collection_name="my_db"
    )
    return len(split_docs)