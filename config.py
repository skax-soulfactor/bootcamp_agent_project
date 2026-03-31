import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma

# 🚀 DB 구축을 위한 라이브러리 추가 임포트
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# 1. 환경 변수 로드
load_dotenv()

# 2. LLM 설정
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AOAI_ENDPOINT"),
    azure_deployment=os.getenv("AOAI_DEPLOY_GPT4O"),
    api_version="2024-10-21",
    api_key=os.getenv("AOAI_API_KEY")
)

# 3. Embedding 설정
embeddings = AzureOpenAIEmbeddings(
    model=os.getenv("AOAI_DEPLOY_EMBED_3_LARGE"),
    openai_api_version="2024-02-01",
    api_key=os.getenv("AOAI_API_KEY"),
    azure_endpoint=os.getenv("AOAI_ENDPOINT")
)    

# 4. Vector DB 및 Retriever 설정
DB_PATH = "./chroma_db"
vectorstore = Chroma(
    persist_directory=DB_PATH, 
    embedding_function=embeddings, 
    collection_name="my_db"
)

retriever = vectorstore.as_retriever(
    search_type="mmr", 
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5}
)

# ==========================================
# 🚀 5. DB 자동 구축 함수 (Streamlit에서 호출)
# ==========================================
def build_vector_db():
    print("사내 소스 코드 로드 및 Vector DB 구축 시작...")
    
    # 1. 문서 로드
    loader = DirectoryLoader(
        'mock_framework', 
        glob="**/*.java", 
        loader_cls=TextLoader,
        loader_kwargs={'autodetect_encoding': True}
    )
    docs = loader.load()
    
    # 2. 청킹 (chunk_size=1500으로 메서드 보존)
    java_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JAVA,
        chunk_size=1500,
        chunk_overlap=150
    )
    split_docs = java_splitter.split_documents(docs)
    
    # 3. Chroma DB에 밀어넣기
    Chroma.from_documents(
        documents=split_docs, 
        embedding=embeddings, 
        persist_directory=DB_PATH, 
        collection_name="my_db"
    )
    print(f"✅ 총 {len(split_docs)}개의 코드 조각이 DB에 저장되었습니다.")
    return len(split_docs)