import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI

# 1. 환경 변수 로드 (.env 파일에서 OPENAI_API_KEY 가져오기)
load_dotenv()

def build_and_test_rag():
    print("1. Java 소스 코드 로드 중...")
    # mock_framework 폴더 내의 모든 .java 파일을 읽어옵니다.
    loader = DirectoryLoader(
        'mock_framework', 
        glob="**/*.java", 
        loader_cls=TextLoader,
        loader_kwargs={'autodetect_encoding': True}
    )
    docs = loader.load()
    print(f"-> 총 {len(docs)}개의 Java 파일을 찾았습니다.\n")

    print("2. Java 문법 기반으로 코드 청킹(Chunking) 중...")
    # 핵심! 일반 텍스트가 아닌 Java 문법(클래스, 메서드 등) 기준으로 코드를 분할합니다.
    java_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JAVA,
        chunk_size=1500,
        chunk_overlap=150
    )
    split_docs = java_splitter.split_documents(docs)
    print(f"-> 총 {len(split_docs)}개의 의미 있는 코드 조각(Chunk)으로 분할되었습니다.\n")

    print("3. Vector DB (Chroma) 구축 및 임베딩 중...")
    # OpenAI 임베딩 모델을 사용하여 Chroma DB에 저장합니다.
    # persist_directory를 지정하면 로컬 폴더('./chroma_db')에 DB가 저장되어 매번 비용을 낼 필요가 없습니다.
    embeddings = AzureOpenAIEmbeddings(
        model=os.getenv("AOAI_DEPLOY_EMBED_3_LARGE"),
        openai_api_version="2024-02-01",
        api_key= os.getenv("AOAI_API_KEY"),
        azure_endpoint=os.getenv("AOAI_ENDPOINT")
        )    
    # 저장할 경로 지정
    DB_PATH = "./chroma_db"
    # 문서를 디스크에 저장합니다. 저장시 persist_directory에 저장할 경로를 지정합니다.
    vectorstore = Chroma.from_documents(
        split_docs, embeddings, persist_directory=DB_PATH, collection_name="my_db"
    )
    print("-> Vector DB 구축 완료!\n")

    print("4. Retriever 테스트 (검색 실행)")
    # Vector DB를 검색기(Retriever)로 변환 (MMR 검색 방식 도입 및 k값 증가)
    retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={
            "k": 3,          # 최종적으로 가져올 문서 개수 (3개로 증가)
            "fetch_k": 10,   # 일단 임시로 10개를 찾아온 뒤, 그 안에서 가장 다양성이 높은 3개를 고름
            "lambda_mult": 0.5 # 0에 가까울수록 다양성 중시, 1에 가까울수록 유사도(정확도) 중시
        }
    )

    # 사용자의 자연어 질문 (DB 페이징과 사번 조회를 동시에 물어봄)
    query = "DB에서 데이터를 리스트로 가져올 때 페이징 처리는 어떻게 해? 그리고 로그인한 사람 사번은 어떻게 가져와?"
    print(f"질문: {query}\n")
    
    results = retriever.invoke(query)
    
    print("=== 검색된 사내 프레임워크 소스 코드 ===")
    for i, res in enumerate(results):
        print(f"\n[결과 {i+1}] 출처: {res.metadata.get('source')}")
        print("-" * 50)
        print(res.page_content)
        print("-" * 50)

if __name__ == "__main__":
    build_and_test_rag()