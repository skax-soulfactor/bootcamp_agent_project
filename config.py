import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma

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