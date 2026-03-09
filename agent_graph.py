import os
from typing import TypedDict, List
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

# 환경 변수 로드
load_dotenv()

# ==========================================
# 1. 상태(State) 정의
# 에이전트들이 대화하며 주고받을 데이터 구조입니다.
# ==========================================
class AgentState(TypedDict):
    question: str       # 사용자의 질문
    intent: str         # 라우터가 파악한 의도 ('code_request' 또는 'general')
    context: List[str]  # RAG를 통해 검색된 사내 프레임워크 코드/문서
    generation: str     # 최종 생성된 답변(코드)

# ==========================================
# 2. LLM 및 Retriever (Step 2에서 만든 DB 로드)
# ==========================================
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AOAI_ENDPOINT"),
    azure_deployment=os.getenv("AOAI_DEPLOY_GPT4O"),
    api_version="2024-10-21",
    api_key=os.getenv("AOAI_API_KEY")
)
embeddings = AzureOpenAIEmbeddings(
    model=os.getenv("AOAI_DEPLOY_EMBED_3_LARGE"),
    openai_api_version="2024-02-01",
    api_key= os.getenv("AOAI_API_KEY"),
    azure_endpoint=os.getenv("AOAI_ENDPOINT")
)    

# 기존에 만들어둔 로컬 Chroma DB를 불러옵니다.
DB_PATH = "./chroma_db"
vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings, collection_name="my_db")
retriever = vectorstore.as_retriever(
    search_type="mmr", 
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5}
)

# ==========================================
# 3. 노드(Node) 및 에이전트 함수 정의
# ==========================================

# [Node 1] 라우터 에이전트 (질문 의도 파악)
class RouteOutput(BaseModel):
    intent: str = Field(description="질문이 사내 프레임워크, 자바 코드 작성, DB/보안 등 개발과 관련되면 'code_request', 단순 인사나 일반적인 대화면 'general'을 반환")

def router_node(state: AgentState):
    print("--- [Agent: Router] 질문 의도 분석 중 ---")
    question = state["question"]
    
    # Structured Output을 활용해 무조건 'code_request'나 'general' 중 하나를 반환하게 강제함
    structured_llm = llm.with_structured_output(RouteOutput)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 사용자의 질문 의도를 분석하는 라우터입니다. 질문이 개발/코드 작성/사내 시스템과 관련되어 있다면 'code_request', 그렇지 않으면 'general'로 분류하세요."),
        ("user", "{question}")
    ])
    
    chain = prompt | structured_llm
    result = chain.invoke({"question": question})
    
    return {"intent": result.intent}

# [Node 2] 검색 에이전트 (RAG)
def retrieve_node(state: AgentState):
    print("--- [Agent: Retriever] 사내 프레임워크 문서/코드 검색 중 ---")
    question = state["question"]
    docs = retriever.invoke(question)
    
    # 검색된 문서들의 내용을 리스트 형태로 상태에 저장
    context = [f"[출처: {doc.metadata.get('source', '알수없음')}]\n{doc.page_content}" for doc in docs]
    return {"context": context}



# [Node 3] 코드 생성 에이전트 (이 부분을 찾아서 아래처럼 print 문을 추가해 주세요)
def generate_node(state: AgentState):
    print("--- [Agent: Code Generator] 사내 표준 기반 코드 생성 중 ---")
    
    # 🎯 디버깅용: Retriever가 진짜로 사내 코드를 잘 찾아왔는지 터미널에 출력해 봅니다.
    print("\n[디버그] 주입된 Context 내용:\n", state["context"]) 
    
    question = state["question"]
    context = "\n\n".join(state["context"])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 10년 차 사내 백엔드 자바 아키텍트입니다.
        반드시 아래 제공된 [사내 프레임워크 Context]만을 참고하여 질문에 답하고 코드를 작성하세요.
        Context에 없는 외부 라이브러리(Spring 기본 기능 등)를 마음대로 지어내지 마세요.
        답변 시, 참조한 클래스나 어노테이션에 대해 주석으로 친절히 설명해 주세요.
        
        [사내 프레임워크 Context]
        {context}
        """),
        ("user", "{question}")
    ])
    
    chain = prompt | llm
    result = chain.invoke({"question": question, "context": context})
    return {"generation": result.content}

# [Node 4] 일반 대화 에이전트
def general_qa_node(state: AgentState):
    print("--- [Agent: General QA] 일반적인 답변 생성 중 ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 친절한 AI 어시스턴트입니다. 개발과 관련 없는 질문에 짧고 친절하게 답해주세요."),
        ("user", "{state_question}")
    ])
    chain = prompt | llm
    result = chain.invoke({"state_question": state["question"]})
    return {"generation": result.content}

# ==========================================
# 4. 엣지(Edge) 조건 및 그래프 컴파일
# ==========================================
def decide_route(state: AgentState):
    if state["intent"] == "code_request":
        return "retrieve" # 코드가 필요하면 검색 노드로 이동
    else:
        return "general"  # 아니면 일반 QA 노드로 이동

workflow = StateGraph(AgentState)

# 노드 등록
workflow.add_node("router", router_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("general", general_qa_node)

# 흐름(엣지) 연결
workflow.set_entry_point("router") # 무조건 라우터부터 시작
workflow.add_conditional_edges(
    "router",
    decide_route,
    {
        "retrieve": "retrieve",
        "general": "general"
    }
)
workflow.add_edge("retrieve", "generate") # 검색이 끝나면 무조건 코드 생성으로
workflow.add_edge("generate", END)
workflow.add_edge("general", END)

# 최종 에이전트 그래프 컴파일
app = workflow.compile()

# ==========================================
# 5. 실행 테스트
# ==========================================
if __name__ == "__main__":
    print("\n=======================================================")
    print("🤖 DevX-Copilot Multi-Agent 테스트를 시작합니다.")
    print("=======================================================\n")
    
    # 테스트 1: 사내 프레임워크 관련 질문
    test_query_1 = "결제 내역을 조회하는 비즈니스 로직을 짤건데, DB 페이징 처리랑 현재 로그인한 사람 사번 가져오는 사내 표준 코드를 적용해서 Service 클래스로 만들어줘."
    print(f"User: {test_query_1}\n")
    
    result_1 = app.invoke({"question": test_query_1})
    print("\n[최종 답변]\n")
    print(result_1["generation"])
    
    print("\n" + "="*50 + "\n")
    
    # 테스트 2: 일반 인사 질문 (라우팅 테스트)
    test_query_2 = "안녕! 오늘 날씨 참 좋다 그치?"
    print(f"User: {test_query_2}\n")
    
    result_2 = app.invoke({"question": test_query_2})
    print("\n[최종 답변]\n")
    print(result_2["generation"])