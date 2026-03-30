import os
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage

# 최신 LangGraph 모듈들 (ToolNode, 조건부 라우팅, 메모리 등)
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# 환경 변수 로드
load_dotenv()

# ==========================================
# 1. LLM 및 DB 세팅 (기존과 동일, 최신 Chroma 반영)
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
    api_key=os.getenv("AOAI_API_KEY"),
    azure_endpoint=os.getenv("AOAI_ENDPOINT")
)    

DB_PATH = "./chroma_db"
vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings, collection_name="my_db")
retriever = vectorstore.as_retriever(
    search_type="mmr", 
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5}
)

# ==========================================
# 2. 🚀 도구(Tool) 정의 (심사위원 핵심 요구사항!)
# ==========================================
# @tool 데코레이터를 붙이면 LLM이 스스로 이 함수를 언제 호출할지 결정합니다.
@tool
def search_inhouse_framework(query: str) -> str:
    """사내 Java 프레임워크(DB, 보안, 페이징 등)의 소스코드와 사용법을 검색합니다.
    사용자가 Java 코드 작성을 요청하거나 사내 표준 가이드가 필요할 때 반드시 이 도구를 사용하세요.
    단순한 일상 대화나 인사말에는 사용하지 마세요."""
    print(f"\n[Tool 실행] 🔍 사내 DB에서 다음 키워드로 검색 중: '{query}'")
    docs = retriever.invoke(query)
    if not docs:
        return "관련된 사내 코드를 찾을 수 없습니다. 사용자에게 추가 정보를 요청하세요."
    
    context = "\n\n".join([f"[출처: {doc.metadata.get('source', '알수없음')}]\n{doc.page_content}" for doc in docs])
    return context

# LLM에 도구 바인딩 (LLM에게 '너 이 도구 쓸 수 있어'라고 알려줌)
tools = [search_inhouse_framework]
llm_with_tools = llm.bind_tools(tools)

# ==========================================
# 3. 노드(Node) 정의
# ==========================================
# 시스템 프롬프트: 에이전트의 역할과 도구 사용 규칙을 강력하게 정의합니다.
sys_msg = SystemMessage(content="""당신은 10년 차 사내 백엔드 자바 아키텍트입니다.
사용자가 코드를 요청하면, 먼저 'search_inhouse_framework' 도구를 사용하여 사내 코드를 검색하세요.
반드시 도구로 검색된 [사내 프레임워크 Context]만을 참고하여 질문에 답하고 코드를 작성해야 합니다.
없는 외부 라이브러리를 지어내지 말고, 참조한 클래스나 어노테이션에 대해 주석으로 친절히 설명해 주세요.
일상적인 대화(안녕, 날씨 등)에는 도구를 쓰지 말고 가볍게 대답하세요.""")

def agent_node(state: MessagesState):
    print("\n--- [Agent Node] LLM 생각 중... ---")
    # 대화 기록(messages)의 맨 앞에 시스템 프롬프트를 삽입하여 LLM에 전달
    messages = [sys_msg] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]} # 상태(State)의 messages 배열에 AI의 답변(또는 도구 호출 요청)을 추가

# ==========================================
# 4. 그래프(Graph) 구성 및 메모리 연결
# ==========================================
workflow = StateGraph(MessagesState)

# 노드 등록
workflow.add_node("agent", agent_node)
# LangGraph에서 제공하는 기본 ToolNode를 사용하면 도구 실행 결과를 자동으로 State에 합쳐줍니다.
workflow.add_node("tools", ToolNode(tools))

# 흐름(엣지) 연결
workflow.add_edge(START, "agent")

# 🚀 조건부 라우팅 (LLM이 도구를 쓰겠다고 판단하면 'tools'로, 아니면 대화 종료(END)로 자동 분기)
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent") # 도구 실행이 끝나면 다시 에이전트에게 돌아와서 최종 답변을 만들게 함 (Loop)

# 🚀 메모리 기능 추가 (심사위원 요구사항: 멀티턴 대화 기억)
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# ==========================================
# 5. 실행 테스트 (대화 세션 유지)
# ==========================================
if __name__ == "__main__":
    print("\n=======================================================")
    print("🤖 자율형 Multi-Agent(Tool Calling) 테스트를 시작합니다.")
    print("=======================================================\n")
    
    # 동일한 thread_id를 사용하면 이전 대화를 기억합니다.
    config = {"configurable": {"thread_id": "test_user_session_1"}}
    
    # 테스트 1: 일상 대화 (도구 미사용)
    query1 = "안녕! 넌 누구야?"
    print(f"User: {query1}")
    for event in app.stream({"messages": [("user", query1)]}, config=config, stream_mode="values"):
        last_msg = event["messages"][-1]
    print(f"\n[최종 답변] {last_msg.content}\n")
    print("-" * 50)
    
    # 테스트 2: 코드 작성 요청 (도구 자동 사용 및 순환 루프 확인)
    query2 = "결제 내역 조회 로직 짤 건데, DB 페이징이랑 현재 로그인한 사람 사번 가져오는 사내 표준 코드로 Service 클래스 짜줘."
    print(f"User: {query2}")
    for event in app.stream({"messages": [("user", query2)]}, config=config, stream_mode="values"):
        last_msg = event["messages"][-1]
        # 도구 호출(Tool Call)이 발생했는지 확인
        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            print(f"🤖 (LLM의 판단): 사내 코드 검색이 필요하군! -> 도구 호출 요청함.")
    print(f"\n[최종 답변]\n{last_msg.content}\n")