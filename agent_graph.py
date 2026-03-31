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

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate


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
# 3. 노드(Node) 정의 및 프롬프트 고도화
# ==========================================
from langchain_core.messages import SystemMessage

# 🚀 개선점: Few-shot 예시 및 출력 포맷(템플릿) 강제 적용
SYSTEM_PROMPT = """당신은 10년 차 사내 백엔드 자바 아키텍트입니다.
사용자가 코드를 요청하면, 먼저 'search_inhouse_framework' 도구를 사용하여 사내 코드를 검색하세요.
반드시 도구로 검색된 [사내 프레임워크 Context]만을 참고하여 질문에 답하고 코드를 작성해야 합니다.

[출력 형식 가이드라인]
반드시 아래의 4가지 섹션으로 나누어 마크다운 형식으로 답변하세요.
1. 📝 **요약**: 구현할 비즈니스 로직과 사용된 사내 핵심 클래스 요약
2. 💻 **코드**: 사내 표준을 준수한 Java 코드 스니펫
3. 🔗 **근거 출처**: 참조한 사내 프레임워크 문서/코드명
4. ⚠️ **주의사항**: 코드 적용 시 개발자가 주의해야 할 점 (예: 쿼리 ID 등록 등)

[Few-shot 예시: 좋은 답변과 나쁜 답변]
* 사용자 질문: "사용자 정보 수정 로직 짜줘"
* ❌ 나쁜 답변 (환각/외부 API 무단 사용): 
  "Spring의 @RestController와 JpaRepository를 사용하여 구현합니다. (코드 솰라솰라...)" -> 사내 규칙 위반!
* ✅ 좋은 답변 (사내 표준 준수):
  "📝 요약: 사내 CompanyDbTemplate의 update 메서드를 활용해 수정 로직을 구현합니다.
  💻 코드: ... (CompanyDbTemplate 사용 코드) ...
  🔗 근거 출처: mock_framework/db/CompanyDbTemplate.java
  ⚠️ 주의사항: 사내 DB 템플릿 사용 시 반드시 파라미터를 Map으로 전달해야 합니다."

리뷰어(Reviewer)의 피드백이 있다면, 피드백을 반영하여 코드를 다시 작성하세요.
일상적인 대화(안녕, 날씨 등)에는 도구를 쓰지 말고 가볍게 대답하세요.
"""

sys_msg = SystemMessage(content=SYSTEM_PROMPT)

def agent_node(state: MessagesState):
    print("\n--- [Agent Node] 코드 작성 및 도구 사용 판단 중... ---")
    messages = [sys_msg] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]} 

# (2) 🚀 Reviewer 노드 (코드 검증기 - 핵심 가산점 포인트)
class ReviewResult(BaseModel):
    is_valid: bool = Field(description="코드가 사내 규칙을 모두 준수했는지 여부")
    feedback: str = Field(description="위반 사항이 있다면 구체적인 수정 요청 피드백. 통과면 'PASS'")

def reviewer_node(state: MessagesState):
    print("\n--- [Reviewer Node] 사내 코딩 컨벤션 및 규칙 검증 중... ---")
    last_msg = state["messages"][-1].content
    
    # 코드가 포함되지 않은 단순 일상 대화는 검증 없이 통과
    if "class" not in last_msg and "public" not in last_msg:
        print("✅ [Reviewer] 일반 대화로 감지되어 검증을 패스합니다.")
        return {"messages": []} 
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 깐깐한 사내 코드 리뷰어입니다. 
        작성된 코드가 다음 사내 규칙을 준수했는지 엄격하게 평가하세요:
        1. 사내 DB 처리를 위해 CompanyDbTemplate을 사용했는가? (직접 Connection이나 Spring JdbcTemplate 사용 금지)
        2. 권한 처리를 위해 CompanySecurityContext 또는 @CompanyAuth를 사용했는가?
        3. 외부 프레임워크(Spring Boot 등)의 어노테이션을 무단으로 추가하지 않았는가?
        
        단 하나라도 위반했다면 is_valid=False로 하고 어떻게 고쳐야 할지 feedback을 작성하세요.
        완벽하게 준수했다면 is_valid=True로 하세요."""),
        ("user", "{code}")
    ])
    
    # Structured Output으로 엄격한 검증 결과 추출
    reviewer_llm = llm.with_structured_output(ReviewResult)
    result = (prompt | reviewer_llm).invoke({"code": last_msg})
    
    if result.is_valid:
        print("✅ [Reviewer] 검증 통과! 완벽한 사내 코드입니다. 사용자에게 전달합니다.")
        return {"messages": []} # 변경 없이 그대로 종료로 넘김
    else:
        print(f"❌ [Reviewer] 사내 규칙 위반 발견! 피드백: {result.feedback}")
        # 피드백을 HumanMessage로 감싸서 다시 Agent에게 던짐
        feedback_msg = HumanMessage(
            content=f"[코드 리뷰어 피드백 - 이 피드백을 반영해서 코드를 다시 작성하세요]\n{result.feedback}", 
            name="Reviewer"
        )
        return {"messages": [feedback_msg]}

# ==========================================
# 4. 그래프(Graph) 엣지 흐름 제어 (Custom Routing)
# ==========================================
# (1) Agent 실행 후 어디로 갈지 결정
def route_after_agent(state: MessagesState):
    last_message = state["messages"][-1]
    # 도구를 호출했다면 Tool 노드로 이동
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    # 도구 호출이 없고 코드를 생성했다면 무조건 Reviewer 노드로 이동하여 검사받음
    return "reviewer"

# (2) Reviewer 실행 후 어디로 갈지 결정
def route_after_reviewer(state: MessagesState):
    last_message = state["messages"][-1]
    # 방금 추가된 메시지가 리뷰어의 '피드백'이라면, 다시 Agent로 돌아가서 코드 수정 (Loop)
    if last_message.name == "Reviewer":
        return "agent"
    # 피드백이 추가되지 않았다면(검증 통과), 대화 종료
    return END

# 그래프 구성
workflow = StateGraph(MessagesState)

# 노드 등록
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("reviewer", reviewer_node)

# 흐름 연결 (복잡한 순환 루프 완성!)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", route_after_agent)
workflow.add_edge("tools", "agent") # 도구 검색이 끝나면 다시 에이전트로
workflow.add_conditional_edges("reviewer", route_after_reviewer) # 리뷰 결과에 따라 재수정(agent) 또는 끝(END)

# 메모리 연동 컴파일
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