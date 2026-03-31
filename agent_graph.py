import os
from typing import Annotated, Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ==========================================
# 1. State 정의 (Router 의도 파악용 변수 추가)
# ==========================================
# 기존 MessagesState에 Router가 판단한 'intent'를 저장할 수 있도록 확장합니다.
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    intent: str

# ==========================================
# 2. LLM, DB, Tool 세팅
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
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5})

@tool
def search_inhouse_framework(query: str) -> str:
    """사내 Java 프레임워크의 소스코드와 사용법을 검색합니다. 코드 작성 요청 시 반드시 사용하세요."""
    print(f"\n[Tool 실행] 🔍 사내 DB 검색 중: '{query}'")
    docs = retriever.invoke(query)
    if not docs:
        return "관련된 사내 코드를 찾을 수 없습니다."
    return "\n\n".join([f"[출처: {doc.metadata.get('source', '알수없음')}]\n{doc.page_content}" for doc in docs])

tools = [search_inhouse_framework]
llm_with_tools = llm.bind_tools(tools)

# ==========================================
# 3. 🚀 독립된 에이전트 노드(Node) 정의
# ==========================================

# (1) Router Node: 의도만 파악하고 다음 길을 안내합니다.
class RouteOutput(BaseModel):
    intent: Literal["code_request", "general"] = Field(description="코드/개발 관련이면 'code_request', 단순 인사면 'general'")

def router_node(state: AgentState):
    print("\n--- [Node 1: Router] 의도 분석 중 ---")
    last_msg = state["messages"][-1].content
    structured_llm = llm.with_structured_output(RouteOutput)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "질문이 개발, 사내 프레임워크, 코드 작성과 관련되면 'code_request', 단순 인사나 일반 대화면 'general'을 반환하세요."),
        ("user", "{question}")
    ])
    result = (prompt | structured_llm).invoke({"question": last_msg})
    return {"intent": result.intent}

# (2) General QA Node: 개발과 무관한 일상 대화를 전담합니다.
def general_qa_node(state: AgentState):
    print("\n--- [Node 2: General QA] 일반 대화 처리 중 ---")
    last_msg = state["messages"][-1].content
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 친절한 AI 어시스턴트입니다. 코드나 개발 지식이 필요 없는 일상적인 질문에 짧고 친절하게 답해주세요."),
        ("user", "{question}")
    ])
    response = (prompt | llm).invoke({"question": last_msg})
    return {"messages": [response]}

# (3) Agent Node (코드 생성기): 도구를 활용해 사내 코드를 작성합니다.
SYSTEM_PROMPT = """당신은 10년 차 사내 백엔드 자바 아키텍트입니다.
사용자가 코드를 요청하면, 'search_inhouse_framework' 도구를 사용하여 사내 코드를 검색하세요.
검색된 [사내 프레임워크 Context]만을 참고하여 질문에 답하고 코드를 작성해야 합니다.

[출력 형식 가이드라인]
반드시 아래의 4가지 섹션으로 나누어 마크다운 형식으로 답변하세요.
1. 📝 **요약**: 구현할 비즈니스 로직과 사용된 사내 핵심 클래스 요약
2. 💻 **코드**: 사내 표준을 준수한 Java 코드 스니펫
3. 🔗 **근거 출처**: 참조한 사내 프레임워크 문서/코드명
4. ⚠️ **주의사항**: 코드 적용 시 개발자가 주의해야 할 점

[Few-shot 예시]
* 나쁜 답변: "Spring의 @RestController를 사용하여 구현합니다." (사내 규칙 위반)
* 좋은 답변: "📝 요약: CompanyDbTemplate 활용... 💻 코드: ... 🔗 출처: mock_framework... ⚠️ 주의: ..."

리뷰어의 피드백이 있다면, 피드백을 반영하여 코드를 다시 작성하세요."""

def agent_node(state: AgentState):
    print("\n--- [Node 3: Code Generator] 코드 작성 및 도구 사용 판단 중 ---")
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]} 

# (4) Reviewer Node: 코딩 컨벤션을 깐깐하게 검증합니다.
class ReviewResult(BaseModel):
    is_valid: bool = Field(description="규칙 준수 여부")
    feedback: str = Field(description="위반 시 수정 요청 피드백, 통과면 'PASS'")

def reviewer_node(state: AgentState):
    print("\n--- [Node 4: Reviewer] 사내 코딩 컨벤션 검증 중 ---")
    last_msg = state["messages"][-1].content
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 깐깐한 사내 코드 리뷰어입니다. 
        작성된 코드가 다음 사내 규칙을 준수했는지 엄격하게 평가하세요:
        1. 사내 DB 처리를 위해 CompanyDbTemplate을 사용했는가?
        2. 권한 처리를 위해 CompanySecurityContext 또는 @CompanyAuth를 사용했는가?
        3. 외부 프레임워크(Spring Boot 등)의 어노테이션을 무단으로 추가하지 않았는가?
        단 하나라도 위반했다면 is_valid=False로 하고 구체적인 feedback을 작성하세요. 통과 시 is_valid=True."""),
        ("user", "{code}")
    ])
    
    result = (prompt | llm.with_structured_output(ReviewResult)).invoke({"code": last_msg})
    
    if result.is_valid:
        print("✅ [Reviewer] 검증 통과! 완벽한 사내 코드입니다.")
        return {"messages": []} 
    else:
        print(f"❌ [Reviewer] 사내 규칙 위반 발견! 피드백: {result.feedback}")
        return {"messages": [HumanMessage(content=f"[리뷰어 피드백 - 코드를 다시 작성하세요]\n{result.feedback}", name="Reviewer")]}

# ==========================================
# 4. 그래프(Graph) 엣지 제어 및 라우팅 로직
# ==========================================
def route_after_router(state: AgentState):
    if state["intent"] == "general":
        return "general_qa"
    return "agent"

def route_after_agent(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return "reviewer"

def route_after_reviewer(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.name == "Reviewer":
        return "agent"
    return END

# ==========================================
# 5. 그래프 컴파일
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("general_qa", general_qa_node)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("reviewer", reviewer_node)

workflow.add_edge(START, "router")
# 라우터의 결과에 따라 일반 대화(general_qa)로 갈지, 코드 생성(agent)으로 갈지 분기
workflow.add_conditional_edges("router", route_after_router)

workflow.add_edge("general_qa", END)
workflow.add_conditional_edges("agent", route_after_agent)
workflow.add_edge("tools", "agent")
workflow.add_conditional_edges("reviewer", route_after_reviewer)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)