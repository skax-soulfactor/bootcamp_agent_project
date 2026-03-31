from typing import Annotated, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# 모듈화된 파일들에서 설정과 프롬프트를 불러옵니다.
from config import llm, retriever
from prompts import ROUTER_PROMPT, QA_PROMPT, RETRIEVER_PROMPT, GENERATOR_PROMPT, REVIEWER_PROMPT

# ==========================================
# 1. State 및 Tool 정의
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    intent: str

@tool
def search_inhouse_framework(query: str) -> str:
    """사내 Java 프레임워크의 소스코드와 사용법을 검색합니다. 코드 작성 요청 시 반드시 사용하세요."""
    print(f"\n[Tool 실행] 🔍 사내 DB 검색 중: '{query}'")
    docs = retriever.invoke(query)
    if not docs:
        return "관련된 사내 코드를 찾을 수 없습니다."
    return "\n\n".join([f"[출처: {doc.metadata.get('source', '알수없음')}]\n{doc.page_content}" for doc in docs])

tools = [search_inhouse_framework]
# 🚀 중요: 검색 도구는 이제 Retriever Agent에게만 쥐여줍니다!
llm_with_tools = llm.bind_tools(tools)

# ==========================================
# 2. 🚀 독립된 5개의 Agent Node 정의
# ==========================================
class RouteOutput(BaseModel):
    intent: Literal["code_request", "general"] = Field(description="분류 의도")

def router_node(state: AgentState):
    print("\n--- [Node 1: Router Agent] 의도 분석 중 ---")
    last_msg = state["messages"][-1].content
    result = (ChatPromptTemplate.from_messages([
        ("system", ROUTER_PROMPT), ("user", "{question}")
    ]) | llm.with_structured_output(RouteOutput)).invoke({"question": last_msg})
    return {"intent": result.intent}

def general_qa_node(state: AgentState):
    print("\n--- [Node 2: General QA Agent] 일반 대화 처리 중 ---")
    last_msg = state["messages"][-1].content
    response = (ChatPromptTemplate.from_messages([
        ("system", QA_PROMPT), ("user", "{question}")
    ]) | llm).invoke({"question": last_msg})
    return {"messages": [response]}

# 🚀 신규: 검색 및 도구 호출을 전담하는 Retriever Agent
def retriever_agent_node(state: AgentState):
    print("\n--- [Node 3: Retriever Agent] 사내 코드 검색 판단 중 ---")
    messages = [SystemMessage(content=RETRIEVER_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]} 

# 🚀 수정됨: 코드를 짜기만 하는 Generator Agent (도구 없이 프롬프트로만 작업)
def generator_agent_node(state: AgentState):
    print("\n--- [Node 4: Generator Agent] 사내 표준 코드 생성 중 ---")
    messages = [SystemMessage(content=GENERATOR_PROMPT)] + state["messages"]
    response = llm.invoke(messages) # Tool 바인딩 없는 순수 LLM 사용
    return {"messages": [response]} 

class ReviewResult(BaseModel):
    is_valid: bool
    feedback: str

def reviewer_node(state: AgentState):
    print("\n--- [Node 5: Reviewer Agent] 사내 코딩 컨벤션 검증 중 ---")
    last_msg = state["messages"][-1].content
    result = (ChatPromptTemplate.from_messages([
        ("system", REVIEWER_PROMPT), ("user", "{code}")
    ]) | llm.with_structured_output(ReviewResult)).invoke({"code": last_msg})
    
    if result.is_valid:
        print("✅ [Reviewer] 검증 통과!")
        return {"messages": []} 
    else:
        print(f"❌ [Reviewer] 규칙 위반 발견! 피드백 전달")
        return {"messages": [HumanMessage(content=f"[리뷰어 피드백]\n{result.feedback}", name="Reviewer")]}

# ==========================================
# 3. 그래프(Graph) 엣지 제어 및 라우팅 로직
# ==========================================
def route_after_router(state: AgentState):
    return "general_qa" if state["intent"] == "general" else "retriever" # 코딩 질문이면 검색 에이전트로!

def route_after_retriever(state: AgentState):
    last_message = state["messages"][-1]
    # Retriever가 도구를 쓰기로 결심했다면 ToolNode로, 검색이 끝났다면 Generator로 이동
    return "search_tools" if hasattr(last_message, 'tool_calls') and last_message.tool_calls else "generator"

def route_after_reviewer(state: AgentState):
    # 반려 시 Generator가 다시 코드를 짜도록 루프!
    return "generator" if state["messages"][-1].name == "Reviewer" else END

workflow = StateGraph(AgentState)

# 노드 등록 (총 6개: 5개의 에이전트 + 1개의 ToolNode)
workflow.add_node("router", router_node)
workflow.add_node("general_qa", general_qa_node)
workflow.add_node("retriever", retriever_agent_node)
workflow.add_node("search_tools", ToolNode(tools))
workflow.add_node("generator", generator_agent_node)
workflow.add_node("reviewer", reviewer_node)

# 워크플로우 엣지 연결 (아키텍처 완벽 일치)
workflow.add_edge(START, "router")
workflow.add_conditional_edges("router", route_after_router)
workflow.add_edge("general_qa", END)

workflow.add_conditional_edges("retriever", route_after_retriever)
workflow.add_edge("search_tools", "retriever") # 검색 끝내고 다시 Retriever에게 요약 지시

workflow.add_edge("generator", "reviewer")
workflow.add_conditional_edges("reviewer", route_after_reviewer)

app = workflow.compile(checkpointer=MemorySaver())