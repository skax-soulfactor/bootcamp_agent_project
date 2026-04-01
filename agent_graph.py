from typing import Annotated, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
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
    context_sufficient: bool  # 🚀 신규: Generator가 컨텍스트 충분 여부를 스스로 판단해 저장

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

class QADecision(BaseModel):
    topic: Literal["greeting", "emotion", "general_knowledge", "out_of_bounds"] = Field(description="질문의 주제를 스스로 판단하여 분류")
    response: str = Field(description="판단된 주제에 맞는 맞춤형 답변")

def general_qa_node(state: AgentState):
    print("\n--- [Node 2: General QA Agent] 질문 의도 심층 분석 및 답변 판단 중 ---")
    last_msg = state["messages"][-1].content
    
    # 스스로 주제를 분류(의사결정)한 후 답변을 생성합니다.
    result = (ChatPromptTemplate.from_messages([
        ("system", QA_PROMPT), ("user", "{question}")
    ]) | llm.with_structured_output(QADecision)).invoke({"question": last_msg})
    
    print(f"💡 [QA Agent 의사결정]: 이 질문은 '{result.topic}' 카테고리로 분류되었습니다. 이에 맞춰 대답합니다.")
    return {"messages": [AIMessage(content=result.response)]}

# 🚀 신규: 검색 및 도구 호출을 전담하는 Retriever Agent
def retriever_agent_node(state: AgentState):
    print("\n--- [Node 3: Retriever Agent] 사내 코드 검색 판단 중 ---")
    messages = [SystemMessage(content=RETRIEVER_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]} 

class GeneratorDecision(BaseModel):
    analysis: str = Field(description="Retriever가 찾아온 컨텍스트가 코드를 작성하기에 충분한지 단계별로 평가")
    is_enough: bool = Field(description="충분하면 True, 정보가 부족해 재검색이 필요하면 False")
    code_or_feedback: str = Field(description="True일 경우 완성된 코드. False일 경우 Retriever에게 지시할 구체적인 추가 검색 요청 메시지")

def generator_agent_node(state: AgentState):
    print("\n--- [Node 4: Generator Agent] 검색 데이터 평가 및 코드 생성 판단 중 ---")
    messages = [SystemMessage(content=GENERATOR_PROMPT)] + state["messages"]
    
    # 단순 생성이 아니라, 제공받은 정보를 스스로 평가(의사결정)합니다.
    result = llm.with_structured_output(GeneratorDecision).invoke(messages)
    
    if result.is_enough:
        print("💡 [Generator 의사결정]: 컨텍스트가 충분합니다. 코드를 생성합니다.")
    else:
        print("⚠️ [Generator 의사결정]: 정보가 부족합니다! Retriever에게 추가 검색을 지시합니다.")
        
    return {
        # name="Generator"를 달아서 이 메시지가 Generator의 추가 지시임을 명시합니다.
        "messages": [AIMessage(content=result.code_or_feedback, name="Generator")],
        "context_sufficient": result.is_enough
    }

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

def route_after_generator(state: AgentState):
    # Generator가 스스로 판단해서 정보가 부족하다고 하면, Reviewer로 가지 않고 Retriever로 반송시킵니다!
    if not state.get("context_sufficient", True):
        return "retriever"
    return "reviewer"

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

workflow.add_conditional_edges("generator", route_after_generator)
workflow.add_conditional_edges("reviewer", route_after_reviewer)

app = workflow.compile(checkpointer=MemorySaver())