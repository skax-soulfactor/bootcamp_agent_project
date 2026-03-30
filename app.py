import streamlit as st
import uuid
# 우리가 만든 LangGraph 에이전트를 불러옵니다.
from agent_graph import app as agent_app 

# 페이지 기본 설정
st.set_page_config(page_title="DevX-Copilot", page_icon="🤖", layout="centered")

st.title("🛠️ DevX-Copilot")
st.caption("사내 Java 프레임워크 특화 자율형 AI 에이전트")

# ==========================================
# 🚀 1. 세션 상태 및 Thread ID 초기화
# ==========================================
# 사용자마다 고유한 대화 기록(Memory)을 유지하기 위해 고유 ID 생성
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 사내 프레임워크 적용이 막막하신가요? 구현하고자 하는 비즈니스 로직을 말씀해 주시면 사내 표준 코드로 작성해 드립니다."}
    ]

# 이전 채팅 기록들을 화면에 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==========================================
# 🚀 2. 채팅 입력 및 에이전트 실행
# ==========================================
if prompt := st.chat_input("예: 결제 내역 페이징해서 가져오는 로직 짜줘"):
    
    # 사용자가 입력한 메시지를 화면에 출력 및 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 에이전트 실행 및 답변 출력
    with st.chat_message("assistant"):
        with st.spinner("에이전트가 생각하고 검증하는 중입니다... (Tool 검색 & 코드 리뷰 진행)"):
            
            # 🎯 해결 포인트 1: Checkpointer가 요구하는 thread_id 설정
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            # 🎯 해결 포인트 2: 입력 포맷을 {"question": ...} 에서 {"messages": ...} 로 변경
            result = agent_app.invoke({"messages": [("user", prompt)]}, config=config)
            
            # 결과에서 가장 마지막에 추가된 메시지(최종 답변) 추출
            answer = result["messages"][-1].content
            
            # 결과 화면 출력
            st.markdown(answer)
            
    # AI의 답변을 채팅 기록에 저장
    st.session_state.messages.append({"role": "assistant", "content": answer})