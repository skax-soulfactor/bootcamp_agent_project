import streamlit as st
# 우리가 만든 LangGraph 에이전트를 불러옵니다.
from agent_graph import app as agent_app 

# 페이지 기본 설정
st.set_page_config(page_title="DevX-Copilot", page_icon="🤖", layout="centered")

st.title("🛠️ DevX-Copilot")
st.caption("사내 Java 프레임워크(Mock) 특화 AI 개발 어시스턴트")

# 세션 상태(Session State)에 채팅 기록 저장
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 사내 프레임워크 적용이 막막하신가요? 구현하고자 하는 비즈니스 로직을 말씀해 주시면 사내 표준 코드로 작성해 드립니다."}
    ]

# 이전 채팅 기록들을 화면에 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력 창
if prompt := st.chat_input("예: 사용자 목록을 페이징해서 가져오는 로직 짜줘"):
    
    # 1. 사용자가 입력한 메시지를 화면에 출력 및 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 에이전트 실행 및 답변 출력
    with st.chat_message("assistant"):
        # 답변을 기다리는 동안 로딩 스피너 표시
        with st.spinner("로직 분석 및 사내 코드 검색 중..."):
            # LangGraph 에이전트에 질문 전달
            result = agent_app.invoke({"question": prompt})
            answer = result["generation"]
            
            # 결과 화면 출력
            st.markdown(answer)
            
    # 3. AI의 답변을 채팅 기록에 저장
    st.session_state.messages.append({"role": "assistant", "content": answer})