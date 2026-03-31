import streamlit as st
import uuid
# 우리가 만든 LangGraph 에이전트를 불러옵니다.
from agent_graph import app as agent_app 
from config import build_vector_db

# 페이지 기본 설정
st.set_page_config(page_title="DevX-Copilot", page_icon="🤖", layout="centered")

st.title("🛠️ DevX-Copilot")
st.caption("사내 Java 프레임워크 특화 자율형 AI 에이전트")

# ==========================================
# 🚀 [신규 기능] 사이드바 관리자 메뉴 (DB 업데이트)
# ==========================================
with st.sidebar:
    st.header("⚙️ 관리자 설정")
    st.info("사내 코드가 변경되었거나 처음 실행하는 경우 아래 버튼을 눌러 DB를 최신화하세요.")
    
    if st.button("🔄 Vector DB 초기화 및 생성", use_container_width=True):
        with st.spinner("사내 소스 코드를 읽어 DB를 구축하고 있습니다..."):
            try:
                # config.py에 만들어둔 함수 호출!
                chunk_count = build_vector_db()
                st.success(f"✅ DB 구축 완료! (총 {chunk_count}개의 코드 조각 저장)")
            except Exception as e:
                st.error(f"🚨 DB 구축 실패: {str(e)}")

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
# 🚀 2. 채팅 입력 및 에이전트 실행 (예외 처리 추가)
# ==========================================
if prompt := st.chat_input("예: 결제 내역 페이징해서 가져오는 로직 짜줘"):
    
    # 🎯 개선점: 사용자 입력 검증 (빈 문자열 차단)
    if not prompt.strip():
        st.warning("⚠️ 질문을 입력해 주세요.")
        st.stop()

    # 사용자가 입력한 메시지를 화면에 출력 및 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 에이전트 실행 및 답변 출력
    with st.chat_message("assistant"):
        with st.spinner("에이전트가 생각하고 검증하는 중입니다... (Tool 검색 & 코드 리뷰 진행)"):
            try:
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                
                # 🎯 개선점: LLM 및 그래프 실행 중 발생할 수 있는 오류를 try-except로 방어
                result = agent_app.invoke({"messages": [("user", prompt)]}, config=config)
                
                answer = result["messages"][-1].content
                st.markdown(answer)
                
                # AI의 답변을 채팅 기록에 저장
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                # 🎯 개선점: 오류 발생 시 앱이 죽지 않고 사용자에게 친절하게 안내
                error_msg = f"🚨 시스템 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.\n\n(상세 에러: {str(e)})"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
    # AI의 답변을 채팅 기록에 저장
    st.session_state.messages.append({"role": "assistant", "content": answer})