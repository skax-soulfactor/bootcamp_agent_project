import streamlit as st
import uuid
# 우리가 만든 LangGraph 에이전트와 DB 빌드 함수를 불러옵니다.
from agent_graph import app as agent_app 
from config import build_vector_db

# 페이지 기본 설정
st.set_page_config(page_title="DevX-Copilot", page_icon="🤖", layout="centered")

st.title("🛠️ DevX-Copilot")
st.caption("사내 Java 프레임워크 특화 자율형 AI 에이전트")

# ==========================================
# ⚙️ 사이드바 관리자 메뉴 (DB 업데이트)
# ==========================================
with st.sidebar:
    st.header("⚙️ 관리자 설정")
    st.info("사내 코드가 변경되었거나 처음 실행하는 경우 아래 버튼을 눌러 DB를 최신화하세요.")
    
    if st.button("🔄 Vector DB 초기화 및 생성", use_container_width=True):
        with st.spinner("사내 소스 코드를 읽어 DB를 구축하고 있습니다..."):
            try:
                chunk_count = build_vector_db()
                st.success(f"✅ DB 구축 완료! (총 {chunk_count}개의 코드 조각 저장)")
            except Exception as e:
                st.error(f"🚨 DB 구축 실패: {str(e)}")

# ==========================================
# 1. 세션 상태 및 Thread ID 초기화
# ==========================================
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
# 🚀 2. 채팅 입력 및 에이전트 실행 (무결점 에러 핸들링)
# ==========================================
if prompt := st.chat_input("예: 사용자 목록 페이징 로직 짜줘"):
    
    # 1) 입력 검증: 빈 텍스트 차단
    if not prompt.strip():
        st.warning("⚠️ 질문을 입력해 주세요.")
        st.stop()

    # 2) 사용자 메시지 화면 출력 및 세션 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3) AI 응답 처리
    with st.chat_message("assistant"):
        with st.spinner("에이전트들이 협업하여 사내 코드를 작성 중입니다..."):
            try:
                # 에이전트 호출
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                result = agent_app.invoke({"messages": [("user", prompt)]}, config=config)
                
                # 에러 없이 정상적으로 결과가 나왔을 때만 변수 할당
                answer = result["messages"][-1].content
                st.markdown(answer)
                
                # 정상 답변만 세션에 깔끔하게 추가
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                # 🎯 해결 포인트: 에러 발생 시 answer 변수 의존성을 완벽히 제거!
                error_msg = f"🚨 시스템 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.\n\n(상세 에러: `{str(e)}`)"
                st.error(error_msg)
                
                # 에러 메시지를 세션에 저장하여, 새로고침해도 에러 상황의 문맥이 유지되도록 함
                st.session_state.messages.append({"role": "assistant", "content": error_msg})