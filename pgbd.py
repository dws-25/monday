import streamlit as st
from datetime import datetime, time

# --- 페이지 설정 (대충) ---
st.set_page_config(
    page_title="연휴 뭐함?", # 확 줄임
    page_icon="🤙",       # 좀 더 캐주얼하게
    layout="wide"
)

# --- 제목 (핵심만) ---
st.title("🔥 5월 연휴 ㄱㄱ? 🔥")
st.header("야, 뭐 할래? 골라봐.")
st.markdown("---")
st.markdown("""
5월 연휴 낀 거 알지? 놀아야지.
두 개 중에 골라. **오늘 저녁 6시까지.** 답 없으면 걍 없는거임.
""")

# --- 마감 시간 ---
deadline_time = time(18, 0) # 오후 6시
today_deadline = datetime.combine(datetime.today(), deadline_time)
current_datetime = datetime.now()

if current_datetime < today_deadline:
    time_left = today_deadline - current_datetime
    time_left_str = str(time_left).split('.')[0]
    st.success(f"**투표 마감:** 오늘 6시까지! (남은 시간: **{time_left_str}**)")
else:
    st.error("**🚫 투표 끝남 🚫**")

st.markdown("---")

# --- 계획 비교 ---
col1, col2 = st.columns(2, gap="large")

# --- 계획 A: 표네 집 ---
with col1:
    st.header("Plan A: 표네 집 쳐들어가기 🤙")
    # *** 네가 준 사진 파일 사용 (같은 폴더에 plan_a_pic.jpg 로 저장!) ***
    st.image("plan_a_pic.jpg", # 로컬 파일 이름
             caption="우리 이렇게 노는거? ㅋㅋ", # 캡션 수정
             use_container_width=True)
    st.subheader("🗓️ 5월 2일 (금) 저녁부터")
    st.markdown("""
    *   **🏠 표네 집 ㄱㄱ:** 저녁때쯤 알아서 모여.
    *   **1) 운동?:** 땀 흘릴 놈만? (아님 걍 먹어?)
    *   **2) 🍕 저녁:** 배달 ㄱㄱ (메뉴 추천 좀) + 노가리.
    *   **3) 🎮 게임:** 겜 ㄱㄱ? PC방 ㄱㄱ? (표 버스 ㄱ?)
    *   **4) 😴 잠:** 자든가 밤새든가.
    """)
    st.info("""
    **✨ 장점 ✨**
    *   걍 편함
    *   싸게 먹힘 (N빵)
    *   표가 반겨줌 (?)
    *   날씨 X까
    """)

# --- 계획 B: 지리산 ---
with col2:
    st.header("Plan B: 지리산 조지기 ⛰️")
    st.image("https://images.pexels.com/photos/167699/pexels-photo-167699.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1",
             caption="가서 이거 하는거임 (feat. 알배김)",
             use_container_width=True)
    st.subheader("🗓️ 5월 2일 (금) 밤 ~ 3일 (토)")
    st.markdown("""
    *   **🌙 동서울 ㄱㄱ:** 밤 11시 반 집합.
    *   **1) 🚌 버스 취침:** 자면서 감 (목베개 챙겨).
    *   **2) 🥘 아침:** 국밥 ㄱㄱ.
    *   **3) 🚶‍♂️ 등산:** 정상 찍고 인증샷 ㄱ.
    *   **4) 🚍 복귀:** 서울로 튐.
    *   **5) 💻 PC방 (선택):** 체력 남으면? (될까?)
    """)
    st.warning("""
    **🔥 장점 🔥**
    *   특별함
    *   성취감 (쩔듯?)
    *   자연 힐링?
    *   인증샷 ㄱㄴ
    """)

st.markdown("---")

# --- 투표 ---
st.header("✨ 그래서 뭐 할건데? ✨")

if current_datetime < today_deadline:
    vote = st.radio(
        "**골라.** 🤔",
        ("Plan A: 표네 집 ㄱㄱ 🤙", "Plan B: 지리산 ㄱㄱ ⛰️", '아직 모름 / 딴거?'),
        index=None,
        key="plan_vote",
    )

    opinion = st.text_area(
        "**할 말 있으면 적든가.** ✍️",
        placeholder="예) A 치킨 ㄱ / B 코스 쉬운걸로 / 아 몰라",
        key="user_opinion",
        height=100, # 좀 줄임
    )

    submit_button = st.button("🗳️ 투표하기") # 버튼 이름 짧게

    if submit_button:
        if 'plan_vote' in st.session_state and st.session_state.plan_vote:
            st.success(f"**ㅇㅋ '{st.session_state.plan_vote}' 투표함.**") # 메시지 짧게
            st.balloons()

            st.markdown("---")
            st.write("**니 선택:**") # 간결하게
            st.write(f"- 계획: {st.session_state.plan_vote}")
            if st.session_state.user_opinion:
                st.write(f"- 의견: {st.session_state.user_opinion}")
            # else:
            #     st.write("- 의견: 없음") # 없으면 굳이 표시 안 함
            st.info("결과는 6시 넘어서.") # 짧게

        else:
            st.warning("⚠️ **야, 뭐라도 골라.**") # 경고 메시지

else:
    st.info("🙏 투표 끝났다고.") # 짧게

st.markdown("---")
# st.caption("이 투표는 우리의 즐거운 연휴를 위해 Streamlit으로 만들어졌습니다. 😎") # <<< 이 줄 삭제됨