import streamlit as st
import requests
import pandas as pd

# ==========================================
# 1. 페이지 설정 (레이아웃 및 제목)
# ==========================================
st.set_page_config(
    page_title="NASA 베어링 AI 진단 시스템",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 백엔드 API 주소 (main.py가 실행 중이어야 함)
API_URL = "http://127.0.0.1:8000/diagnose"

# ==========================================
# 2. 메인 타이틀 및 헤더
# ==========================================
st.title("🏭 NASA 회전기기 AI 예지보전 시스템")
st.markdown("""
**Physics-Informed AI (Data + Domain Knowledge)** 기반의 고장 진단 및 수명 예측 솔루션입니다.
좌측 패널에서 센서 데이터를 입력하면 **생성형 AI(Gemini)**가 정비 지시서를 작성합니다.
""")
st.markdown("---")

# ==========================================
# 3. 사이드바 (데이터 입력 패널)
# ==========================================
with st.sidebar:
    st.header("🎛️ 센서 데이터 조절")
    st.info("가상의 센서 값을 입력하여 AI 모델을 테스트합니다.")
    
    # 슬라이더 입력
    rms = st.slider("RMS (진동 가속도)", 0.0, 1.0, 0.25, help="평균적인 진동의 크기 (정상 < 0.2)")
    kurtosis = st.slider("Kurtosis (첨도)", 0.0, 10.0, 3.0, help="충격 신호의 뾰족한 정도 (베어링 손상 시 급증)")
    max_amp = st.slider("Max Amplitude (최대 진폭)", 0.0, 2.0, 0.6)
    std_dev = st.slider("Standard Deviation (표준편차)", 0.0, 1.0, 0.15)
    skewness = st.slider("Skewness (비대칭도)", -2.0, 2.0, 0.2)
    
    st.markdown("---")
    
    # 진단 실행 버튼 (Primary 컬러 적용)
    predict_btn = st.button("🔍 AI 진단 실행", type="primary", use_container_width=True)

# ==========================================
# 4. 메인 대시보드 (결과 표시 로직)
# ==========================================

# (A) 초기 안내 문구 (버튼 누르기 전)
if 'result' not in st.session_state:
    st.info("👈 왼쪽 사이드바에서 값을 설정하고 **[AI 진단 실행]** 버튼을 눌러주세요.")

# (B) 버튼 클릭 시 백엔드 API 호출
if predict_btn:
    payload = {
        "RMS": rms,
        "Std_Dev": std_dev,
        "Max_Amp": max_amp,
        "Kurtosis": kurtosis,
        "Skewness": skewness
    }
    
    try:
        with st.spinner('AI가 데이터를 분석하고 정비 지시서를 작성 중입니다...'):
            # 백엔드 호출
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                st.session_state['result'] = result # 결과 세션에 저장 (새로고침 방지)
            else:
                st.error(f"서버 오류 발생: {response.status_code}")
                
    except requests.exceptions.ConnectionError:
        st.error("⚠️ 백엔드 서버에 연결할 수 없습니다. 터미널에서 `uvicorn main:app --reload`를 실행했는지 확인해주세요.")

# (C) 결과 화면 렌더링 (저장된 결과가 있을 때만 표시)
if 'result' in st.session_state:
    res = st.session_state['result']
    status = res['status']
    rul = res['rul_hours']
    ai_report = res['ai_report']
    
    # 1. 핵심 지표 카드 (3단 컬럼)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="입력 진동 (RMS)", value=f"{rms:.3f} g")
    
    with col2:
        # 상태에 따른 색상 분기
        if "정상" in status:
            st.success(f"### 상태: {status}")
        elif "주의" in status:
            st.warning(f"### 상태: {status}")
        else:
            st.error(f"### 상태: {status}")
            
    # [수정된 dashboard.py 로직]
    with col3:
        # 상태 텍스트에 따라 메시지와 색상을 정확하게 분기
        if "정상" in status:
            delta_msg = "안전 범위"
            delta_color = "normal"  # 검정/초록
        elif "주의" in status:
            delta_msg = "예방 정비 권장" # '교체 시급' 대신 부드러운 표현
            delta_color = "off"     # 회색/검정 (또는 'inverse'로 강조 가능)
        else: # 위험
            delta_msg = "교체 시급 (Urgent)"
            delta_color = "inverse" # 빨간색
            
        st.metric(label="잔존 수명 (RUL)", value=f"{rul:.1f} hours", delta=delta_msg, delta_color=delta_color)

    st.markdown("---")

    # 2. [핵심] 생성형 AI 리포트 영역
    st.subheader("📝 AI 정비 작업 지시서 (Generative AI Report)")
    
    # 1. 상태에 따라 테두리 색상과 아이콘 다르게 하기
    if "정상" in status:
        box_type = "info"
        icon = "✅"
    elif "주의" in status:
        box_type = "warning"
        icon = "⚠️"
    else: # 위험
        box_type = "error"
        icon = "🚨"

    # 2. 컨테이너 안에 리포트 출력
    with st.container(border=True):
        # 상단에 상태 요약 배너 표시
        if box_type == "info":
            st.info(f"**[{icon} System Status]** 설비가 안정적입니다.")
        elif box_type == "warning":
            st.warning(f"**[{icon} System Status]** 예방 정비가 필요합니다.")
        else:
            st.error(f"**[{icon} System Status]** 긴급 조치가 필요합니다!")
            
        st.markdown("---") # 구분선
        
        # AI가 쓴 마크다운 리포트 출력
        st.markdown(ai_report)
        
        # 하단 서명 (디테일 추가)
        st.caption(f"Generated by NASA AI System • Model: Llama-3-70b • Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")