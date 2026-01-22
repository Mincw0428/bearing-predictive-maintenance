# main.py
# 필요한 라이브러리 임포트
from fastapi import FastAPI               # 웹 서버 프레임워크
from pydantic import BaseModel            # 데이터 구조 정의 및 유효성 검사
import joblib                             # 학습된 머신러닝 모델 로드
from groq import Groq                     # Groq(Llama-3) API 클라이언트
import numpy as np                        # 수치 연산
from rag_system import query_manual       # (직접 만든) RAG 매뉴얼 검색 모듈

# ==========================================
# 🔑 API 키 및 클라이언트 설정
# ==========================================
# Groq Console에서 발급받은 키를 입력하세요.
GROQ_API_KEY = "GROQ_API_KEY" 

try:
    client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    print(f"⚠️ Groq 클라이언트 설정 오류: {e}")
    client = None

# ==========================================
# 1. FastAPI 앱 초기화
# ==========================================
app = FastAPI(
    title="NASA Bearing AI System (SPC Hybrid)",
    description="통계적 공정 관리(SPC) + SVM + XGBoost 하이브리드 진단 시스템",
    version="4.5.0" # Final Version
)

# ==========================================
# 2. AI 모델 로드 (서버 시작 시 1회 실행)
# ==========================================
models = {} 

try:
    # 1) 스케일러: 데이터 정규화용
    models['scaler'] = joblib.load('scaler.pkl')
    # 2) SVM: 패턴 분석 및 결함 유형 분류
    models['svm'] = joblib.load('svm_model.pkl')
    # 3) XGBoost: 잔존 수명(RUL) 회귀 예측
    models['rul'] = joblib.load('xgboost_rul.pkl')
    print("✅ 모든 ML 모델 로드 성공!")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    models['svm'] = None 

# ==========================================
# 3. 입력 데이터 구조 정의
# ==========================================
class VibrationData(BaseModel):
    RMS: float          # 진동의 에너지 (거시적 지표)
    Std_Dev: float      # 표준편차 (변동성)
    Max_Amp: float      # 최대 진폭
    Kurtosis: float     # 첨도 (충격성, 초기 결함 핵심 지표)
    Skewness: float     # 비대칭도 (파형 왜곡)

# ==========================================
# 4. [핵심 알고리즘] 통계 기반 하이브리드 진단
# ==========================================
def hybrid_diagnosis(data, svm_pred, xgb_rul):
    """
    [설계 논리: Statistical Process Control (SPC)]
    ISO 10816(속도) 규격과 본 데이터(가속도)의 단위 불일치 문제를 해결하기 위해,
    NASA 데이터셋 자체의 '정상 구간 분포'를 분석하여 통계적 임계값을 수립함.
    
    - Baseline (정상 평균): ~0.075g
    - Warning (3-Sigma, 약 2.5배): 0.18g (통계적 유의수준 벗어남)
    - Failure (6-Sigma, 약 6.0배): 0.45g (확실한 물리적 파손)
    """
    
    # 1. 통계적 임계값 (Data-Driven Thresholds)
    TH_STAT_WARNING = 0.18  # 주의 단계 진입점
    TH_STAT_FAILURE = 0.45  # 위험 단계 진입점
    TH_KURT_CRITICAL = 5.0  # 첨도(충격) 절대 임계값 (Crack 발생 징후)

    # ---------------------------------------------------------
    # Step 1: 통계적 기준에 따른 1차 상태 분류 (1st Filter)
    # ---------------------------------------------------------
    if data.RMS < TH_STAT_WARNING:
        stat_status = 0 # 정상 (Normal)
    elif data.RMS < TH_STAT_FAILURE:
        stat_status = 1 # 주의 (Warning) - Case 3, 4 커버
    else:
        stat_status = 2 # 위험 (Failure)

    # ---------------------------------------------------------
    # Step 2: AI (SVM) & 충격 신호(Kurtosis) 융합 (2nd Precision)
    # ---------------------------------------------------------
    final_status = stat_status # 기본적으로 통계적 기준을 따름

    # [예외 1] 진동(RMS)은 작지만 '충격(Kurtosis)'이 매우 큼 -> 초기 결함(Crack)
    if data.Kurtosis > TH_KURT_CRITICAL:
        final_status = 2 # 위험으로 격상
        print(f"⚖️ 정밀 보정: RMS({data.RMS})는 낮으나 첨도 과다({data.Kurtosis}) -> '위험'")

    # [예외 2] 통계적으로 '주의' 구간인데, SVM이 '위험'이라고 과민반응 함
    # -> 아직 RMS가 파괴 임계값(0.45)에 도달하지 않았으므로 '주의' 유지
    elif stat_status == 1 and svm_pred == 2:
        final_status = 1 
        print(f"⚖️ 정밀 보정: 진동량(RMS)이 파괴 수준 아님 -> SVM 판단 기각, '주의' 유지")

    # [예외 3] 통계적으로 '위험' 구간(0.45g 이상) -> SVM이 뭐라든 무조건 위험
    # -> 진동이 이렇게 크면 베어링이 멀쩡해도 주변 설비가 망가짐
    elif stat_status == 2 and svm_pred == 0:
        final_status = 2
        print(f"⚖️ 정밀 보정: 통계적 임계치 초과 -> 무조건 '위험'")

    # ---------------------------------------------------------
    # Step 3: XGBoost RUL 동기화 (Prediction Mapping)
    # 상태 판단 결과(Classification)가 수명 예측(Regression)의 범위를 제약함
    # [수정된 main.py RUL 로직]
    # 학습 데이터셋(NASA Bearing 1)의 Max Life가 984시간임을 반영
    
    # ---------------------------------------------------------
    # Step 3: XGBoost RUL 동기화 (Dataset Max Life 반영)
    # ---------------------------------------------------------
    final_rul = float(xgb_rul)
    
    # NASA 데이터셋의 시작점(Max RUL)은 약 984시간입니다.
    DATASET_MAX_RUL = 984.0 
    
    if final_status == 0: # 정상
        # [수정] 1200시간(가상의 값) 대신, 데이터셋의 실제 최댓값(984)을 기준으로 함.
        # 의미: "이 베어링은 실험 시작 시점(가장 건강한 상태)만큼 건강하다."
        
        # 모델 예측값이 984보다 작더라도, 상태가 '정상'이면 984로 보정하여
        # "건강한 상태임"을 보장함. (984 위로 튀는 건 허용)
        final_rul = max(final_rul, DATASET_MAX_RUL)
        
    elif final_status == 1: # 주의
        # 주의 단계: 48시간 ~ 500시간 사이에서 변동
        # (주의 단계는 데이터셋 중간 지점이므로 모델 예측값을 최대한 존중)
        final_rul = max(48.0, min(final_rul, 500.0))
        
    elif final_status == 2: # 위험
        # 위험 단계: 48시간 미만
        # (진동/충격이 클수록 수명 감소 로직 유지)
        
        rms_ratio = max(1.0, data.RMS / TH_STAT_FAILURE)
        kurt_ratio = max(1.0, data.Kurtosis / TH_KURT_CRITICAL)
        decay_factor = max(rms_ratio, kurt_ratio)
        
        natural_limit = 48.0 / decay_factor
        final_rul = min(final_rul, natural_limit)

    return final_status, final_rul

# ==========================================
# 5. Groq 기반 리포트 생성 함수
# ==========================================
def generate_ai_report(status_text, rul, data):
    # RAG 검색 (매뉴얼 찾기)
    try:
        search_query = f"상태: {status_text}, RMS: {data.RMS}, Kurtosis: {data.Kurtosis}"
        found_manuals = query_manual(search_query)
        manual_context = "\n".join(found_manuals)
    except:
        manual_context = "관련 매뉴얼 없음. 일반 베어링 정비 지침을 따르세요."

    # 프롬프트 작성 (한자 금지령 포함)
    prompt = f"""
    당신은 설비 보전 분야의 전문가입니다. 
    아래 데이터를 분석하여 현장 작업자가 즉시 이해할 수 있는 '정비 작업 지시서'를 작성하세요.
    
    [상황 데이터]
    - 진단 결과: {status_text} (잔존 수명 {rul:.1f}시간)
    - 핵심 센서: RMS {data.RMS:.3f}, Kurtosis {data.Kurtosis:.3f}
    
    [참고 매뉴얼]
    {manual_context}
    
    [작성 시 절대 규칙 - 중요!]
    1. **모든 내용은 반드시 '순수 한글'로만 작성하세요.** (한자 사용 절대 금지)
    2. 예: '生産' -> '생산', '振動' -> '진동', '可能性' -> '가능성'
    3. Markdown 문법을 사용하여 가독성을 높이세요.
    
    [양식]
    ### 🚨 1. 진단 요약
    - 현재 상태: **{status_text}**
    - 잔존 수명: 약 **{rul:.1f}시간**으로 예측됩니다.

    ### 🔍 2. 원인 분석
    - **RMS({data.RMS:.3f})**: (분석 내용 작성)
    - **Kurtosis({data.Kurtosis:.3f})**: (분석 내용 작성)
    - 종합 소견: (분석 결론)

    ### 🛠️ 3. 조치 권고
    - **즉시 조치**: (구체적 행동 지시)
    - **교체 부품**: (부품명)
    - **작업 우선순위**: (긴급/보통)
    """
    
    try:
        # Groq 모델 호출 (최신 Llama-3 사용)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile", # or llama-3.1-70b-versatile
            messages=[
                {"role": "system", "content": "You are a helpful industrial expert. Speak Korean only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4, # 사실적 답변을 위해 낮춤
            max_tokens=1024
        )
        return completion.choices[0].message.content
        
    except Exception as e:
        return f"❌ AI 리포트 생성 실패: {str(e)}"

# ==========================================
# 6. API 엔드포인트 (진단 실행)
# ==========================================
@app.post("/diagnose")
async def diagnose_bearing(data: VibrationData):
    # 모델 로드 확인
    if models['svm'] is None:
        return {"error": "Server Error: AI Models not loaded."}

    # (1) 데이터 전처리 & 스케일링
    features = [[data.RMS, data.Std_Dev, data.Max_Amp, data.Kurtosis, data.Skewness]]
    features_scaled = models['scaler'].transform(features)
    
    # (2) 모델 Raw 예측 (AI의 순수 의견)
    svm_raw = models['svm'].predict(features_scaled)[0] # 0, 1, 2
    xgb_raw = models['rul'].predict(features)[0]        # 예측 시간
    
    # (3) [핵심] 하이브리드 로직 실행 (통계 + AI + RUL 동기화)
    final_status_code, final_rul = hybrid_diagnosis(data, svm_raw, xgb_raw)

    # (4) 결과 텍스트 변환
    status_map = {0: "정상 (Normal)", 1: "주의 (Warning)", 2: "위험 (Failure)"}
    status_text = status_map[final_status_code]

    # (5) 리포트 생성 (정상이 아닐 경우에만)
    ai_message = "✅ 설비 상태가 양호합니다. 현재 가동 조건을 유지하십시오."
    
    if final_status_code > 0: # 주의 또는 위험
        print(f"🤖 Groq 리포트 생성 요청... (Status: {status_text})")
        ai_message = generate_ai_report(status_text, final_rul, data)

    # (6) 최종 결과 반환
    return {
        "status": status_text,
        "rul_hours": final_rul,
        "ai_report": ai_message
    }

# 실행 명령어: uvicorn main:app --reload