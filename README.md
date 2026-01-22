#  Hybrid Predictive Maintenance System
> 도메인 지식(SPC)과 생성형 AI(RAG)를 결합한 지능형 회전기기 예지보전 플랫폼

본 프로젝트는 NASA Bearing Dataset을 활용하여 설비의 고장을 진단하고, 잔존 수명(RUL)을 예측하며, LLM을 통해 실시간 정비 가이드를 제공하는 엔드 투 엔드(End-to-End) 시스템입니다.

##  Key Features
* Hybrid Diagnosis Logic : 통계적 관리 기법(SPC)과 AI(SVM/XGBoost)를 결합하여 오진율 최소화
* Real-time Simulation : 사용자가 직접 특징량(RMS, Kurtosis)을 조절하며 결함 시나리오를 테스트하는 인터랙티브 환경
* RAG-based Maintenance Guide: Pinecone 벡터 DB와 Llama-3를 연동하여 실제 매뉴얼 기반의 정비 지침 생성
* Signal Processing : FFT(Fast Fourier Transform)를 통한 고장 주파수 식별 및 특징 추출

##  Tech Stack
* Language : Python
* Backend : FastAPI
* Frontend : Streamlit
* AI/ML : SVM (Diagnosis), XGBoost (RUL Regression)
* GenAI : Google Gemini (Embedding), Groq Llama-3 (LLM), Pinecone (Vector DB)

##  Performance
* Diagnosis Accuracy (SVM) : **92.89%**
* RUL Prediction ($R^2$) : **0.80**
* Reliability : 정상 구간 RUL 고정(984h) 및 물리적 임계값(Kurtosis > 5.0) 적용

##  How to Run
1. 의존성 설치
   ```bash
   pip install -r requirements.txt

2. 환경 변수 설정 (.env)
GROQ_API_KEY=your_api_key
PINECONE_API_KEY=your_api_key

3. 애플리케이션 실행
Bash
# Backend
uvicorn main:app --reload
# Frontend
streamlit run dashboard.py