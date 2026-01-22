# 1. 라이브러리 임포트
import pandas as pd
import numpy as np
import joblib # 모델 저장용 (나중에 현장에 배포할 때 씁니다)
import matplotlib.pyplot as plt

# xgboost: 트리(Tree) 기반의 앙상블 학습 알고리즘으로, 정형 데이터(엑셀 같은 표 데이터) 분석에서
# 현존하는 가장 강력한 성능을 보여주는 알고리즘입니다. (Kaggle 대회 우승 단골)
import xgboost as xgb

from sklearn.model_selection import train_test_split
# 회귀(Regression) 문제이므로 '정확도(Accuracy)' 대신 '오차(Error)'를 계산하는 함수들을 가져옵니다.
from sklearn.metrics import mean_squared_error, r2_score

# 2. 데이터 로드
# 이전 단계에서 RUL(남은 수명) 계산까지 마친 최종 데이터셋을 불러옵니다.
df = pd.read_csv('bearing_dataset_final.csv')

# 3. 데이터 준비 (Feature Selection)
# X (Features): 기계공학적 통계 수치들 (입력값)
# 학습시킬 특징들을 선택합니다. 이 값들이 변하면 수명도 변한다는 가정을 합니다.
features = ['RMS', 'Std_Dev', 'Max_Amp', 'Kurtosis', 'Skewness']
X = df[features]

# y (Target): 예측해야 할 정답지 (RUL)
# 분류(Label)가 아니라, 연속된 숫자(남은 시간)인 RUL을 타겟으로 잡습니다.
# 예: 980 -> 979 -> ... -> 0 (고장)
y = df['RUL']

# 4. 데이터 분리 (Train vs Test)
# 학습용 80%, 성능검증용 20%로 나눕니다.
# 분류 문제와 달리 y값이 연속형 숫자이므로 stratify 옵션은 보통 사용하지 않습니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. XGBoost 회귀 모델 생성 및 학습
print("수명 예측(RUL) 모델 학습 중...")

# XGBRegressor: 'Regressor'는 숫자를 예측하는 모델을 뜻합니다. (분류는 Classifier)
# 하이퍼파라미터 설정 (모델의 성능을 조절하는 나사들):
# - n_estimators=100: 의사결정 나무(Decision Tree)를 100개 만들어서 투표를 시키겠다.
# - learning_rate=0.1: 학습 속도. 너무 크면 오차를 못 줄이고, 너무 작으면 학습이 오래 걸립니다.
# - max_depth=5: 나무의 깊이. 너무 깊으면 과적합(Overfitting)되어 암기식 공부가 됩니다.
rul_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

# .fit(): 데이터를 먹여서 학습을 시킵니다.
# (참고: XGBoost는 트리 기반이라 데이터 스케일링(StandardScaler)이 필수는 아니지만, 하면 더 좋을 때도 있습니다.)
rul_model.fit(X_train, y_train)

# 6. 성능 평가 (채점)
# .predict(): 테스트 문제지(X_test)를 주고 예측값(predictions)을 받아옵니다.
predictions = rul_model.predict(X_test)

# RMSE (Root Mean Squared Error): 평균 제곱근 오차
# "예측한 시간과 실제 남은 시간이 평균적으로 얼마나 차이가 나는가?"
# 예: 실제 50시간 남았는데 55시간이라고 예측했다면 오차는 5입니다. 값이 작을수록 좋습니다.
rmse = np.sqrt(mean_squared_error(y_test, predictions))

# R2 Score (결정 계수)
# 모델이 데이터를 얼마나 잘 설명하는지 보여주는 지표입니다.
# 1.0에 가까울수록 완벽하게 예측했다는 뜻이고, 0에 가까우면 찍은 것과 다름없다는 뜻입니다.
r2 = r2_score(y_test, predictions)

print("-" * 30)
print(f"오차 범위(RMSE): 약 {rmse:.2f} 포인트") # 예: 약 15.3 포인트 (±15단위 정도 틀림)
print(f"설명력(R2 Score): {r2:.2f} (1.0에 가까울수록 완벽)") # 예: 0.95 (매우 우수함)
print("-" * 30)

# 7. 결과 시각화 (Actual vs Predicted)
# 숫자로만 보면 감이 안 오므로 그래프로 그려봅니다.
plt.figure(figsize=(10, 5))

# 실제 값 (정답) 그리기: 점선으로 표시
plt.plot(y_test.values, label='Actual RUL', color='black', linestyle='--')

# AI 예측 값 그리기: 빨간 실선으로 표시
# 두 선이 겹쳐서 거의 하나처럼 보일수록 모델 성능이 좋은 것입니다.
plt.plot(predictions, label='Predicted RUL', color='red', alpha=0.7)

plt.title('RUL Prediction: Actual vs AI Predicted')
plt.xlabel('Test Data Samples') # 테스트 데이터 샘플 번호
plt.ylabel('Remaining Useful Life (RUL)') # 남은 수명
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 8. 모델 저장
# 고장 예측 모델을 파일로 저장합니다. 
# 나중에 이 파일과 이전에 만든 svm_model.pkl 두 개를 이용해 종합 진단 시스템을 구축할 수 있습니다.
joblib.dump(rul_model, 'xgboost_rul.pkl')
print("✅ 모델 저장 완료: xgboost_rul.pkl")