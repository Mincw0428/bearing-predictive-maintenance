# 1. 라이브러리 임포트
import pandas as pd
import joblib  # 학습된 모델을 파일로 저장하거나 불러올 때 사용하는 도구입니다. (현장 배포 필수템)
from sklearn.model_selection import train_test_split # 데이터를 수능 공부용(Train)과 모의고사용(Test)으로 나누는 함수
from sklearn.preprocessing import StandardScaler # 데이터의 단위를 통일시켜주는 스케일러 (SVM에선 필수!)
from sklearn.svm import SVC # Support Vector Classifier (분류를 담당하는 SVM 모델)
from sklearn.metrics import accuracy_score, classification_report # 채점표(정확도, 정밀도 등)를 출력하는 도구

# 2. 최종 데이터셋 로드
# 앞서 라벨링(0:정상, 1:주의, 2:위험)까지 마친 최종 CSV 파일을 불러옵니다.
df = pd.read_csv('bearing_dataset_final.csv')

# 3. 학습용 데이터(X)와 정답(y) 분리
# X (Features): 모델에게 보여줄 문제지 (RMS, 편차, 첨도 등 통계 수치)
features = ['RMS', 'Std_Dev', 'Max_Amp', 'Kurtosis', 'Skewness']
X = df[features]

# y (Label): 모델이 맞춰야 할 정답지 (0, 1, 2 상태 코드)
y = df['Label']

# 4. 데이터 나누기 (Train Set vs Test Set)
# 전체 데이터를 몽땅 학습시키면, 나중에 처음 보는 데이터를 잘 맞추는지 검증할 수 없습니다.
# 그래서 8:2 비율로 나눕니다. (80%는 공부용, 20%는 시험용)
# stratify=y [중요]: 데이터 불균형 해결을 위한 옵션입니다.
# 고장 데이터(2)는 희귀하므로, 랜덤으로 나누다가 시험지에 정상(0) 데이터만 들어갈 수 있습니다.
# 이를 방지하기 위해 정답(y) 비율을 유지하면서 공평하게 나눕니다.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. 데이터 스케일링 (SVM의 핵심 전처리)
# SVM은 데이터 사이의 '거리'를 계산해서 선을 긋는 알고리즘입니다.
# RMS는 0.1 단위인데 Max_Amp는 10 단위라면, 숫자가 큰 Max_Amp가 결과에 너무 큰 영향을 줍니다.
# StandardScaler는 모든 데이터를 평균 0, 표준편차 1인 분포로 강제 변환하여 '공평한 비교'를 가능하게 합니다.
scaler = StandardScaler()

# fit_transform(Train): 학습 데이터의 평균/분산을 계산(fit)하고 변환(transform)합니다.
X_train_scaled = scaler.fit_transform(X_train)

# transform(Test): 학습 데이터의 기준(평균/분산)을 그대로 가져와서 시험 데이터만 변환합니다.
# (절대 시험 데이터(Test)를 fit하면 안 됩니다! 이는 'Data Leakage'라는 부정행위입니다.)
X_test_scaled = scaler.transform(X_test)

# 6. SVM 모델 생성 및 학습
print("모델 학습을 시작합니다...")

# kernel='rbf': 데이터가 직선으로 안 나눠질 때 곡선으로 나누게 해주는 '방사 기저 함수' 커널입니다.
# C=1.0: 마진(여유폭)과 오류 허용 사이의 균형을 조절하는 파라미터입니다.
# random_state=42: 매번 실행할 때마다 결과가 달라지지 않도록 고정합니다.
model = SVC(kernel='rbf', C=1.0, random_state=42)

# .fit(): 드디어 학습 시작! (스케일링된 데이터와 정답을 주고 공부시킴)
model.fit(X_train_scaled, y_train)

# 7. 성능 평가 (채점 시간)
# .predict(): 공부를 마친 모델에게 시험지(Test Set)를 주고 정답을 맞춰보라고 시킵니다.
y_pred = model.predict(X_test_scaled)

# 실제 정답(y_test)과 모델의 답안(y_pred)을 비교해서 점수를 매깁니다.
acc = accuracy_score(y_test, y_pred)

print("-" * 30)
print(f"모델 정확도: {acc * 100:.2f}%") # 예: 98.50%
print("-" * 30)
# classification_report: 정확도뿐만 아니라,
# Precision(정밀도: 고장이라고 했는데 진짜 고장인가?),
# Recall(재현율: 실제 고장을 놓치지 않고 잡았는가?) 등을 상세히 보여줍니다.
print("상세 리포트:\n", classification_report(y_test, y_pred))

# 8. 모델 및 스케일러 저장 (Deployment 준비)
# 학습이 끝난 모델(model)과 스케일러(scaler)를 파일(.pkl)로 저장합니다.
# 나중에 공장 라즈베리파이 같은 엣지 디바이스에서는 이 파일만 불러와서(load) 바로 판별하면 됩니다.
joblib.dump(model, 'svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("✅ 모델 저장 완료: svm_model.pkl, scaler.pkl")