# 1. 라이브러리 임포트
import pandas as pd
import matplotlib.pyplot as plt

# 2. 데이터셋 로드
# 앞서 전처리 단계에서 만든 'bearing_dataset_features.csv' 파일을 불러옵니다.
# 이 파일에는 시간 순서대로 정렬된 베어링의 RMS, 첨도, 왜도 등의 통계값이 들어있습니다.
df = pd.read_csv('bearing_dataset_features.csv')

# 3. 시각화 설정 (도화지 준비)
# 가로 12인치, 세로 6인치의 넉넉한 크기로 그래프 창을 엽니다.
plt.figure(figsize=(12, 6))

# 4. RMS (에너지 크기) 그래프 그리기 [메인 데이터]
# plt.plot(x축, y축, ...)
# x축: df.index (파일 순서, 즉 시간의 흐름을 나타냄. 0부터 980여 개까지)
# y축: df['RMS'] (베어링의 진동 에너지 수치)
# 베어링 마모가 진행될수록 유격이 커져서 진동 에너지(RMS)가 급격히 상승하는 원리입니다.
plt.plot(df.index, df['RMS'], color='black', label='RMS (Vibration Energy)')

# 5. 구간 표시 (가이드라인 그리기)
# 이 부분은 분석가가 데이터를 보고 "아, 이쯤부터 이상하네?"라고 판단한 지점을 표시하는 것입니다.
# (NASA 2nd 데이터셋은 약 530번째 파일 쯤부터 미세한 변화가 시작되고, 후반부에 급격히 나빠집니다.)

# axvline: 수직선(Vertical Line)을 그립니다.
# x=530: 정상 상태가 끝나는 지점 (예시)
plt.axvline(x=530, color='green', linestyle='--', alpha=0.5, label='Healthy End')

# x=700: 위험 수위가 시작되는 지점 (예시)
# 이때부터는 진동이 눈에 띄게 커지는 구간입니다.
plt.axvline(x=700, color='orange', linestyle='--', alpha=0.5, label='Warning Start')

# 6. 그래프 꾸미기 (가독성 향상)
# 제목: 베어링 열화(Degradation) 트렌드
plt.title('Bearing Degradation Trend (RMS)', fontsize=16)

# X축 이름: 시간 (여기서는 파일의 인덱스 순서가 곧 시간입니다)
plt.xlabel('Time (File Index)', fontsize=12)

# Y축 이름: 진동 레벨 (RMS 값)
plt.ylabel('Vibration Level (RMS)', fontsize=12)

# 범례(Legend) 표시: 그래프 왼쪽 상단 등에 어떤 선이 무엇을 의미하는지 박스로 보여줍니다.
plt.legend()

# 격자(Grid) 표시: 값을 읽기 편하게 배경에 연한 모눈을 그립니다.
plt.grid(True, alpha=0.3)

# 7. 텍스트 주석 달기 (Annotation)
# 그래프 위에 글씨를 써서 보고서용으로 만들기 좋게 합니다.
# plt.text(x좌표, y좌표, '내용', ...)

# (200, 0.05) 위치에 '정상 상태'라고 표시
plt.text(200, 0.05, 'Normal State (Healthy)', color='green', fontsize=12, fontweight='bold')

# (800, 0.15) 위치에 '고장 상태'라고 표시
# 그래프 후반부에 RMS가 치솟는 구간을 강조합니다.
plt.text(800, 0.15, 'Failure State (Broken)', color='red', fontsize=12, fontweight='bold')

# 8. 그래프 출력
plt.show()