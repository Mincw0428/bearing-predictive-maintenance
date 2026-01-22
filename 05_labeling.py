# 1. 라이브러리 임포트
import pandas as pd # 데이터 조작을 위한 필수 도구

# 2. 데이터 불러오기
# 이전 단계에서 'Feature Extraction'을 통해 만든 통계 요약 데이터를 가져옵니다.
# RMS, Kurtosis 같은 데이터가 들어있지만, 아직 '정답(Label)'은 없는 상태입니다.
df = pd.read_csv('bearing_dataset_features.csv')

# 3. 라벨링(Labeling) 함수 정의
# 방금 전 단계에서 그래프(RMS Trend)를 눈으로 보고 결정한 '임계값(Threshold)'을 코드로 옮깁니다.
# 이 함수는 데이터의 순서(index)를 입력받아 그 시점의 베어링 상태를 숫자로 반환합니다.
def attach_label(index):
    # 인덱스 0 ~ 530: 그래프가 바닥에 평평하게 붙어있던 구간
    if index <= 530:
        return 0  # 0 = Normal (정상 상태): 아무 문제 없이 잘 돌아감
    
    # 인덱스 531 ~ 700: 그래프가 조금씩 꿈틀거리며 상승하던 구간
    elif index <= 700:
        return 1  # 1 = Warning (주의/경보 단계): 미세 균열 시작, 정밀 점검 필요
    
    # 인덱스 701 ~ 끝: 그래프가 급격히 치솟는 구간
    else:
        return 2  # 2 = Failure (위험/고장 단계): 당장 기계를 멈춰야 함 (진동 폭발)

# 4. 데이터프레임에 라벨 적용 (분류 모델용 정답지)
# map() 함수를 써서 데이터프레임의 모든 행(index)에 위에서 만든 attach_label 함수를 한 번에 적용합니다.
# 결과: 'Label'이라는 새로운 컬럼이 생기고 0, 1, 2가 채워집니다.
# 용도: Random Forest나 CNN 같은 분류 모델이 "지금 상태가 어떤가?"를 맞추는 데 사용합니다.
df['Label'] = df.index.map(attach_label)

# 5. RUL (Remaining Useful Life, 잔존 수명) 계산 (회귀 모델용 정답지)
# 예지보전(Predictive Maintenance)의 꽃입니다. "앞으로 며칠 더 쓸 수 있나요?"에 대한 대답입니다.

# total_life: 베어링이 완전히 고장 난 마지막 시점(인덱스)을 구합니다.
# df.index[-1]은 데이터프레임의 가장 마지막 행 번호(약 982)입니다.
total_life = df.index[-1] 

# RUL 공식: 전체 수명 - 현재 시간 = 남은 시간
# 예: 전체 수명이 980인데, 지금 100번째 시간이라면? RUL = 980 - 100 = 880 (880만큼 더 살 수 있음)
# 고장 시점(끝)에 가까워질수록 RUL은 0에 수렴합니다.
# 용도: XGBoost, LSTM 같은 회귀 모델이 "고장까지 남은 시간"을 예측하는 데 사용합니다.
df['RUL'] = total_life - df.index

# 6. 최종 학습용 데이터 저장
# 기계공학적 특징(RMS 등) + 정답지(Label, RUL)가 모두 합쳐진 완벽한 데이터셋이 완성되었습니다.
# index=False: 저장할 때 불필요한 행 번호가 또 생기지 않도록 설정합니다.
df.to_csv('bearing_dataset_final.csv', index=False)

# 7. 결과 확인 및 출력
print("✅ 라벨링 완료!")

# 데이터 분포 확인 (불균형 데이터인지 확인)
# 정상(0), 주의(1), 위험(2) 데이터가 각각 몇 개인지 세어봅니다.
# 보통 정상 데이터가 압도적으로 많고, 고장 데이터는 적은 '데이터 불균형' 문제가 발생하는데 이를 확인하는 과정입니다.
print(f"데이터 분포:\n{df['Label'].value_counts()}")

print("\n상위 5개 데이터 미리보기:")
# 컬럼에 Label과 RUL이 잘 추가되었는지 눈으로 확인합니다.
print(df.head())