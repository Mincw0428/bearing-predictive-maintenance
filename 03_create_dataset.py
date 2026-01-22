# 1. 라이브러리 임포트
import pandas as pd
import numpy as np
import os
# scipy.stats: 통계적 특성을 구하기 위한 라이브러리입니다.
# kurtosis(첨도): 데이터 분포가 얼마나 뾰족한지(충격 신호 감지) 확인
# skew(왜도): 데이터 분포가 얼마나 비대칭인지 확인
from scipy.stats import kurtosis, skew

# 2. 경로 및 저장 파일 설정
# 원본 데이터가 들어있는 폴더 경로입니다.
data_dir = './data/2nd_test/'
# 추출된 특징(Feature)들을 저장할 최종 CSV 파일명입니다.
output_file = 'bearing_dataset_features.csv'

# 3. 데이터 저장소 준비
# for문을 돌면서 뽑아낸 데이터를 차곡차곡 쌓을 빈 리스트입니다.
# 매번 DataFrame에 append 하는 것보다 리스트에 모았다가 한 번에 변환하는 게 속도가 훨씬 빠릅니다.
data_list = []

print("🚀 데이터 전처리를 시작합니다... (모든 파일 읽는 중)")

# 4. 파일 목록 가져오기 및 정렬 (매우 중요!)
# os.listdir: 해당 폴더에 있는 모든 파일 이름을 가져옵니다.
# if not f.startswith('.'): 맥(Mac)의 .DS_Store 같은 숨김 파일은 제외합니다.
# sorted(): 파일명(예: 2004.02.12...)이 곧 시간 순서이므로, 과거->미래 순으로 정렬합니다. 
# 시계열 분석(Time-Series)에서는 순서가 뒤섞이면 안 되기 때문에 필수입니다.
filenames = sorted([f for f in os.listdir(data_dir) if not f.startswith('.')])

# 5. 전체 파일 루프 시작 (Batch Processing)
# enumerate: 몇 번째 파일인지(idx)와 파일명(filename)을 같이 꺼냅니다.
for idx, filename in enumerate(filenames):
    # os.path.join: 폴더 경로와 파일명을 합쳐서 전체 경로를 만듭니다. (운영체제 호환성 확보)
    file_path = os.path.join(data_dir, filename)
    
    # 예외 처리: 혹시 파일이 아니라 폴더가 섞여 있으면 건너뜁니다.
    if os.path.isdir(file_path):
        continue

    try:
        # 6. 개별 파일 로드
        # NASA 데이터는 Tab(\t)으로 구분되고 헤더가 없습니다.
        df = pd.read_csv(file_path, sep='\t', header=None)
        
        # 7. 분석 대상 센서 선택
        # 여기서는 'Bearing 1' (0번 컬럼)의 데이터만 추출하여 numpy 배열로 변환합니다.
        # 필요하다면 나중에 1, 2, 3번 컬럼도 추가할 수 있습니다.
        signal = df[0].values
        
        # --- [핵심] 기계공학적 Feature 추출 (Time Domain) ---
        # 이 부분이 2만 개의 데이터를 단 5개의 숫자로 압축하는 과정입니다.
        
        # (1) RMS (Root Mean Square, 실효값)
        # 진동의 '에너지 크기'를 나타냅니다. 베어링이 심각하게 망가지면 전체적으로 진동이 커지므로 RMS가 상승합니다.
        rms = np.sqrt(np.mean(signal**2))
        
        # (2) Standard Deviation (표준편차)
        # 진동이 평균값에서 얼마나 퍼져있는지를 나타냅니다.
        std = np.std(signal)
        
        # (3) Maximum Amplitude (최대값)
        # 순간적으로 튀는 가장 큰 충격(Peak)의 크기입니다. 
        # 절댓값(abs)을 취해 +, - 방향 상관없이 가장 큰 진동폭을 잡습니다.
        max_val = np.max(np.abs(signal))
        
        # (4) Kurtosis (첨도) - ★베어링 예지보전의 핵심★
        # 정규분포(3.0)보다 크면 뾰족한 신호(Impulse)가 많다는 뜻입니다.
        # 베어링 초기 결함 시 '탁, 탁' 치는 충격음이 발생하는데, 이때 RMS는 변화가 없어도 첨도는 급격히 올라갑니다.
        kur = kurtosis(signal)
        
        # (5) Skewness (왜도)
        # 파형이 한쪽으로 찌그러진 정도를 봅니다. 회전체 불균형 등이 있을 때 유의미할 수 있습니다.
        skw = skew(signal)
        
        # 8. 결과 모으기
        # 파일 이름(시간 정보)과 위에서 구한 5가지 특징을 딕셔너리로 묶어서 리스트에 추가합니다.
        data_list.append({
            'filename': filename, # 나중에 이 시간 정보를 이용해 시계열 그래프를 그립니다.
            'RMS': rms,
            'Std_Dev': std,
            'Max_Amp': max_val,
            'Kurtosis': kur,
            'Skewness': skw
        })

        # 9. 진행 상황 모니터링
        # 파일이 수천 개라 오래 걸리므로, 100개 처리할 때마다 로그를 찍어 멈추지 않았음을 확인합니다.
        if (idx + 1) % 100 == 0:
            print(f"✅ {idx + 1}개 파일 처리 완료...")

    except Exception as e:
        # 파일을 읽다가 에러가 나도 멈추지 않고 에러 메시지만 출력 후 다음 파일로 넘어갑니다.
        print(f"⚠️ 에러 발생 ({filename}): {e}")

# 10. 최종 데이터프레임 변환
# 리스트에 모아둔 딕셔너리들을 판다스 DataFrame으로 바꿉니다. (행: 시간, 열: 특징들)
final_df = pd.DataFrame(data_list)

# 11. 결과 파일 저장
# index=False: 불필요한 인덱스 번호(0, 1, 2...)는 파일에 저장하지 않습니다.
final_df.to_csv(output_file, index=False)

# 12. 완료 메시지 및 확인
print("-" * 30)
print(f"🎉 모든 작업 완료!")
print(f"총 {len(final_df)}개의 데이터를 처리했습니다.") # 예: 984개
print(f"결과 파일 저장됨: {output_file}")
print("-" * 30)

# 데이터가 잘 만들어졌는지 앞/뒤 5줄씩 확인
print(final_df.head())
print(final_df.tail())