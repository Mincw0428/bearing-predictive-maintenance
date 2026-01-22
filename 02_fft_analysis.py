# 1. 라이브러리 추가
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # 수치 계산(배열 처리, 절댓값 계산 등)을 위해 필수입니다.
from scipy.fft import fft, fftfreq  # 과학 계산용 라이브러리 Scipy에서 고속 푸리에 변환(FFT) 도구를 가져옵니다.

# 2. 데이터 로드 (이전 단계와 동일)
file_path = './data/2nd_test/2004.02.12.10.32.39' 
dataset = pd.read_csv(file_path, sep='\t', header=None)
dataset.columns = ['Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4']

# 분석할 데이터 추출
# .values를 사용해 판다스 시리즈를 넘파이 배열(numpy array)로 변환합니다. 
# 계산 속도가 더 빠르고 Scipy 함수에 넣기 좋습니다.
data = dataset['Bearing 1'].values

# -------------------------------------------------------------------------
# 3. FFT 설정 (기계공학적 도메인 지식이 반영되는 구간)
# -------------------------------------------------------------------------

# Fs (Sampling Rate): 샘플링 주파수입니다. 
# NASA 데이터셋 문서를 보면 20kHz로 측정했다고 명시되어 있습니다. 
# 즉, 1초에 20,000번 데이터를 찍었다는 뜻입니다. 이 값이 틀리면 주파수 축(X축)이 완전히 틀어집니다.
Fs = 20000 

# T (Sampling Interval): 샘플링 간격입니다. (1초 / 20000번 = 0.00005초마다 측정)
T = 1 / Fs

# N: 데이터의 전체 개수입니다. (NASA 데이터는 보통 20,480개)
N = len(data) 

# -------------------------------------------------------------------------
# 4. FFT 계산 (수학적 변환)
# -------------------------------------------------------------------------

# fft(data): 시간 영역의 데이터를 주파수 영역으로 변환합니다.
# 결과값은 '복소수(Complex Number)' 형태(실수부+허수부)로 나옵니다.
yf = fft(data)

# fftfreq(N, T): FFT 결과의 X축(주파수)을 생성해주는 함수입니다.
# [:N//2]: 중요! FFT 결과는 대칭(Symmetric)입니다. 
# 0Hz를 기준으로 양의 주파수와 음의 주파수가 대칭되므로, 우리는 절반(양수 부분)만 필요합니다.
# Nyquist 이론에 따라 최대 주파수는 Fs의 절반인 10,000Hz까지만 유효합니다.
xf = fftfreq(N, T)[:N//2] 

# Amplitude 계산 (Y축, 진동의 크기)
# yf[0:N//2]: 앞서 자른 X축에 맞춰 Y축 데이터도 절반만 가져옵니다.
# np.abs(): 복소수에서 크기(Magnitude)를 구하기 위해 절댓값을 취합니다.
# 2.0/N: 정규화(Normalization) 과정입니다. 
# FFT의 결과값은 데이터 길이에 비례해 커지므로, 원래 신호의 진폭과 맞추기 위해 데이터 개수(N)로 나누고 2를 곱해줍니다.
amplitude = 2.0/N * np.abs(yf[0:N//2]) 

# -------------------------------------------------------------------------
# 5. 시각화 (Time Domain vs Frequency Domain 비교)
# -------------------------------------------------------------------------
plt.figure(figsize=(12, 8)) # 이번엔 위아래로 그릴 거라 세로 길이를 8로 늘렸습니다.

# (1) 시간 영역 그래프 (Time Domain)
plt.subplot(2, 1, 1) # 2행 1열 중 첫 번째 그림
plt.plot(data, color='blue', alpha=0.5)
plt.title('Time Domain (Raw Signal)') # "시간에 따른 진동 변화"
plt.xlabel('Time Sample')
plt.ylabel('Amplitude')
plt.grid(True)

# (2) 주파수 영역 그래프 (Frequency Domain) - 여기가 핵심 결과물!
plt.subplot(2, 1, 2) # 2행 1열 중 두 번째 그림
plt.plot(xf, amplitude, color='red') # X축: Hz, Y축: 진폭
plt.title('Frequency Domain (FFT Spectrum)') # "주파수 성분별 진동 크기"
plt.xlabel('Frequency (Hz)') # X축은 0 ~ 10,000Hz가 됩니다.
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout() # 그래프끼리 겹치지 않게 간격 자동 조절
plt.show()