# 1. 라이브러리 임포트 (도구 상자 준비)
import pandas as pd  # 데이터 분석의 핵심 도구입니다. 엑셀처럼 표(DataFrame) 형태로 데이터를 다루기 위해 사용합니다.
import matplotlib.pyplot as plt  # 데이터를 그래프로 시각화(그리기)하기 위한 라이브러리입니다.
import os  # 운영체제(OS)와 상호작용하기 위한 도구로, 파일 경로 확인이나 폴더 내 파일 목록 읽기 등에 사용합니다.

# 2. 데이터 파일 경로 설정
# 분석할 대상 파일의 위치를 변수에 저장합니다.
# './'는 현재 파이썬 파일이 있는 위치(Current Directory)를 의미합니다.
file_path = './data/2nd_test/2004.02.12.10.32.39' 

# 3. 파일 존재 여부 확인 (에러 방지 안전장치)
# os.path.exists() 함수는 괄호 안의 경로에 실제 파일이 있으면 True, 없으면 False를 반환합니다.
if not os.path.exists(file_path):
    # 파일이 없을 경우 당황하지 않도록 명확한 에러 메시지를 출력해줍니다.
    print(f"❌ 오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {file_path}")
else:
    # 파일이 정상적으로 존재할 경우 실행되는 블록입니다.
    print(f"✅ 파일을 찾았습니다! 데이터를 로드합니다...")

    # 4. 데이터 불러오기 (핵심)
    # pd.read_csv: 파일을 읽어 DataFrame으로 변환합니다.
    # sep='\t': NASA 데이터는 쉼표(,)가 아니라 탭(Tab) 키로 데이터가 구분되어 있어서 이 옵션이 필수입니다.
    # header=None: 원본 파일 첫 줄에 컬럼명(제목)이 없고 바로 데이터가 시작되므로, 첫 줄을 제목으로 쓰지 말라고 설정합니다.
    dataset = pd.read_csv(file_path, sep='\t', header=None)

    # 5. 컬럼 이름 지정
    # 위에서 header=None으로 불렀기 때문에 컬럼명이 0, 1, 2, 3으로 되어 있습니다.
    # NASA 2nd Test 문서를 보면 4개의 채널이 각각 Bearing 1~4의 진동 센서 값임을 알 수 있습니다. 보기 좋게 이름을 붙여줍니다.
    dataset.columns = ['Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4']

    # 6. 데이터 형태 확인 (검증)
    # dataset.shape: (행의 개수, 열의 개수)를 튜플 형태로 알려줍니다.
    # NASA 데이터는 20kHz로 1초간 샘플링했기 때문에 보통 20,480개의 행(row)이 나와야 정상입니다.
    print(f"데이터 크기: {dataset.shape}") # 예상 출력: (20480, 4)
    
    # dataset.head(): 데이터의 상위 5개 행을 출력하여 데이터가 깨지지 않고 잘 들어왔는지 눈으로 확인합니다.
    print(dataset.head())

    # 7. 데이터 시각화 (그래프 그리기)
    # plt.figure: 그림을 그릴 도화지(Figure)를 생성합니다.
    # figsize=(12, 4): 도화지의 크기를 가로 12인치, 세로 4인치로 설정합니다. (옆으로 긴 그래프)
    plt.figure(figsize=(12, 4))
    
    # plt.plot: 선 그래프를 그립니다.
    # dataset['Bearing 1']: X축은 생략(자동으로 인덱스 사용), Y축 데이터로 베어링 1번 컬럼을 사용합니다.
    # color='blue': 선 색상을 파란색으로 지정합니다.
    # alpha=0.5: 투명도를 0.5(50%)로 설정합니다. 데이터가 2만 개로 빽빽해서 겹쳐 보일 때 투명도를 주면 밀도 확인에 좋습니다.
    plt.plot(dataset['Bearing 1'], color='blue', alpha=0.5)
    
    # 그래프의 제목, X축 이름, Y축 이름을 설정하여 그래프가 무엇을 의미하는지 설명합니다.
    plt.title('Raw Vibration Signal (Bearing 1)') # 제목: 원본 진동 신호 (베어링 1)
    plt.xlabel('Time (Sample)')   # X축: 시간 (샘플 순서, 0 ~ 20480)
    plt.ylabel('Amplitude')       # Y축: 진폭 (진동의 크기)
    
    # plt.grid(True): 그래프 배경에 격자(모눈)를 그려서 값을 읽기 편하게 만듭니다.
    plt.grid(True)
    
    # plt.show(): 메모리에 그려둔 그래프를 실제 화면에 출력합니다. 이 코드가 없으면 그래프가 안 보일 수 있습니다.
    plt.show()