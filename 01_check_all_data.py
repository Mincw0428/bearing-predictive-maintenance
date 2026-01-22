import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 데이터 폴더 경로 (사용자 환경에 맞게 수정)
data_dir = "./data/2nd_test"  # 또는 2nd_test (Set 2라면 2nd_test 경로)

# ※ 주의: 파일이 너무 많아서 일부만 샘플링해서 그립니다.
files = sorted([f for f in os.listdir(data_dir) if not f.startswith('.')])
rms_history = {'B1': [], 'B2': [], 'B3': [], 'B4': []}

print("데이터 읽는 중...")
# 10개씩 건너뛰며 읽기 (속도 위해)
for file in files[::10]: 
    df = pd.read_csv(os.path.join(data_dir, file), sep='\t', header=None)
    # Set 2 기준: col 0=B1, 1=B2, 2=B3, 3=B4
    rms_history['B1'].append(np.sqrt(np.mean(df[0]**2)))
    rms_history['B2'].append(np.sqrt(np.mean(df[1]**2)))
    rms_history['B3'].append(np.sqrt(np.mean(df[2]**2)))
    rms_history['B4'].append(np.sqrt(np.mean(df[3]**2)))

plt.figure(figsize=(12, 6))
plt.plot(rms_history['B1'], label='Bearing 1 (Failure)', color='red')
plt.plot(rms_history['B2'], label='Bearing 2', color='blue', alpha=0.3)
plt.plot(rms_history['B3'], label='Bearing 3', color='green', alpha=0.3)
plt.plot(rms_history['B4'], label='Bearing 4', color='orange', alpha=0.3)
plt.title("RMS Trend of All 4 Bearings")
plt.legend()
plt.show()