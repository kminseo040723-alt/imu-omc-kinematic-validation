# IMU–OMC Kinematic Validation

**2026-2 센서이론·신호처리 I · MOBI311 · Minseo Kim**

발목에 부착한 IMU의 자세와 위치 추정값을 Vicon 광학 모션캡처(OMC) 데이터와 비교한 프로젝트입니다. 
## 폴더 구성 

| 폴더 | 내용 |
|---|---|
| `raw_csv/` | 본 측정의 원본 `imu.csv`, `omc_vicon.csv` 각 1개 |
| `code/` | Tasks 2–5 최종 분석 코드, Task 1 마커 진단·복원 실험 코드 |
| `results/` | 최종 분석의 CSV·JSON·PNG, Task 1 결과와 후속 분석에 필요한 처리 데이터 |
| `report/` | 보고서용 폴더 |

## 주요 파일

| 파일 | 용도 |
|---|---|
| [raw_csv/imu.csv](raw_csv/imu.csv) | IMU 원본, 58,488 samples |
| [raw_csv/omc_vicon.csv](raw_csv/omc_vicon.csv) | Vicon 내보내기 CSV, 60,806 frames, 100 Hz |
| [code/task 2-5](code/task%202-5) | 동기화, 고정 공간 보정, 자세·위치 추정 |
| [code/task 1](code/task%201) | 마커 거리 검사, 인공 결측 복원 검증 |
| [results/results_summary.json](results/results_summary.json) | Tasks 2–5 전체 수치 요약 |
| [results/task1_summary.json](results/task1_summary.json) | Task 1 진단·인공 결측 검증 요약 |
| [results/omc_refined_task1.csv](results/omc_refined_task1.csv) | 최종 Tasks 2–5에서 사용한 OMC 좌표 |
| [results/task1_initial_OL_to_OG.csv](results/task1_initial_OL_to_OG.csv) | 최종 분석에서 사용한 초기 좌표축 |
| [results/task1_template_mm.csv](results/task1_template_mm.csv) | 최종 분석에서 사용한 마커 기준 형상 |
| [report/MinSeo_Kim_2023170677.pdf](report/MinSeo_Kim_2023170677.pdf) | 최종 report |


## 분석 방법과 주요 결과

각속도 크기로 시계를 먼저 동기화한 후, 5–120 s의 OMC·IMU 각속도에서 SVD/Kabsch 고정 회전과 자이로 바이어스를 추정합니다. 보정된 자이로를 quaternion으로 적분해 자세를 구하고, 전역 좌표계의 중력 제거 가속도에 BPF와 이중 적분을 적용해 위치를 추정합니다.

| 항목 | 저장된 결과 |
|---|---:|
| 동기화 각속도 크기 상관계수 | 0.971786 |
| 시간 배율 α | 1.00421650 |
| 시간 오프셋 β | −2.473879 s |
| 평균 자세 오차, gravity-only | 91.37° |
| 평균 자세 오차, dynamic SVD + bias | 19.82° |
| 최종 자세 오차, dynamic SVD + bias | 43.73° |
| 위치 RMSE, 0.3–5 Hz | 1,966.72 mm |
| 위치 RMSE, 0.5–7 Hz | 1,892.02 mm |
| 위치 RMSE, 1.1–10 Hz | 2,293.92 mm |

![자세 오차](results/orientation_error.png)

![0.5–7 Hz 위치 비교](results/trajectory_0.5_7Hz.png)

OMC는 자세 초기화와 보정의 기준으로 사용했습니다. IMU 위치는 마커 중심과 같다고 가정했으며, 실제 lever arm 보정이나 ZUPT는 적용하지 않았습니다. Vicon CSV는 내보내기 전에 복원이 수행된 자료이며, 기존 복원 이력은 CSV만으로 판별할 수 없습니다.
