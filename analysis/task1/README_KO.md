# Task 1: OMC Marker Refinement

## 실행

Python 3.9 이상이 필요합니다. ZIP을 푼 폴더에 Vicon 파일 record4(1).csv를 넣고 그 폴더에서 터미널을 여세요.

```bash
python -m pip install numpy pandas scipy matplotlib
python task1_marker_refinement.py --input "record4(1).csv" --demo
```

결과는 task1_results 폴더에 저장됩니다. 입력 파일은 덮어쓰지 않습니다.
PNG를 저장하므로 팝업 그래프 창은 뜨지 않습니다.

다른 위치에 저장:

```bash
python task1_marker_refinement.py --input "record4(1).csv" --out "results/task1" --demo
```

의심 좌표 수정은 기본적으로 꺼져 있습니다. 진단 그래프를 검토한 뒤에만 아래 옵션을 사용하세요. 이 결과는 기존 기본 결과와 별도 폴더에 저장하세요.

```bash
python task1_marker_refinement.py --input "record4(1).csv" --out "results/task1_repair" --repair-suspects --demo
```

## 코드가 하는 일

1. Vicon Trajectories 헤더, Sampling frequency, Frame, XYZ(mm)를 읽습니다.
2. 첫 Export Frame을 t=0으로 두고 Frame 번호 차이 / Sampling frequency로 시간을 만듭니다. IMU와 동기화한 시간이 아닙니다.
3. Marker 간 거리 6개를 계산합니다. 각 거리의 중앙값으로부터 max(5 mm, 6 × 1.4826 × MAD)를 넘는 Frame을 표시합니다. 5 mm와 6은 분석자가 선택한 초기 진단 기준이며 장비 정확도나 통계적 신뢰수준이 아닙니다.
4. 특정 Marker와 연결된 세 거리만 모두 기준을 벗어나고 나머지 세 Marker의 Triangle은 정상일 때 해당 Marker를 suspect로 표시합니다. 애매한 경우는 Frame Flag만 남깁니다. 이 방식은 모든 이상점을 찾는 방법이 아닙니다.
5. 완전한 Frame 중 여섯 거리가 전체 거리 중앙값에 가까운 Frame 하나를 선택해 Empirical template을 만듭니다. 안내문의 Marker 배치를 그대로 쓰지 않습니다. 한 Frame의 오차가 Template에 남을 수 있고, 부착체 변형이 있으면 Rigid-body 가정이 성립하지 않을 수 있습니다.
6. 세 개의 비일직선상 Marker가 관측되면 Kabsch fitting(SVD, proper rotation)으로 한 개의 결측 Marker를 복원합니다. Anchor 간 거리와 fitting residual이 기준을 통과해야 합니다. 원래 관측값만 Anchor로 사용합니다.
7. Rigid 복원이 안 되는 결측은 양쪽 관측 경계 간 시간이 0.20 s 이하인 내부 구간에 한해 보간합니다. 양쪽에 두 점씩 있으면 Natural cubic spline, 부족하면 Linear interpolation을 사용합니다. 이 0.20 s 제한은 Temporal interpolation에만 적용합니다. 긴 구간과 Extrapolation이 필요한 구간은 unresolved/NaN으로 남깁니다.
8. --demo를 사용하면 정상 구간에 인공 결측을 넣어 별도 검증합니다. 하나의 Marker를 약 0.10 s 숨기는 실험과 네 Marker를 같은 구간에서 모두 숨기는 실험입니다. 가린 구간은 Template 및 거리 기준 추정에서도 제외됩니다. 검증 구간은 결과 RMSE를 보고 고르지 않습니다.

## 출력

- omc_refined.csv: 원본 Frame, omc_time(s), Marker 1–4 XYZ(mm), 각 Marker의 처리 상태.
- quality_flags.csv: 거리, Pair/Frame Flag, Marker suspect, 원본 결측 여부.
- pair_distance_summary.csv: Marker 거리의 중앙값, 범위, 기준, Flag 수.
- pair_distances.png: 여섯 거리의 전체 시간 변화. 빨강은 의심 지점입니다.
- empirical_template_mm.csv: 선택한 Template의 상대 XYZ.
- actual_gap_*.png: 실제 입력 결측 또는 opt-in으로 Mask한 구간의 복원 확대 그림. 해당 구간이 없으면 생성되지 않습니다.
- demo_one_marker.png / demo_all_markers.png: 인공 결측 검증 그래프.
- demo_*.csv: 인공 결측을 복원한 별도 데이터. 실제 Task 2 이후 입력으로 사용하지 마세요.
- summary.json: 파라미터, 데이터 요약, 검증 오차 및 한계.

상태: observed=입력 CSV에 존재했던 값, rigid=강체 복원, spline/linear=시간 보간, unresolved=복원하지 못함.
여기서 observed는 'Vicon 원시 관측으로 확인됨'을 뜻하지 않습니다. 기존 Vicon 복원 값도 입력 CSV에 존재하면 observed로 표시됩니다. 기존 복원 이력은 이 CSV에서 판별할 수 없습니다.

CSV Marker 1/2/3/4는 각각 Export 열 순서 20261023:11/12/13/14입니다. 안내문의 Marker 번호와 동일하다고 가정하지 않습니다. Global 좌표의 축·부호는 변경하지 않습니다.

## 제공 데이터 실행 결과

example_results 폴더는 record4(1).csv에 기본 설정 + --demo를 실행한 결과입니다.

- Frame 수: 60,806; 시간 길이: 608.05 s; Sampling frequency: 100 Hz.
- 빈 좌표: 0개. 따라서 기본 omc_refined.csv의 XYZ는 입력과 같습니다.
- Geometry Flag: 1,845 Frame. 오류로 확정된 Frame 수가 아닙니다.
- 보수적으로 특정 Marker에 귀속된 suspect: 99 Marker-Frame. 수정 옵션은 꺼져 있습니다.
- 인공 결측 구간: 297.63–297.72 s, 10 Sample.
- 한 Marker 인공 결측: Rigid 복원 Position RMSE 약 8.475 mm.
- 네 Marker 동시 인공 결측: Spline 복원 Position RMSE 약 0.397 mm.

Position RMSE = sqrt(mean(||p_estimated - p_exported||²)). 네 Marker 실험은 가린 모든 Marker와 시간에 대해 평균합니다.
이는 한 구간에서 Export된 좌표와의 일치도이며 절대적인 장비 정확도가 아닙니다. 실험 조건이 다르므로 두 RMSE로 일반적인 방법의 우열을 판단하면 안 됩니다. 한 Marker의 Rigid 복원 오차가 남는 결과도 그대로 보존했습니다.

보고서에는 실제 입력의 결측 유무, Vicon에서 이미 수행한 복원(기억/기록에 근거), 인공 결측 검증을 구분해서 서술하세요. 인공 구간을 실제 측정 결측이라고 쓰면 안 됩니다.

## 검증

실제 Vicon 파일 기본 실행 완료. 별도의 알려진 강체 회전/이동 데이터에서 한 Marker Rigid 복원, 전체 Marker 단기 보간, 장기 구간 NaN 유지, 관측값 보존을 확인했습니다.
