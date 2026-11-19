# Light SuperMimic: 로컬 작업 코드를 Colab GPU에서 학습

이 경로는 현재 저장소의 Light 로봇 정의·결합 PD·screened CMU/LAFAN 데이터와,
버전을 고정해 로컬에 복사한 공식 SuperMimic 학습 코어를 함께 전달한다.
공식 구현은 IsaacLab 전용이 아니라 **mjlab + MuJoCo Warp + PyTorch** 기반이다.
기존 `src/super_mimic`의 G1 운반 실험은 그대로 보존한다.

## 여기서 바로 Colab GPU로 실행

공식 Colab CLI로 로컬 작업 코드를 직접 전송할 수 있다. 노트북이나 Drive 업로드는 필요하지 않다.

```bash
uv tool install google-colab-cli  # 최초 1회; 이미 설치되어 있으면 생략
colab sessions                  # 최초 로그인 확인
python3 humanoid.py launch --experiment light_supermimic_smoke --runtime colab \
  --submit --colab-gpu T4
```

현재 미커밋 소스와 필요한 데이터만 SHA256 검증 후 새 전용 GPU 세션에 전달한다.
선택한 실험의 Python 환경을 설치하고 같은 학습기를 실행한다.
실행 중 터미널을 유지한다. 로그와 완료된 checkpoint를 주기적으로 PC의
`runs/<experiment>/<run-id>/`로 회수하며, 종료 시 최종 결과를 가져오고 전용 세션을 해제한다.
`Ctrl+C`도 회수 가능한 결과를 가져온 뒤 세션을 해제한다. 다른 Colab 세션은 건드리지 않는다.
전송 묶음과 세션 종료 여부는 `runs/colab/<experiment>_<run-id>/submission.json`에 기록한다.
`--results-root /path/to/runs`로 PC의 결과 위치를 변경할 수 있다.

전체 모션 실험도 같은 방식이다. 할당 가능한 GPU는 계정과 현재 자원 상황에 따라 다르다.

```bash
python3 humanoid.py launch --experiment light_supermimic --runtime colab \
  --submit --colab-gpu T4
```

무료 T4에서 전송량을 줄여 먼저 학습을 시험하는 설정은 다음과 같다.

```bash
python3 humanoid.py launch --experiment light_supermimic_t4_trial --runtime colab_t4 \
  --submit --colab-gpu T4
```

trial은 CMU 16개·LAFAN 4개 원본 파일의 accepted interval을 사용한다. 별도 validation
파일 3개는 학습에 사용하지 않는다. `configs/motions/light_supermimic_t4_trial.json`에
원본 manifest의 SHA256과 선택 내역을 고정하며, 원본 모션·구간·파일 해시는 수정하지 않는다.
전체 데이터 약 593MiB를 전송하던 시도가 원격 세션 종료로 중단되어, 최초 실행은
약 14MiB의 데이터로 전송 시간을 줄인다. 이 subset 실험으로 전체 데이터의 추종 품질을 판단하지 않는다.

전체 모션을 사용하는 짧은 pilot은 다음과 같다.

```bash
python3 humanoid.py launch --experiment light_supermimic_pilot --runtime colab_t4 \
  --submit --colab-gpu T4
```

trial과 pilot은 51,200 transition, 6,400 transition마다 checkpoint를 저장한다.
`colab_t4` runtime은 64개 환경, WM batch 64, 정책 batch 32와 eager 업데이트를 선택한다.
네트워크 크기와 WM/정책 rollout horizon은 본 실험의 기본값을 유지한다.
pilot의 학습 데이터는 전체 screened train split이며 validation은 학습에 사용하지 않는다.
명령은 결제나 구독 변경을 수행하지 않는다. 무료 GPU의 할당과 실행 시간은 Colab 가용 자원에 따라 달라진다.

학습된 trial checkpoint에서 처리속도를 측정하고 약 200만 transition까지 이어가려면:

```bash
python3 humanoid.py launch --experiment light_supermimic_t4_long --runtime colab_t4_optimized \
  --submit --colab-gpu T4 \
  --resume runs/light_supermimic_t4_trial/free_t4_trial_20261117/training/checkpoints/step_51200
```

같은 20개 원본 모션과 모델을 사용한다. 초기 비교에서 64환경 eager는 약 17.5 transition/s,
512~4,096환경 CUDA Graph는 약 418~444 transition/s였다. T4용 profile은 Muon의
행렬 직교화에 FP16을 선택한다. 정규화·가중치·optimizer momentum·모델·물리는 FP32를 유지하며,
원본 NS 다항식과 계수는 그대로 사용한다. 이는 BF16을 하드웨어로 가속하지 못하는 T4를 위한
정밀도 선택이며, `training.muon_dtype=bfloat16`으로 원본 경로를 선택할 수 있다.

같은 체크포인트에서 1,024/2,048/4,096개 환경의 CUDA Graph 실행을 비교한다.
선택된 설정의 목표 학습 시간이 20분 이상으로 예상되면 torch.compile도 시험한다.
초기 컴파일과 graph 준비 시간을 제외한 실제 시뮬레이션·WM·정책
처리속도를 측정하고 VRAM 여유가 12% 이상인 설정 중 가장 빠른 것을 선택한다.
컴파일에 실패하거나 제한 시간을 초과하면 성공한 CUDA Graph 설정을 사용한다.

WM batch 256·정책 batch 128을 사용하며, 환경 수에 따라 업데이트 횟수를 2/4/8/16으로 조정한다.
기존 64환경·WM batch 64·정책 batch 32·업데이트 1회와 transition당 학습 샘플 비율을 유지한다.
가중치·optimizer·RNG는 복원하고 replay는 새로 채운다. 배치 크기와 업데이트 빈도 변경은
허용하되 모델 구조·horizon·물리 설정·데이터·제어기 검증은 유지한다.

실측 결과와 선택은 `training/tuning.json`, 실제 학습 설정은 `training/selected_settings.json`,
개별 측정은 `training/benchmark/*/performance.json`에 저장한다. 비교 실험의 가중치를 이어 쓰지 않고,
선택 후 원래 체크포인트에서 본 학습을 시작한다. 최종 transition 수는 환경 배치 크기로 올림할 수 있다.
직접 제출의 결과 전송은 각 그룹의 최신 checkpoint 2개를 포함해 압축파일 크기를 제한한다.
이미 PC로 받은 checkpoint와 VM에 있는 이전 checkpoint는 삭제하지 않는다.

검증용 실행은 4개 환경·40 transition·3회의 WM/BPTT 업데이트로,
학습 수렴이나 모션 추종 품질을 확인하는 실험은 아니다.

## 노트북에서 실행할 묶음 만들기

먼저 작은 검증용 묶음을 실행한다.

```bash
python3 humanoid.py launch --experiment light_supermimic_smoke --runtime colab \
  --bundle-data --output runs/colab/light_supermimic_smoke
```

생성된 **폴더 전체**를 Drive의 `MyDrive/humanoid_rl/colab/`에 복사한다.
그 안의 `train_colab.ipynb`를 Colab에서 열고 GPU 런타임을 선택한 뒤 셀을 실행한다.
현재 미커밋 코드도 source.zip에 포함하므로 코드를 GitHub에 먼저 올릴 필요는 없다.
노트북은 Python 3.13과 전용 환경을 설치하며 기존 Colab 커널 환경을 덮어쓰지 않는다.

전체 모션을 사용하려면:

```bash
python3 humanoid.py launch --experiment light_supermimic --runtime colab \
  --bundle-data --output runs/colab/light_supermimic
```

기본 본 실험은 128개 환경, 1M 실제 transition, WM horizon 8, 정책 BPTT horizon 32다.
할당된 GPU에 맞게 `--set training.num_envs=...`, `--set runner.policy.batch_size=...` 등을
지정한다. 최종 설정과 실제 패키지 버전은 실행 결과에 기록된다.

## 무엇을 학습하는가

- `qpos/qvel`은 **reference와 초기 상태**다. 실행한 action의 물리 결과를 대신하지 않는다.
- 각 screening interval을 독립 clip으로 변환하며 제거된 구간, 원본 경계, train/validation을 연결하지 않는다.
- free-joint의 각속도는 MuJoCo body frame에서 world frame으로 회전한다. root quaternion은 wxyz다.
- 정책이 Light 시뮬레이터에서 실제 행동을 실행하고 replay를 갱신한다.
- 공식 world-model learner가 이 실제 transition으로 동역학을 fitting한다.
- 공식 actor-critic BPTT learner가 짧은 imagined rollout의 추종 보상을 통해 정책을 갱신한다.
- 시뮬레이터와 월드 모델의 입력 모두 같은 비대칭 관절 범위 → PD 목표각 함수를 사용한다.
- 발목·몸통의 motor-coordinate 결합 PD와 원본 collision hull·관성을 유지한다.

현재 Light adapter는 generalized coordinates, uniform reference resets,
state 관측 정책과 고정 물리 설정을 사용한다. 공식 학습 코어는 그대로 사용하지만,
로봇/데이터/실행 스케줄/저장 경로를 추가한 통합이며 논문의 전체 benchmark 재현은 아니다.

## NaN과 저장·재개

2.5ms physics × 8 substep으로 50Hz 제어를 유지하고, 기존 Light BFM과 같은 Newton 100회·line search 50회 상한을 사용한다.
이 설정은 이전의 짧은 NaN 후보 검사에
근거하며 장기 안정성 인증은 아니다. 데이터는 삭제하거나 재작성하지 않는다.
source checksum, finite 값, 관절 순서, quaternion, 50Hz, 원본 trajectory 경계를 검사한다.
실행 중 nonfinite action/state/reward가 나오면 학습기 업데이트 전에 중단하고
`nonfinite_physics.pt`에 상태와 reference frame을 저장한다.

결과는 공통 `runs/<experiment>/<run-id>/training/` 아래에 쓴다.
`checkpoints/step_N/`은 actor·WM·critic·optimizer·RNG를 저장한다.
replay는 저장하지 않으며 재개 시 새 episode에서 다시 채운다. GPU 실행 상태의 정확한 연속 재생은 아니다.
직접 제출은 완료된 checkpoint를 PC로 회수한다. 노트북 실행은 공통 runtime 설정에 따라
완료된 checkpoint를 Drive에 복사한다.

```bash
python3 humanoid.py launch --experiment light_supermimic --runtime colab \
  --resume /path/to/checkpoints/step_N --set training.total_env_steps=2000000 \
  --bundle-data --output runs/colab/light_supermimic_resume
```

## PC GPU에서 연결만 검증할 때

Colab에서 학습하는 것과 동일한 코드를 사용한다.

```bash
python3 humanoid.py setup --method supermimic --task Humanoid-Light-Motion-Tracking
python3 humanoid.py launch --experiment light_supermimic_smoke --runtime local
```

Light는 `.venv_supermimic_light`, 기존 G1 실험은 `.venv_super_mimic`을 사용한다.
Light의 공식 소스 commit은 `environments/supermimic_sources.json`, 의존성은
`environments/light_supermimic/uv.lock`으로 고정한다. 이 통합 환경은 기존 native 작업과 같은
Torch 2.14.1을 고정하며 공식 저장소의 최소 버전 요구사항도 만족한다.
공식 학습 코어와 의존성의 최소 버전 선언은 수정하지 않는다.
