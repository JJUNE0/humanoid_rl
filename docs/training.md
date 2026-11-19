# 공통 학습 실행

실험은 **로봇·태스크·방법론**, runtime은 **실행 위치·장치·경로·자원 설정**을 정의한다.
같은 실험을 로컬 GPU 또는 Colab GPU에서 실행한다. `--runtime colab --submit`은
Colab CLI로 원격 실행하고 결과를 PC로 가져온다. `--submit`을 생략한 Colab export는
실행 가능한 소스·설정·노트북 묶음을 만든다.

## 시작

저장소 루트에서 실행한다. 공통 CLI의 목록 조회, dry-run, Colab export에는 Python 3.10 이상만
필요하며 torch/mjlab 설치나 GPU는 필요하지 않다.

```bash
python3 humanoid.py list
python3 humanoid.py launch --experiment light_bfm_zero --runtime local --dry-run
python3 humanoid.py setup --method bfm_zero
python3 humanoid.py launch --experiment light_bfm_zero_smoke --runtime local
```

설치 후에는 `.venv/bin/humanoid` 또는 `python -m humanoid_rl`로도 같은 CLI를 쓸 수 있다.
기존 `train.py`, `play.py`, `humanoid-train`, `humanoid-play`와
`humanoid-rl --task ...`는 기존 동작을 유지한다. `src` 안의 각 `train.py`는 구현이며,
새 실험은 공통 명령으로 선택한다.

현재 지원 조합:

| 방법론 | 태스크 | 실행 환경 |
| --- | --- | --- |
| `bfm_zero` | Light/G1/H1 Motion Prior | Python 3.13, mjlab 1.6 |
| `flashsac` | G1 Carry Box, Cartpole; AMP/ADD 보상 선택 가능 | 위와 동일 |
| `supermimic` | 기존 G1 Carry/Stand/Walk | Python 3.10, mjlab 1.4 |
| `supermimic` | Light Motion Tracking | Python 3.13, 공식 mjlab research fork |

`supermimic`은 Light에는 공식 학습 코어와 전용 환경을, 기존 G1에는 보존한 자체 구현을 연결한다.
[Light 코드를 Colab에서 실행하는 방법](light_supermimic.md)을 참고한다.
`setup --method supermimic --task Humanoid-Light-Motion-Tracking`으로 Light 환경을 선택한다.
지원하지 않는 방법론/태스크 조합은 실행 전에 거절한다.

## 코드 소유권

```text
src/humanoid_rl/cli.py          공통 사용자 명령
src/humanoid_rl/methods/        방법론 등록과 기존 학습기 연결
src/humanoid_rl/algorithm/      기존 학습 수식·버퍼·체크포인트 구현
src/humanoid_rl/tasks/          환경·관측·보상·종료
src/humanoid_rl/launchers/      설정 해석·로컬 실행·Colab 제출/export·영속 저장
src/robots/                    공유 로봇 정의·제어기·asset
src/humanoid_retarget/          공유 모션 처리
src/super_mimic/                격리된 기존 G1 학습 구현
configs/experiments/            방법론·태스크·학습 설정
configs/runtimes/               local/colab 실행 설정
environments/                   방법론별 Python·mjlab·venv 선택
data/                          새 실행의 재생성 가능한 cache (Git 제외)
runs/                          새 실행 결과·Colab 묶음 (Git 제외)
```

학습기 전체를 하나의 Trainer 상속 구조로 합치지 않는다. 각 방법론은 자기 업데이트·데이터·
체크포인트 형식을 유지하고, 공통 실행부는 해당 방법론의 공개 CLI만 호출한다.
새 방법론은 `methods/`의 adapter와 registry, 필요하면 별도 environment profile을 추가한다.
선택 시 다른 방법론의 학습 라이브러리를 import하지 않는다.

기존 파일과 데이터는 일괄 이동하지 않았다. 과거 체크포인트와 절대 경로를 보존하면서 새 실행의
기록 위치부터 통일한다. 기존 모션은 현재 `outputs/` 아래 경로로 계속 참조할 수 있다.

## 실험과 runtime 설정

기존 알고리즘별 JSON defaults → 실험 `config` → runtime의 방법론별 `overrides` →
명시적 `--set` 순으로 적용한다. `device`와 출력 경로는 runtime이 소유한다.

```bash
python3 humanoid.py launch --experiment light_bfm_zero --runtime local \
  --set training.num_envs=128 --set training.compile=false --dry-run
```

runtime의 `overrides` 예:

```json
{"bfm_zero": {"training": {"num_envs": 128, "buffer_size": 131072}}}
```

기본 local/colab profile은 학습 hyperparameter를 바꾸지 않는다. 할당된 GPU에 맞는 환경 수와
buffer/batch 크기를 명시적으로 선택한다. `--dry-run`은 파일 생성이나 학습을 수행하지 않는다.

실험 경로에는 `${repo}`, `${data_root}`, `${cache_root}`, `${run_dir}`를 사용할 수 있다.
`--data-root`, `--cache-root`, `--run-root`, `--persistent-root`, `--device`로 runtime 경로와
장치를 변경할 수 있다. BFM manifest의 record 경로는 `motion.manifest_root` 기준이다.

BFM과 SuperMimic adapter는 설정에서 필요한 입력을 추론한다. 추가 파일이 필요한 실험은
`inputs`에 `{"path": "${data_root}/...", "kind": "file"}`를 선언한다.
`kind`는 `file`, `directory`, `motion_manifest` 중 하나다. 모션 manifest는 train/validation에
등장하는 파일을 함께 수집하고, export 시 원본 SHA256을 검증한다.

Light BFM 새 preset은 2.5ms physics × 8 substep과 인위적 lying reset 0을 사용한다.
이는 기존 짧은 NaN 후보 검사에 근거한 실험 설정이다. 장기 학습 안정성/성능을 검증한 것은 아니다.
기존 알고리즘 defaults와 예전 실행 스크립트의 5ms 설정은 변경하지 않았다.

## Colab

노트북 없이 터미널에서 직접 실행하려면 공식 `google-colab-cli`를 설치하고 로그인한 뒤:

```bash
python3 humanoid.py launch --experiment light_supermimic_smoke --runtime colab \
  --submit --colab-gpu T4
```

직접 제출은 필요한 데이터도 자동으로 포함한다. 전용 세션을 만들고 소스·입력의 SHA256을
검증한 뒤 학습한다. 실행 중 터미널을 유지하면 로그와 완료된 checkpoint가 주기적으로
PC의 `runs/<experiment>/<run-id>/`에 복사된다. 최종 회수 후 전용 세션을 해제한다.
Drive mount는 필요하지 않다. `--results-root`로 PC의 저장 위치를 선택한다.
`--resume`는 로컬 checkpoint를 함께 전송한다. GPU 기본값은 T4이며 L4/G4/A100/H100도
명시적으로 선택할 수 있다. 실제 할당 가능 여부는 Colab 계정과 가용 자원에 달려 있다.

노트북에서 실행할 묶음만 만들려면:

```bash
# 작은 연결 검증용 묶음
python3 humanoid.py launch --experiment light_bfm_zero_smoke --runtime colab \
  --bundle-data --output runs/colab/light_bfm_zero_smoke

# 전체 screened CMU/LAFAN을 포함한 본 실험 묶음
python3 humanoid.py launch --experiment light_bfm_zero --runtime colab \
  --bundle-data --output runs/colab/light_bfm_zero
```

출력 폴더 전체를 Google Drive의 `MyDrive/humanoid_rl/colab/`에 복사한 뒤,
그 안의 `train_colab.ipynb`를 Colab에서 열고 GPU runtime을 선택한다.
노트북의 `BUNDLE_DIR`가 업로드한 폴더를 가리키는지 확인하고 셀을 실행한다.
인증은 Colab의 Drive mount 셀에서 진행한다.

묶음 내용:

- `source.zip`: 현재 작업 트리의 소스·필요 asset·선택한 외부 학습 코어. 미커밋 변경도 포함한다.
  venv, Git 설정, `.env`, 기존 artifacts, 모션·모델 가중치는 포함하지 않는다.
- `experiment.json`, `runtime.json`, `bundle.json`: 재현 설정, CLI override, 소스/압축파일 지문.
- `data.zip`: `--bundle-data`를 선택했을 때 선언된 입력만 포함한다.
  Light BFM의 feature cache는 포함하지 않고 Colab에서 재생성한다.
- `resume.zip`: `--resume`를 선택했을 때 해당 checkpoint를 포함한다.
- `inputs.json`: 데이터 루트 기준의 필수 입력 목록.

export 대상의 코드가 Git에 모두 올라가지 않았어도 source.zip으로 현재 소스를 전달한다.
대용량 데이터는 `--bundle-data` 없이 별도로 준비할 수도 있다. 기존 G1 SuperMimic의
carry cache는 상당히 크므로 `inputs.json`의 실제 입력 범위를 확인한다.

노트북은 source/data를 `/content`에 풀고 선택한 방법론의 잠긴 환경을 설치한 뒤 아래와 같은
공통 명령을 호출한다. 학습 중 데이터/replay는 VM에서 사용하며 결과는 Drive로 복사한다.

```bash
python3 humanoid.py train --experiment /path/to/experiment.json --runtime /path/to/runtime.json
```

## 기록과 재개

새 실행은 `runs/<experiment>/<run-id>/`에 다음을 저장한다.

- `launch.json`: 요청한 실험·runtime·최종 method 설정·실제 실행 명령
- `method.json`: 기존 학습기에 전달하는 설정 (G1 SuperMimic은 CLI 인자)
- `environment.json`, `source.json`: 실제 환경 버전·GPU 정보·소스 지문
- `launch_status.json`, `training.log`: 완료/실패/중단 상태와 학습기 출력
- `training/`: 학습기의 원래 체크포인트·metrics·최종 설정

기존 G1 SuperMimic의 생략된 CLI 기본값은 학습기가 `training/run_config.json`에 기록한다.
학습기별 실제 최종 설정과 체크포인트 포맷은 유지한다.

```bash
python3 humanoid.py launch --experiment light_bfm_zero --runtime local \
  --resume /path/to/checkpoints/step_N --set training.total_env_steps=100000000
```

BFM은 저장된 학습 설정을 복원하고 장치·환경 수·저장 간격·학습 예산·데이터 경로를 변경할 수 있다.
모델/물리/보상 변경은 resume의 `--set`으로 허용하지 않는다. 모션 bytes와 manifest가 같으면
데이터 경로를 옮겨도 expert fingerprint가 같다. 평가에도 `--motion-data`, `--motion-manifest`,
`--motion-cache`를 지정할 수 있다.

기존 G1 SuperMimic은 `last.pt` 또는 해당 디렉터리로 재개한다. 공통 실행부는 checkpoint를
새 출력 폴더에 복사하며 기존 실행을 수정하지 않는다. 학습기 자체의 resume 제약은 유지한다.

`persistent_root`가 있으면 기본 60초 간격과 종료 시 결과를 복사한다. 작성 중인 임시 파일은 제외하고,
발행된 checkpoint 디렉터리는 복사 완료 후 공개한다. 영속 저장 복사에 실패하면 로컬 결과를 보존하고
오류를 표시한다. 복사 완료 전의 강제 VM 종료까지 저장을 보장하지는 않는다.
영속 저장소의 과거 checkpoint는 자동 삭제하지 않는다.

검증:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_launchers.py' -v
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python -m unittest discover \
  -s tests -p 'test_humanoid_light_bfm.py' -v
```
