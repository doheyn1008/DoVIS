# Mutual Lab — 언어 상호학습 실험 환경

공유 그림을 보고 자기 언어로 말하는 두 학습자의 **상대 언어 습득 비용을 방향별로 기록**합니다. 모국어 이해 가중치를 재사용하는 누적 신경망 학습, 독립 사전검사, 우연 보정 이해도, 체크포인트 재개, 모델·경험 반복을 지원합니다.

현재는 색·모양·상대 위치를 다루는 **제한된 자연어 영역의 파이프라인**입니다. 유효한 언어 거리나 충분한 모국어 숙련도를 갖춘 본실험 결과를 확보했다는 뜻이 아닙니다. 적격성·발화 정확성·사전검사를 못 넘은 실행도 사유와 함께 남깁니다. [프로토콜과 해석 범위](docs/methodology.md)를 먼저 확인하세요.

## 추가 설치 없이 장치 확인

Python **3.11 이상**에서 API 키·GPU·외부 패키지 없이 실행합니다.

```bash
cd experiments/mutual-understanding
python -m mutual_lab demo --lang-a ko --lang-b es
python -m mutual_lab serve
```

**http://127.0.0.1:8765**에서 실행·곡선·대화 재생·비교·ZIP 내보내기를 볼 수 있습니다. 지원 언어는 영어·한국어·일본어·스페인어입니다. 화면의 실행 버튼은 기호학습 데모만 시작하며 학습 비용이 발생하는 신경망 훈련을 시작하지 않습니다. CLI로 저장한 신경망 실행은 같은 대시보드에서 조회합니다.

| 모드 | 변화하는 것 | 결과의 의미 |
| --- | --- | --- |
| `demo` | 빈 naive Bayes 수신자의 기호 통계 | 모국어 전이가 없는 측정 장치 검사 |
| `endpoint` | 고정 VLM에 전달하는 최근 대화 기록 | 초기 외국어 지식이 미검증인 문맥 적응 파일럿 |
| `neural` | 모국어로 훈련한 이해 GRU의 실제 가중치 | 제한된 과제의 누적 습득. 분석 포함 조건을 별도 판정 |

## 모국어부터 신경망 학습

작은 GRU를 CPU에서 직접 훈련합니다. 공개 다국어 가중치나 학습된 토크나이저를 사용하지 않습니다. 기본 프로파일은 경로 확인용 초소형 모델이며 모국어 적격성 통과나 실행 시간을 보장하지 않습니다.

```bash
# CPU wheel을 먼저 설치하려는 경우
python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[neural]'

python -m mutual_lab validate configs/neural.example.json
python -m mutual_lab train-native configs/neural.example.json --language ko --output checkpoints/ko.pt
python -m mutual_lab train-native configs/neural.example.json --language ja --output checkpoints/ja.pt
python -m mutual_lab run configs/neural.example.json
python -m mutual_lab serve
```

언어별 복제본은 같은 초기화 시드와 구조에서 자기 언어 자료만 배웁니다. 기본 L1 조건은 같은 step·batch 예산입니다. 실제 이해·생성 점수는 서로 다를 수 있어 함께 기록합니다. 상대 언어를 배울 때는 **L1 이해 embedding·GRU·head를 그대로 이어서** 업데이트합니다. 원어 생성용 GRU는 별도이며 만남 뒤 고정합니다. 둘 모두 발화한 후 양쪽을 업데이트합니다.

[neural.example.json](configs/neural.example.json)의 설정은 다음처럼 분리되어 있습니다.

| 설정 | 통제 내용 |
| --- | --- |
| `model_profile` | GRU 구조·용량·입력 단위·최대 길이 |
| `native_training_profile` | L1 자료 버전·훈련 예산·적격성 기준 |
| `acquisition_profile` | 노출당 업데이트·Adam·학습률·자기 언어/외국어 복습 |
| `evaluation_profile` | 목표 척도·사전검사·독립 최종 평가·공통 노출 구간 |

`tiny-gru-v1`과 더 큰 `gru-12m-v1` 예시가 있으며, 유효한 GRU 차원을 직접 지정할 수도 있습니다. 실제 매개변수 수와 모국어 성취도를 기록하며 크기를 지능 점수로 취급하지 않습니다. 지원되지 않는 구조·자료·학습 규칙과 호환되지 않는 체크포인트는 거부합니다.

기본 입력은 UTF-8입니다. `model_profile.tokenizer="unicode-bmp"`는 학습된 vocabulary 없이 고정 BMP 코드포인트를 쓰는 민감도 조건입니다. vocabulary 크기·실제 매개변수 수가 달라지므로 별도 프로파일로 비교하세요. 같은 바이트/토큰 예산을 같은 의미 경험이라고 가정하지 않습니다.

대조 조건은 `acquisition_profile`에서 지정합니다.

```json
{"updates_per_exposure": 0}
```

또는 `association="shuffled"`, `reset_l1_understanding=true`를 사용합니다. 기본값인 `aligned`, L1 가중치 유지, 노출당 1회 업데이트와 구분해 기록합니다. 원어 가중치를 초기화해 초기 점수를 억지로 0에 맞추지 않습니다.

## 독립 모델 × 경험 반복 × 모든 언어 쌍

```bash
# 훈련 없이 전체 작업 목록과 예산 확인
python -m mutual_lab grid-plan configs/grid.example.json

# 기본: 모델 조건 1 × 독립 L1 모델 2 × 경험 반복 2 × 언어 4개의 6쌍 = 24실행
python -m mutual_lab grid configs/grid.example.json

# 파일럿으로 pending 작업 1개만 실행
python -m mutual_lab grid configs/grid.example.json --limit 1

# 같은 계획의 미실행 작업 계속 실행
python -m mutual_lab grid configs/grid.example.json --resume-dir runs/grids/GRID_DIRECTORY
```

실제 생성된 `GRID_DIRECTORY` 경로를 사용하세요. 예산은 모델별 L1 step과 최대 L2 업데이트 수입니다. 시간·GPU 메모리를 임의로 추정하지 않으므로 작은 계획부터 실제 장치에서 측정하세요.

각 독립 모델 반복은 다른 최초 가중치와 L1 표본 순서로 **다시 훈련**합니다. 한 반복 안에서는 언어별 최초 가중치·의미 범위를 대응시킵니다. 각 쌍/경험은 같은 원본 L1 체크포인트를 새로 로드하므로 이전 외국어 학습이 다음 쌍으로 넘어가지 않습니다.

같은 경험 반복의 모든 쌍에 공통 장면·대화·평가 시드를 사용합니다. 일본어를 목표로 모국어만 바꾸는 비교에는 동일한 일본어 생성 체크포인트와 그림·발화 스트림이 들어갑니다. 현재 grid는 장면 분할을 고정하고 경험별 대화 순서·평가 보기 순서를 바꿉니다. 분할까지 바꾸려면 새 L1 모델을 준비한 별도 계획을 만드세요.

`grid.json`과 `outcomes.json`은 미실행·실행 중·완료·실패·취소를 구분합니다. `aggregate.json`은 모델 프로파일과 적격성 분류를 나누고, **독립 모델마다 그 안의 경험 반복을 보존**합니다. 같은 체크포인트에서 대화 100번을 반복한 것을 독립 모델 100개로 세지 않습니다. 실패·취소는 자동 재시도하지 않고, 재개 시 미실행 작업을 진행합니다.

기존 `sweep`은 데모/endpoint 시드 반복용이며 신경망 독립 모델 반복을 대신하지 않습니다.

```bash
python -m mutual_lab sweep configs/demo.json --seeds 7,11,23
```

## 점수와 분석 포함 기준

원정답률 p와 우연 보정값 `q=(p−0.25)/0.75`를 함께 저장합니다. 25%가 보정 0, 100%가 보정 1이며 음수도 그대로 남깁니다.

- 기존 설정의 90% 목표는 원정답률입니다. 조용히 재해석하지 않습니다.
- `threshold_scale="corrected"`의 90%는 원정답률 **92.5%**입니다. 신경망 예시는 이 조건을 명시합니다.
- 화면의 곡선은 원정답률이고 보정값은 별도 열에 표시합니다. 목표 척도도 라벨에 적습니다.
- `A → B`는 **A 발화를 B 화자가 이해**한다는 뜻입니다. 결과의 `learner_native_language`, `target_language`, `speaker_language`로 방향을 명확히 합니다.
- 노출 1회는 공유 장면 1장과 각자 받은 상대 발화 1개입니다. 문자·바이트·입력 unit·복습·업데이트·시간·계산량 대용치도 별도 저장합니다.

신경망 모드에는 학습·L1 검사·사전검사·곡선·최종 평가가 물체 쌍 단위로 분리됩니다. 초기 점수가 우연과 유의하게 다르지 않다는 이유만으로 0이라고 하지 않습니다. 사전에 정한 우연 주변 구간 안에 **쌍 단위의 보수적 불확실성 구간 전체**가 들어와야 1차 포함 조건을 충족합니다. 현재 36개 물체 쌍의 작은 세계는 이 검정력이 부족해 우연 수준의 동등성을 확정할 수 없습니다. 표본 확대가 필요한 파일럿이며 판정 보류를 정상 결과로 취급합니다.

`summary.json`의 `inclusion.analysis_group`은 `primary`, `positive_baseline_extension`, `uncertain`, `ineligible`, `control`을 구분합니다. 모국어/출처/고정 발화 검사 실패, 정밀도 부족과 대조 조건을 이유와 함께 남깁니다. 문항을 사후 제거해 점수를 맞추지 않습니다.

목표는 0회부터 동일하게 검사하며 최초 통과·연속 통과 시작·확인 시점을 따로 남깁니다. 완료 상한까지 미도달이면 도달량은 `null`이고 `right_censored`입니다. 실패·취소와 구분합니다. 도달 실행만의 중앙값에는 미도달 비율도 함께 표시합니다.

조기 종료한 실행의 `normalized_auc_observed`는 실행별 실제 관측 구간의 기술적 평균입니다. 동일 노출 비교에는 사전등록한 `common_horizon`의 AUC와 고정 노출 정확도를 사용합니다. 구간을 끝까지 못 봤으면 결측이며 외삽하지 않습니다. 신경망 예시는 곡선 비교를 위해 조기 종료를 끕니다.

## 체크포인트와 실행 재개

```bash
python -m mutual_lab resume runs/RUN_DIRECTORY
```

두 학습자가 모두 저장된 마지막 완전한 체크포인트에서 **새 실행**을 시작합니다. 원래 고정 문장·시드·분할·학습 계획과 평가 이력을 보존하며 원래 실행도 남습니다. 가중치·optimizer·복습 버퍼·난수 상태·노출 카운터를 복원하므로 저장 지점 뒤의 학습을 이어갑니다. 저장 지점 뒤 미완료 작업은 재실행합니다.

첫 접촉을 새로 측정하는 `run`의 `native_checkpoint`는 **`kind="native"`만** 받습니다. 이미 만남 후 학습한 `kind="acquisition"` 파일은 텐서 로딩·실행 폴더 생성 전에 거부합니다. 이 파일은 명시적인 `restore`/`resume`에서만 복원하며, 반대로 원본 native 파일을 resume 상태로 사용할 수 없습니다.

체크포인트 sidecar의 `metadata`는 원본 L1 훈련의 출처입니다. 현재 상태의 종류·상대 발화 노출·업데이트·외국어 복습 버퍼 수는 별도 `provenance`에 저장합니다. 따라서 원본 metadata의 빈 외국어 훈련 목록을 현재 상태까지 외국어 경험이 없다는 뜻으로 해석하지 않습니다. 기존 sidecar에 `provenance`가 없어도 체크포인트의 실제 카운터에서 재개 정보를 복원합니다.

| 결과 파일 | 내용 |
| --- | --- |
| `summary.json` | 실행 상태·모국어 적격성·포함 사유·방향별 비용·노출/업데이트/시간 |
| `metrics.csv`, `evaluations.json` | 원정답률·보정값·문항별 선택·Wilson 구간·방향 매핑 |
| `pretest.json`, `final_evaluations.json` | 별도 사전검사와 독립 최종 평가 |
| `native_qualification.json`, `caption_audit.json` | 모국어 이해·생성 성취도, 고정 발화 감사 |
| `config.json`, `manifest.json`, `prompts.json` | 실제 프로파일·코드/설정 해시·언어 준비 상태·환경 |
| `dataset.json`, `all_frozen_captions.json`, `dialogue_plan.json` | 고정 분할·평가 원어 발화·시드별 노출 계획 |
| `native/` | 원본 L1 체크포인트와 SHA256 sidecar 사본 |
| `checkpoints/`, `learning_checkpoint.json` | 실험 중 두 학습자의 누적 상태와 재개 포인터 |
| `events.jsonl`, `assets/` | 발화·학습 그림·평가·실패 로그와 이미지 |

`neural`은 원본 L1과 누적 학습 체크포인트를 구분합니다. 대시보드 ZIP에는 학습 상태도 포함하므로 큰 모델은 내보내기 파일이 커집니다. 데모/endpoint에는 persistent 체크포인트가 없습니다. 기존 파일명 `frozen_captions.json`도 곡선용 호환 파일로 유지합니다.

## 로컬 VLM의 문맥 적응 측정

이미지를 받는 Chat Completions 호환 서버 두 개를 별도로 실행하고 [local.example.json](configs/local.example.json)을 `configs/local.private.json`으로 복사해 수정하세요.

```bash
python -m mutual_lab validate configs/local.private.json
python -m mutual_lab run configs/local.private.json
```

서버는 `POST /v1/chat/completions`의 `model`, `messages`, `temperature`, `max_tokens`, `seed`, `stream=false`와 PNG data URL을 받아 문자열 `message.content`를 반환해야 합니다. 예시 요청 상한은 272회, 예산은 400회입니다. 생성 실패도 서버가 유효하게 보고한 토큰 사용량은 집계하고 자동 재시도하지 않습니다. 인증 키는 환경변수에 두고 설정에는 변수 이름만 적습니다.

한 요청에 기록 그림 `context_exchanges`장과 현재 그림 1장이 들어갑니다. 모델 이미지 수·컨텍스트 길이 제한을 확인하세요. 호출 사이 숨은 상태나 온라인 학습이 없어야 하며 실제 서버의 비결정성까지 통제하지는 않습니다. 기본 loopback만 허용하고 외부 서버는 `allow_remote=true`를 명시합니다. 클라이언트는 리다이렉트·환경 프록시·웹/번역 도구를 사용하지 않습니다.

"자기 언어로만 말하라"는 지시나 인터넷 차단은 다국어 사전학습 지식을 지우지 않습니다. `initialization_note`에 근거를 적더라도 이 모드는 단일언어 지속 학습 본실험으로 자동 승격되지 않습니다.

## 자동 검증

```bash
# 외부 의존성이 없으면 신경망 검사만 명시적으로 skip
python -m unittest discover -s tests -v
node --test tests/test_dashboard_state.mjs

# 선택적 실제 브라우저 검사
python -m pip install -e '.[browser]'
python -m playwright install chromium
python tests/browser_smoke.py --artifacts-dir browser-artifacts
```

CPU 신경망 CI는 실제로 작은 L1 모델을 훈련한 뒤 L1 가중치 재사용·역전파 경로, 원어 생성기 고정, 업데이트 0회/초기화 대조, 스페인어 경로, 평가 상태 불변, 버퍼/optimizer/난수 복원, 중단/재개 동일성, 쌍 간 초기화와 grid 재개를 확인합니다. 특정 모국어가 반드시 빠르게 학습되도록 만든 검사는 없습니다. 짧은 훈련으로 숙련도 미달인 모델은 부적격으로 남는지도 검사합니다. 모델 반복이 충분한 연구 결과를 얻었다는 검사는 아닙니다.
