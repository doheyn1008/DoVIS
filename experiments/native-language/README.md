# 실제 단일언어 모델의 다음 단계 준비

이 디렉터리는 한국어·일본어·영어·스페인어를 **무작위 초기화부터 자기 언어 자료만으로 학습하는 모델**의 독립 준비 공간입니다. 기존 `mutual-understanding`의 모델·자료·계획·보호 평가를 변경하거나 가져오지 않습니다. 현재 G와 새 시드 cohort가 통과하더라도, 그것은 5색·3도형·공간 관계·작성된 문형이라는 유한 세계의 결과입니다. 일상 대화와 자유 주제의 모국어 능력을 완성했다는 뜻이 아닙니다.

현재 준비 범위는 scratch causal Transformer, 검토된 단일언어 manifest의 로더·명시적 학습 CLI, 작은 모델의 저장·복원 검사와 합성 입력 CPU 자원 측정입니다. 직접 작성한 검사 문장으로만 2~4회 업데이트를 검증했으며 실제 코퍼스 수집·장시간 언어 학습·외국어 습득 실험은 시작하지 않았습니다. 공개 사전학습 가중치, 학습된 다국어 토크나이저, 다른 실험의 언어 체크포인트를 사용하지 않습니다.

## 고정한 첫 모델 프로파일

| 항목 | 준비 프로파일 |
|---|---|
| 입력 | 고정 UTF-8 바이트 vocabulary 259개 |
| ID | PAD=0, BOS=1, EOS=2, 바이트 0…255는 ID 3…258 |
| 구조 | decoder-only causal Transformer, pre-LayerNorm, GELU |
| 크기 | hidden 384, 8층, attention head 6개, FFN 1,536 |
| 문맥 | 최대 입력 256 unit, 학습 가능한 위치 embedding |
| 출력 | 입력 embedding과 출력 가중치 공유 |
| 정밀도·dropout | 첫 CPU 측정은 float32, dropout 0 |
| 매개변수 수 | 고유 학습 매개변수 **14,394,240개**, 출력 가중치 공유를 중복 집계하지 않음 |

다음 바이트 예측에는 미래 입력을 차단하는 causal mask와 PAD 처리가 필요합니다. 정확성 검사는 작은 모델에서 미래 token 변경의 영향, 유효 위치와 PAD의 분리, 같은 seed 초기화의 재현성, 다른 언어 문서 및 미검토 문서의 입력 거부를 확인합니다. 이 검사는 언어 습득 성능 검사가 아닙니다.

같은 초기값·구조·매개변수 수는 **학습 뒤 능력이나 의미 경험의 동일성**을 보장하지 않습니다. UTF-8에서 한글·일본어의 시퀀스는 ASCII 중심 텍스트보다 길어질 수 있습니다. 바이트 예산·문서 수·문장 수·의미 경험·계산량을 동시에 같게 만들었다고 해석하지 않습니다. 언어별 subword tokenizer를 조사한다면 별도의 사전등록 민감도 조건으로 두고, 이 바이트 기준선과 조용히 합치지 않습니다.

## 실제 자료를 받기 전에 확정할 프로토콜

각 학습자는 선택한 한 언어의 문서만 받습니다. 자료 분할·라이선스·언어 감사의 승인 상태를 확인한 뒤에만 학습용 window를 만들며, benchmark의 임의 token을 실제 코퍼스에 섞지 않습니다.

| Manifest에 남길 내용 | 확인 목적 |
|---|---|
| 문서 ID, 언어, 원본 URL, 제공자 | 자료의 출처와 학습자별 언어 경계 |
| dump·문서 revision, 수집 시각, 원본 SHA256 | 같은 자료의 재현 |
| 개별 자료의 라이선스, 이용 조건, 권리 검토 상태, attribution | 공개 접근 가능성과 재사용 권리를 구분 |
| 정제 코드·설정·버전, 정제 뒤 텍스트 SHA256 | 전처리의 추적 |
| 장르·주제·길이, 제외 사유 | 언어 간 자료 구성 차이의 기록 |
| 원문 문자·UTF-8 바이트·모델 unit 수 | 서로 다른 노출 단위의 보존 |
| 문서 계열·정확 중복·근접 중복 그룹, 분할 | 문서·수정본·유사 문서 누출 방지 |
| 외국어 구절·번역 표·혼합 언어 감사 결과 | 언어판 이름만으로 단일언어성을 가정하지 않기 |

Wikipedia 본문은 네 언어의 공통 출발점 후보입니다. 다만 Wikimedia의 일반적인 CC BY-SA 4.0 텍스트 조건만 보고 모든 페이지·인용·외부 반입 자료의 권리를 일괄 승인하지 않습니다. 예외와 추가 attribution, 수정 사실 및 라이선스 고지를 원본 단위로 보존해야 합니다. 이용 조건이 불명확하거나 별도 허가가 필요한 자료는 승인 전 학습 풀에서 제외합니다. 이 문서는 법적 권리를 자동 판정하는 도구가 아닙니다.

언어별 Wikipedia를 골랐다는 사실도 외국어 노출 부재의 증명은 아닙니다. 외국어 문단·병렬 번역·대역 표·코드·로마자 표기 등을 감사하고 정책과 표본 검수 결과를 기록합니다. 자연스러운 차용어나 고유명사를 사후 제거하여 초기 상호이해도를 억지로 0에 맞추지 않습니다. 전처리에 언어 식별 도구를 쓴다면 그 출처와 한계도 남기며, 해당 도구의 가중치나 내부 표현을 학습자에게 전달하지 않습니다.

분할은 window나 문장으로 잘라 무작위 배분하기 **전에** 문서·revision 계열·중복 그룹 단위로 결정합니다. 근접 중복 탐지 알고리즘·정규화·임계값·seed를 고정하고 train/dev/보호 test 사이의 그룹 교집합을 검사합니다. 공개 문법 평가 문장과 그 변형의 오염도 별도로 감사합니다. 언어 간 주제·장르 비율은 가능한 한 대응시키되, 서로 다른 원문을 의미적으로 같은 경험이라고 부르지 않습니다. 최종 평가를 반복적으로 보고 자료·모델을 고르지 않습니다.

현재 `NativeDocument` 검사는 제공된 검토 상태·언어·텍스트 해시를 확인할 뿐 실제 권리나 문장 언어를 독립 판정하지 않습니다. `corpus_loader.py`는 모든 JSONL 행과 파일 해시를 먼저 검사하고, 이미 검토한 중복 그룹으로 train/dev/test를 나누며 ID·그룹·정확 텍스트 해시의 분할 교집합을 거부합니다. 근접 중복 탐지 자체는 아직 외부 검토 결과에 의존합니다. `train_native.py`는 반환된 train 문서의 window만 표집하고 dev는 평가에만 사용하며 test는 토큰화하지 않습니다. 코퍼스 수집 기능은 제공하지 않습니다.

## 학습 예산과 통제

실제 코퍼스를 고른 뒤 문서·문자·바이트 규모를 단계적으로 늘리는 작은 계획부터 별도로 고정합니다. 한 번의 관측 결과를 보고 실패한 seed만 재시도하거나 예산·기준을 늘리지 않습니다. 학습 기록에는 문서/window 수, 고유 문서·중복 그룹 수, 문자·바이트·unit 노출, 반복 epoch, optimizer step, batch·문맥 길이, 처리량·경과 시간·메모리를 함께 남깁니다.

동일 학습 예산 비교와 능력 기준을 충족하도록 학습하는 비교는 다른 추정 대상입니다. 첫 조건을 선택하면 언어별 실제 성취도의 차이를 보고하고 부적격 실행도 보존합니다. 성취도를 맞추려면 목표·최대 예산·정지 규칙을 먼저 정한 별도 조건이 필요합니다. 모델 크기를 지능 점수로 간주하지 않습니다.

이 단계에서도 초기화·학습 표본 순서를 언어 간 대응시키고, 반복 간에는 새 언어 가중치와 새 optimizer에서 시작합니다. 자료가 바뀌면 기존 유한 세계의 개발 점수나 선택 결과가 새 영역에 그대로 유효하다고 가정하지 않습니다.

## L2 이전에 필요한 평가 확대

실제 모국어 모델의 완료 기준은 다음을 포함하도록 별도로 정해야 합니다. 현재의 네 문형 exact match나 낮은 next-byte loss 하나로 대체하지 않습니다.

- 학습 문서·근접 중복과 분리된 원문에서의 next-byte loss와 bits-per-byte. 언어 간 코퍼스가 다르므로 이 숫자만으로 언어 난이도나 능력을 순위화하지 않습니다.
- 원어민 또는 신뢰할 수 있는 언어학적 근거로 검토한 문법 최소대립쌍. 조사·격·어순·일치·부정·수량·시제·관계절·긴 의존 관계 등 각 언어에 실제로 존재하는 현상을 따로 보고합니다. 영어 문제를 기계적으로 번역해 동일한 문법 시험이라고 부르지 않습니다.
- 새 어휘 조합뿐 아니라 보지 않은 문장 구조와 더 긴 조합의 이해. 세 개 이상 대상, 여러 관계, 양화·부정, 대명사·상호참조, 새로운 배치·대상 수를 분리합니다.
- 자기 언어의 다양한 바꿔쓰기와 자유 생성. 올바른 종료·인코딩, 의미의 대상·역할·속성·관계 보존을 함께 평가하고, 제한된 정답 표현에 없는 문장을 곧바로 의미 오류로 처리하지 않습니다.
- 언어와 시각 grounding을 연결하는 독립 과제. 평가 그림·의미·문장은 시각 prior 학습과 언어 튜닝에도 사용하지 않으며, 현재 보호 자료를 새 개발 자료로 재활용하지 않습니다.

그 이후에도 새 L1 적격성 검사·독립 시드 검증·외국어 훈련 이력 감사와 첫 접촉 사전검사가 필요합니다. L2 주 노출량은 공유 의미 경험으로 기록하고 바이트·문자·unit·업데이트·복습량·계산량을 별도로 남깁니다. 단일언어 모델의 구조가 같아도 초기 상호이해도가 같다고 가정하지 않습니다. 양수 baseline은 확장 조건으로 보존합니다.

## CPU 실행 가능성 측정

```bash
cd experiments/native-language
/tmp/dovis-cpu-env/bin/python -m unittest discover -s tests -v
/tmp/dovis-cpu-env/bin/python benchmark_cpu.py --steps 10 --batch-size 1 --threads 1 --output results/cpu-smoke.json
```

실행 환경의 PyTorch와 표준 라이브러리를 사용합니다. 위 절대 경로는 이번 준비에 사용한 CPU 환경이며, 다른 환경에서는 PyTorch가 설치된 Python으로 대체합니다.

`results/cpu-smoke.json`은 **`SMOKE_NOT_LANGUAGE_LEARNING`** 자원 측정입니다. 임의 합성 바이트 ID로 실제 forward/backward와 Adam 계열 업데이트를 실행하며, 구조·실제 매개변수 수·optimizer·문맥·batch·thread·시간·처리량·RSS를 기록합니다. 임의 가중치의 `.pt` 체크포인트를 보관하지 않습니다. 이 손실·처리량을 실제 언어 학습이나 유창성 성과로 읽지 않습니다.

RSS는 CPU 프로세스 메모리이며 GPU VRAM 측정이 아닙니다. 현재 GPU에서 실행하지 않았고 필요한 VRAM이나 GPU 처리량을 실측했다고 주장하지 않습니다. 실제 코퍼스를 여러 언어·seed로 학습하기 전 선택한 장치에서 warmup을 포함한 처리량과 최대 메모리를 먼저 측정하고, 그 결과로 별도의 자원·시간 상한을 정합니다.

이번 실제 측정은 PyTorch `2.15.0+cpu`, Xeon Platinum 8370C, float32, batch1, 입력256, intra-op1 thread에서 실시했습니다. 두 warmup 업데이트 뒤10회 업데이트의2,560개 예측 unit을 **3.392초, 약754.8 unit/초**에 처리했고, 프로세스 최대 RSS는 **598,196KiB(약584.2MiB)**였습니다. 측정은 forward·loss·backward·gradient clipping·AdamW를 포함합니다. optimizer tensor는115,154,320byte이고, 가중치 tensor는57,576,960byte입니다. 이 하드웨어에서의 짧은 합성 입력 측정이며 실제 코퍼스 I/O·장문 처리·GPU 성능으로 외삽하지 않습니다. 원본 설정·환경·소스 해시와 모든 손실은 JSON에 있습니다.

같은 소스의 추가 batch4·intra-op4 측정은 `results/cpu-b4-t4-smoke.json`에 있습니다. 20회 업데이트의20,480개 예측 unit에11.552초, 약1,772.8unit/초, 최대 RSS744,104KiB였습니다. 두 측정은 batch와 thread가 함께 다르므로 순수 thread 효과로 해석하지 않습니다. 실제 코퍼스나 GPU의 처리량을 보장하지 않습니다.

별도 `native-language-ci.yml`은 CPU torch `2.15.0+cpu`와 NumPy `2.5.4`를 고정하고 작은 모델·로더·명시적 학습 CLI의 회귀 검사, `smoke_small.py --steps 2`, 학습 CLI 도움말만 실행합니다. 학습 검사는 직접 작성한 작은 임시 fixture로2~4회만 업데이트하고 체크포인트를 삭제합니다. 매 push마다 위14.4M benchmark나 실제 코퍼스 학습을 반복하지 않으며 체크포인트·artifact를 업로드하지 않습니다.

## 검토된 단일언어 manifest로 실행

Manifest는 정확히 다음 여섯 필드를 갖습니다. 다음은 **형식 설명용 가상 예시**이며 실제 코퍼스·권리 승인·유효한 SHA를 제공하는 예제가 아닙니다. `document_file`은 manifest 옆의 일반 파일 basename이어야 합니다.

```json
{
  "schema_version": 1,
  "kind": "native-monolingual-corpus-v1",
  "language": "ko",
  "document_file": "documents.jsonl",
  "document_file_sha256": "<실제 JSONL 원문 바이트의 SHA256 64자리>",
  "split_seed": 79003
}
```

JSONL 각 행은 아래 필드를 정확히 갖습니다. 실제 검토가 끝난 자료만 `rights_review="approved"`, `foreign_passage_audit="passed"`로 변경할 수 있습니다. 라이선스 표기만으로 승인 상태를 만들지 않습니다. 아래 가상 문장은 현재 학습 입력으로 허용되지 않습니다.

```json
{"document_id":"fictional-ko-001","language":"ko","text":"형식 설명을 위한 가상 문장이다.","source_url":"https://example.invalid/not-a-corpus","revision":"fictional-v1","license_id":"CC-BY-SA-4.0","attribution":"실제 출처와 저자 정보를 검토해 기입","text_sha256":"<정제된 실제 text UTF-8 SHA256>","near_duplicate_group":"reviewed-group-001","rights_review":"pending","foreign_passage_audit":"pending"}
```

로더는 train/dev가 비어 있으면 실패하고 test는 별도로 반환합니다. 다른 언어의 마지막 행, 미검토 행, 중복 ID, 일치하지 않는 해시, 모순된 정확 중복 그룹을 조용히 제거하지 않습니다. 전체 입력을 검증한 다음에만 모델과 train/dev 토큰 window를 구성합니다. window는 문서 경계를 넘겨 packing하지 않습니다.

다음 명령은 별도로 검토·고정한 실제 manifest가 준비된 뒤에만 명시적으로 실행합니다. 기본 모델은14,394,240개 매개변수이며 CPU가 기본입니다. `--profile`에 `ModelConfig`의 JSON 필드를 주면 검사·별도 사전등록 프로파일을 지정할 수 있습니다. `--device cuda:0`은 해당 장치가 명시적으로 제공된 경우에만 선택하며, 현재 GPU 검증이나 VRAM 측정 결과는 없습니다.

```bash
python train_native.py --manifest reviewed/ko/manifest.json --language ko \
  --steps 100 --batch-size 1 --eval-every 25 --max-seconds 600 \
  --initialization-seed 79007 --data-seed 79009 --threads 1 --device cpu \
  --output runs/native-scratch
```

모델은 새 무작위 가중치와 새 AdamW에서 시작합니다. train window의 모든 선택, 문서/window별 횟수, 선택 스트림 해시, 바이트·EOS·PAD·전체 예측 unit 수를 기록합니다. `max-seconds`는 준비 시간을 포함하며 초과·실패·중단을 완료로 표시하지 않습니다. manifest 로딩의 행 처리·검증, 문서 인코딩 전후와 각 window, dev의 각 배치 전후에도 같은 deadline을 확인합니다. 이는 **협력적 시간 제한**이므로 진행 중인 단일 파일 I/O·인코딩·Torch 연산을 중간에 강제 종료하지 못합니다. 해당 연산 뒤 다음 경계에서 중단하며, 최소 실패 기록·체크포인트 저장은 마감 뒤에도 시도할 수 있습니다. 검증이 끝나기 전에 준비 시간이 초과하면 학습 없이 setup 단계의 time_limit 보고서만 남깁니다.

개발 평가는 시작점과 고정 간격·고정 최종 step에서만 수행하고 가장 좋은 dev checkpoint를 고르지 않습니다. dev 평가 전후 모델·optimizer·표집 RNG·Torch RNG가 같은지 검사합니다. 마지막 업데이트 직후 평가 전에 중단된 안전한 상태를 재개하면, 빠진 그 최종 평가를 먼저 정확히 한 번 수행한 뒤 완료를 표시합니다. 현재 step이 사전 계획한 평가 시점이 아니면 재개를 이유로 새 평가점을 추가하지 않습니다.

`loss_per_target_unit`과 `bits_per_prediction_unit`은 byte와 EOS 예측을 포함하고 PAD를 제외합니다. `bits_per_byte`는 **target ID≥3인 실제 바이트의 NLL만** 합산하고 바이트 수로 나누며 EOS/BOS/PAD를 제외합니다. 두 지표를 동의어로 쓰지 않고 EOS·PAD 수를 별도로 남깁니다. 어느 값도 L1 적격성·일반 유창성·언어 간 난이도 판정을 자동으로 만들지 않습니다.

체크포인트에는 전체 모델·AdamW·로컬 표집 RNG·Torch RNG·기존 선택 스트림·dev 이력을 저장하며, sidecar에 SHA256·모델·소스·코퍼스·분할 정보를 남깁니다. 저장할 때마다 `step-XXXXXX-STATUS-UUID.pt`라는 새 파일 이름을 사용하며, 같은 step의 running/completed/failed 상태도 이전 tensor·sidecar 쌍을 덮어쓰지 않습니다. tensor 공개 뒤 새 sidecar 저장이 실패해도 이전의 유효한 쌍은 그대로 남습니다. 보고서의 `checkpoints[].path`가 성공적으로 저장된 쌍을 가리킵니다.

모든 파일은 `development_only`, `excluded_from_research=true`, `eligible_for_l2=false`, `eligible_for_fresh_experiment=false`입니다. 복원은 `weights_only=True`와 정확한 소스·코퍼스·프로파일·계획 검사를 사용합니다. 새 실행 디렉터리에서 같은 계획으로만 `--resume <step-XXXXXX-STATUS-UUID.pt>`를 지정합니다.

완료된 step 사이의 중단은 정확히 재개할 수 있도록 검사했습니다. 업데이트 도중 중단된 live state는 `incomplete-after-XXXXXX-STATUS-UUID.pt`, `resume_safe=false`로 별도 보존하며, 마지막 안전한 `step-*.pt`를 덮어쓰지 않고 그 불완전 상태의 재개를 거부합니다. 이 도구와 작은 fixture의 성공은 일반 모국어 모델을 학습했다는 결과가 아닙니다.

## 확인한 1차 자료

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762): Transformer 구조의 원 논문. 이 준비 코드를 원 논문 성능의 재현으로 주장하지 않습니다.
- [ByT5](https://aclanthology.org/2022.tacl-1.17/): 바이트 입력과 시퀀스 길이·계산량의 trade-off 참고. 공개 다국어 가중치를 가져오는 근거가 아닙니다.
- [PyTorch scaled dot product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html): attention mask와 해당 API의 공식 명세.
- [Wikimedia 이용 조건](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use/en): 텍스트의 라이선스·attribution·외부 자료 예외와 재사용 조건.
- [JBLiMP](https://aclanthology.org/2023.findings-eacl.117/): 일본어의 현상별 문법 최소대립쌍 평가 사례. 다른 언어의 검증을 대신하지 않습니다.
- [BabyLM 2023](https://babylm.github.io/archive_2023.html): 제한된 단일언어 자료 예산의 scratch 사전학습·평가 설계 참고. 해당 영어 자료량을 네 언어의 의미적으로 동등한 예산이나 인간 수준 능력의 기준으로 그대로 쓰지 않습니다.
