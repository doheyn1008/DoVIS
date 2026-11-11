# 개발용 학습 진단과 표본 설계

이 도구는 본실험 준비를 위한 진단이다. 보고서의 `excluded_from_research=true`는 결과를 본실험의 모국어 적격성이나 언어 습득 비용으로 쓰지 않는다는 뜻이다. 기존 세계·모델 구조·학습 규칙·첫 접촉 판정·모델 생성 발화 평가를 유지한다.

## 작은 자료의 학습 경로 확인

```bash
python -m mutual_lab diagnose-native configs/neural.example.json --language es \
  --steps 600 --eval-every 100 --train-pairs 2 --dev-pairs 2
```

`model_profile`과 `native_training_profile`의 구조·토큰화·batch size·Adam 학습률을 사용한다. 진단의 실제 훈련 step은 `--steps`이며, 원본 프로파일의 L1 훈련 step을 바꾸지 않는다. 이해/생성 loss를 1:1로 더하고 joint gradient norm을 1로 clip하는 기존 native 훈련 절차를 사용한다. 실제 예산은 보고서의 `effective_training_budget`에 기록한다.

기본 개발 시드는 자료 19003, 초기화 19007, 훈련 표본 순서 19009다. `--data-seed`, `--initialization-seed`, `--training-seed`로 바꿀 수 있다. 서로와 실험 설정의 시드가 같으면 거부한다. 미래 독립 모델 반복에서도 이 개발 시드를 재사용하지 않아야 한다. 임의 외부 모국어 체크포인트의 초기 시드와의 충돌까지 자동 감사하는 것은 아니다.

자료는 `make_research_dataset(config).train` 안에서만 선택한다. 기본 train/dev는 각각 물체 쌍 2개 × 4변형 = 8장면이며, 각 장면의 자기 언어 4문형을 사용한다. train/dev 물체 쌍은 겹치지 않는다. 기존 L1 적격성·첫 접촉 사전검사·곡선·최종 평가의 문항/고정 발화를 진단 점수에 사용하거나 학습하지 않는다.

작은 무작위 자료에는 어휘가 빠질 수 있다. `feature_coverage`, `unseen_dev_features`를 먼저 확인한다. 예를 들어 train에 초록색이 없고 dev에 초록색이 있다면 dev 점수는 새 어휘 문제도 포함한다. 그 결과를 순수한 조합 일반화의 실패로 단정하지 않는다. 이 도구는 점수가 잘 나오도록 dev를 사후 선별하지 않는다.

0 step부터 설정 간격과 마지막 step에 아래를 기록한다. 진단 평가 전후 가중치·optimizer·훈련 RNG가 바뀌면 실패 처리한다. 평가 간격을 바꾸더라도 최종 훈련 가중치가 같아야 한다.

| 항목 | 의미 |
| --- | --- |
| `comprehension_loss`, `comprehension_accuracy` | 네 그림을 고르는 이해 과제의 loss와 정확도. 모든 자기 언어 문형 검사 |
| `production_loss` | 정답 앞부분을 제공한 teacher forcing 토큰 loss. 자유 생성 성공률과 다름 |
| `teacher_forced_token_accuracy` | PAD를 제외한 다음 토큰 정확도. EOS 포함 |
| `generation_exact_accuracy` | 각 그림에서 greedy 자유 생성한 문장이 허용 원어 문형과 정확히 일치하는 비율 |
| `semantic_success_lower_bound` | 유한 문형 감사기가 의미를 맞는 것으로 확인한 비율. 일반적인 의미 정확도 추정치가 아님 |
| `semantic_unknown_count` | 감사기로 해석할 수 없는 문장 수. 의미 오답으로 자동 확정하지 않음 |
| `generation_error_counts` | 알려진 색/모양/물체/관계/역할 오류, 표면형 차이, 반복 heuristic, 미종결·특수 토큰·인코딩 오류 |

의미 감사기는 연구자 쪽에서 현재 세계의 네 원어 문형을 해석한다. 대소문자·Unicode NFC·여분 공백·문장 끝 구두점만 정규화하며 파트너 번역문이나 의미 레이블을 학습자에 전달하지 않는다. 모르는 바꿔 말하기는 `null`로 남긴다. 내용 오류 플래그는 겹칠 수 있으므로 합이 문장 수보다 클 수 있다.

생성의 실제 토큰 ID, EOS 도달 여부, 무시된 잘못된 UTF-8/BMP 출력도 보존한다. 기존 decoder의 문자열 반환/greedy 선택은 유지한다. 시각 진단은 언어 없이 네 변형의 고정 픽셀 투영 거리·충돌·자기 그림 검색을 확인한다. 서로 다른 그림이 붕괴하지 않았다는 점검이며 의미적인 시각 이해나 학습 성공의 증거가 아니다.

출력:

- `report.json`: 프로파일, 실제 예산, 개발 시드·자료·가중치 해시, train/dev 곡선, 시각 진단, 상태·해석 제한.
- `learning_curve.csv`: step별 두 loss와 이해/생성 점수.
- `generations.json`: 각 체크포인트의 전체 작은 train/dev 생성 문장·원본 토큰·오류 예시.
- `dataset.json`: 실제 사용한 개발 train/dev 자료. 본실험의 보호 평가 자료는 포함하지 않음.

이 보고서는 main dashboard의 연구 실행으로 집계되지 않는다. 적격 L1 또는 acquisition 체크포인트를 저장하지 않는다. 실패/중단 시 이미 계산한 곡선과 상태를 보존한다. `full_small_train_fit_observed`는 마지막 작은 train 자료의 이해·정확 생성이 모두 100%이고 출력 오류가 없었는지만 나타내며 성인 모국어 숙련도나 일반화를 보증하지 않는다. 해당 조건에 도달해도 훈련을 자동 종료하거나 예산을 바꾸지 않는다.

## 사전검사 포함 확률 시뮬레이션

```bash
python -m mutual_lab plan-screening configs/screening-plan.example.json
```

추가 패키지 없이 실행한다. 기존 `pair-mean-hoeffding-family-v1` 구간·분류 함수를 실제 사전검사와 공유한다. 기본은 원정답률 0.25±0.15, 후보 98/200/320쌍, 사례별 Monte Carlo 2000회다. 실제 표본 수·허용 폭·세계·설정 상한은 바꾸지 않는다.

시뮬레이션 단위는 **한 물체 쌍에서 얻은 네 문항의 정답 수 0~4**다. 쌍들 간에는 iid를 가정하며, 본실험 모델 반복을 2000개 돌렸다는 뜻이 아니다. 결과는 L1 적격성·출처·발화 감사 등 다른 조건을 통과했다고 가정한 **사전검사 구간의 동시 포함 확률**이다.

시나리오 종류:

1. `binomial_shared`: 방향별 평균 `means=[p_a,p_b]`, 쌍 안 상관 `within_pair_correlation=rho`. 확률 `1-rho`로 네 독립 Bernoulli, 확률 `rho`로 네 문항이 같은 Bernoulli 결과를 공유하는 혼합이다. 평균을 보존하며 0<p<1에서 문항 ICC가 정확히 rho다. rho는 0~1 범위다.
2. `custom_hits`: `hits_a`, `hits_b`에 0~4개 정답의 확률 다섯 개를 직접 넣는다. 음의 ICC나 항상 한 문항만 맞히는 분포도 표현할 수 있다. ICC는 정답 수가 주어졌을 때 정답 위치가 균등한 교환 가능한 문항들을 가정한 유도값이다. 이때 평균/ICC는 입력 확률에서 계산하며 `means`/`within_pair_correlation`을 중복 지정하면 거부한다.

두 방향의 관계는 `direction_shared_probability`로 정한다. 0이면 독립 cluster count, 1이면 같은 uniform 수로 두 주변분포의 quantile을 선택한다. 두 분포가 같고 1이면 두 정답 수가 같다. **이 값 자체가 Pearson 상관은 아니다.** 실제 유도된 count 상관은 보고서의 `implied_direction_count_correlation`에 별도로 계산하며 분산이 0이면 `null`이다.

기본 예시는 우연 평균, 양의 문항 상관, 양 방향 공유, 약한 전이, 허용 범위 위 평균, 항상 한 문항 정답인 음의 상관 사례를 포함한다. 가정한 분포가 실제 모델의 분포라는 증거는 없다. 필요하면 파일럿에서 추정한 분포나 추가 보수적 민감도 사례를 계획에 넣는다.

`report.json`과 `probabilities.csv`에는 각 방향/양 방향의 포함 확률, 분류 횟수, Monte Carlo 오차의 Wilson 구간과 표준오차를 기록한다. 이 구간은 실제 모델 간 불확실성이나 본실험 결과의 구간이 아니다. 시나리오별 RNG를 분리해 다른 후보를 추가/재정렬해도 기존 사례의 결과를 바꾸지 않는다. 시뮬레이션 표본을 통과할 때까지 자동 추가하지 않는다.

표준오차는 표본 포함률을 넣은 plug-in 값이다. 통과가 0회여서 이 값이 0이어도 실제 확률이 0이라고 확정하지 않으며 Wilson 구간을 함께 본다.

`within_current_world_pair_count`, `within_current_pretest_item_limit`는 현재 세계·설정에서의 필요조건만 표시한다. 둘 다 참이어도 다른 보호 분할과 학습 분할에 공간이 필요하다. 현재 36개 물체 쌍으로 98~320쌍을 실제 사전검사하는 기능을 추가한 것은 아니다. 기본 `max_pair_draws` 예산은 5천만 회이고, 초과 계획은 실행 전에 거부한다.

현재 시뮬레이션은 유한모집단 비복원 추출 보정, 실제 모델 반복의 변동, 특정 언어의 오류 구조를 자동 추정하지 않는다. 표본 수·허용 폭·평가 정책의 결정은 이 민감도 결과와 연구 목표를 함께 보고 별도로 해야 한다. 숫자가 잘 나오는 가정/허용 폭만 선택해 본실험 조건을 사후 맞추지 않는다.
