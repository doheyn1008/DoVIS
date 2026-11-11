# DoVIS
ai만들어서키우기

## 언어 상호학습 실험

두 에이전트가 그림을 공유하면서 자기 언어로 대화하고, **대화 횟수에 따른 양방향 이해도**를 측정하는 실험 환경입니다.

실행 방법과 연구 범위: [experiments/mutual-understanding](experiments/mutual-understanding/README.md)

```bash
cd experiments/mutual-understanding
python -m mutual_lab demo
python -m mutual_lab serve
```

Python 3.11 이상이면 추가 패키지나 API 키 없이 데모를 실행할 수 있습니다. 선택적으로 로컬 VLM의 문맥 적응을 측정하거나, PyTorch GRU를 모국어부터 학습시킨 뒤 같은 이해 가중치로 상대 언어를 누적 학습할 수 있습니다. 모국어 적격성·초기 이해도·독립 모델 반복을 구분하며, 아직 유효한 언어 습득 비용을 검증한 연구 결과는 아닙니다.
