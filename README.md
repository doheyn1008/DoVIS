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

Python 3.11 이상이면 추가 패키지나 API 키 없이 데모를 실행할 수 있습니다. 실제 모델은 별도의 로컬 비전 언어 모델 서버 두 개를 연결합니다.
