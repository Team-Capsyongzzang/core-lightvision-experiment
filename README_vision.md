# xBD 재난 분류기 (Disaster Classifier)

위성 이미지(post-disaster)를 보고 재난 종류를 분류하는 파이프라인.

## 태스크 정의

```
입력: post-disaster 위성 이미지 (3ch, 224×224)
출력: 재난 종류 — earthquake / flood / hurricane / tornado / tsunami / wildfire
```

## 레이블 생성 방식 (방식 A → B)

xBD 파일명에서 재난 종류를 자동 추출 (방식 A) 하여  
ResNet50 분류 모델을 학습 (방식 B) 합니다.

```
파일명: socal-fire_00000371 → wildfire
파일명: midwest-flooding_00000181 → flood
```

## 프로젝트 구조

```
xbd_disaster_classifier/
├── config/config.py          ← 모든 설정 (클래스, 하이퍼파라미터)
├── data/
│   ├── indexer.py            ← 파일명 → 레이블 자동 생성
│   └── dataset.py            ← Dataset / DataLoader
├── models/classifier.py      ← ResNet50 분류기
├── losses/losses.py          ← Focal Loss + 클래스 가중치
├── metrics/metrics.py        ← Accuracy, F1, Confusion Matrix
├── training/trainer.py       ← Phase1(frozen) + Phase2(fine-tuning)
├── evaluation/evaluator.py   ← 평가
├── visualization/visualizer.py ← 혼동 행렬, 예측 그리드
├── utils/seed.py
└── main.py                   ← CLI 진입점
```

## 실행

```bash
# 전체 파이프라인
python main.py

# 단계 선택
python main.py --steps index train
python main.py --steps finetune eval_test predict
```

## 학습 전략

```
Phase 1 (20 epochs): backbone frozen → head만 학습
  → lr=1e-4, Cosine scheduler

Phase 2 (10 epochs): backbone unfrozen → 전체 fine-tuning
  → backbone lr=2e-6, head lr=2e-5
  → catastrophic forgetting 방지
```

## 예상 성능

| 클래스 | 난이도 | 이유 |
|---|---|---|
| wildfire | 쉬움 | 탄 지형 패턴 뚜렷 |
| flood | 보통 | 갈색 물 영역 |
| hurricane | 보통 | 해안 범람 + 건물 파손 |
| tornado | 어려움 | 국소적 파손, 산불과 혼동 |
| earthquake | 어려움 | 건물 붕괴 패턴이 허리케인과 유사 |
| tsunami | 어려움 | 샘플 수 적음, 홍수와 유사 |

## 세그멘테이션 파이프라인과의 연결

```
위성 이미지
    │
    ├─→ [이 모델] 재난 분류 → "홍수"
    │
    └─→ [세그멘테이션 모델] 건물 피해 등급 맵
    
→ "이 지역은 홍수로 인해 건물 37%가 Major 이상 피해"
→ LLM 리포트 생성 (제안서 파트3)
```
