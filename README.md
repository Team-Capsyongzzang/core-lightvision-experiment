# core-lightvision-experiment

xBD pre/post disaster satellite tile을 보고 재난 분석 작업의 처리 우선순위를 분류하는 ResNet18 기반 vision pipeline.

## 태스크 정의

```
입력: pre-disaster RGB 이미지 + post-disaster RGB 이미지
      → resize 후 6채널 tensor로 결합

출력: no_building / low / medium / high
```

이 프로젝트는 xBD 라벨의 건물 피해 정보를 이용해 tile별 risk score를 만들고, 이를 4단계 priority class로 변환한 뒤 ResNet18 분류 모델을 학습합니다. 최종 목적은 재난 분석에서 더 중요한 tile을 먼저 처리하도록 우선순위를 부여하는 것입니다.

## 레이블 생성 방식

`build_index.py`는 xBD 데이터셋의 split 디렉터리(`tier1`, `tier3`, `hold`, `test`)를 순회하면서 pre/post 이미지와 라벨 JSON pair를 찾습니다.

```
{stem}_pre_disaster.png
{stem}_post_disaster.png
{stem}_pre_disaster.json
{stem}_post_disaster.json
```

각 post-disaster 라벨의 building damage subtype은 다음 점수로 변환됩니다.

| damage subtype | score |
|---|---:|
| `no-damage` | 0 |
| `minor-damage` | 1 |
| `major-damage` | 2 |
| `destroyed` | 3 |

risk score는 피해 강도와 건물 밀도를 함께 반영합니다.

```text
severity_score = 0.7 * mean_damage_norm + 0.3 * max_damage_norm
impact_score   = 0.5 * building_coverage_norm + 0.5 * building_count_norm
risk_score     = 0.6 * severity_score + 0.4 * impact_score
```

클래스는 4가지입니다.

| class | 의미 |
|---|---|
| `no_building` | 건물이 없는 tile |
| `low` | 낮은 우선순위 |
| `medium` | 중간 우선순위 |
| `high` | 높은 우선순위 |

기본 threshold는 `low < 0.35`, `medium < 0.65`, 그 이상은 `high`입니다. 건물이 없는 tile은 별도로 `no_building`으로 분류합니다.

## 프로젝트 구조

```text
core-lightvision-experiment/
├── constants.py             # damage score, split, threshold 기본값
├── labels.py                # xBD JSON parsing, risk score 계산
├── build_index.py           # 이미지/라벨 pair 탐색 및 index CSV 생성
├── dataset.py               # pre/post 이미지를 6채널 tensor로 구성
├── models.py                # ResNet18 입력 채널 확장 및 classifier 교체
├── train.py                 # ResNet18 학습, metric 저장
├── eval.py                  # checkpoint 평가
├── priority_classifier.py   # 학습된 모델 기반 tile priority 추론
├── sqs_setup.py             # priority별 SQS queue/DLQ 생성
├── sqs_producer.py          # tile job 분류 후 SQS publish
└── priority_worker.py       # 높은 priority queue부터 polling
```

## 실행 준비

코드가 package relative import를 사용하므로 저장소 안에서 파일을 직접 실행하지 않고, 상위 디렉터리에서 module 방식으로 실행합니다.

```bash
cd /home/mglee
```

필요한 주요 패키지:

```bash
pip install torch torchvision pillow numpy tqdm
pip install boto3  # SQS 기능을 사용할 때만 필요
```

## Index CSV 생성

4-class building-aware label을 생성합니다.

```bash
python3 -m core-lightvision-experiment.build_index \
  --dataset-root /path/to/xBD \
  --output runs/index_buildingaware.csv \
  --label-mode building_threshold \
  --severity-weight 0.6 \
  --impact-weight 0.4
```

생성되는 CSV에는 이미지 경로, 건물 수, 피해 통계, `risk_score`, `risk_class`, `risk_class_name`이 포함됩니다.

## ResNet18 학습

```bash
python3 -m core-lightvision-experiment.train \
  --index-csv runs/index_buildingaware.csv \
  --output-dir runs/resnet18_buildingaware \
  --model resnet18 \
  --task four_class_building_aware \
  --image-size 320 \
  --epochs 10 \
  --batch-size 16 \
  --lr 1e-3 \
  --use-weighted-sampler
```

학습 과정에서는 train split(`tier1`, `tier3`)을 사용하고, validation은 `hold`, test는 `test` split을 사용합니다. 클래스 불균형을 줄이기 위해 class weight와 weighted sampler를 사용할 수 있습니다.

학습 결과는 `output-dir`에 저장됩니다.

```text
best_model.pt
last_model.pt
best_val_metrics.json
test_metrics.json
history.json
train_config.json
```

## 평가

```bash
python3 -m core-lightvision-experiment.eval \
  --index-csv runs/index_buildingaware.csv \
  --checkpoint runs/resnet18_buildingaware/best_model.pt \
  --model resnet18 \
  --task four_class_building_aware \
  --split test \
  --output-json runs/resnet18_buildingaware/eval_test.json
```

평가 결과로 confusion matrix, macro F1, priority recall이 출력됩니다.

## Priority 추론

단일 tile에 대해 priority를 예측합니다.

```bash
python3 -m core-lightvision-experiment.priority_classifier \
  --pre-image /path/to/tile_pre_disaster.png \
  --post-image /path/to/tile_post_disaster.png \
  --stage0-checkpoint runs/resnet18_stage0_building_vs_no_building_320/best_model.pt \
  --stage1-checkpoint runs/resnet18_stage1_high_vs_non_high_320/best_model.pt \
  --stage2-checkpoint runs/resnet18_stage2_medium_vs_low_320_buildingaware/best_model.pt \
  --model resnet18 \
  --image-size 320
```

출력은 priority, label, stage별 예측 결과, softmax probability를 포함한 JSON입니다.

```text
priority 0 → no_building
priority 1 → low
priority 2 → medium
priority 3 → high
```

## SQS Priority Scheduling

priority별 SQS queue와 DLQ를 생성합니다.

```bash
python3 -m core-lightvision-experiment.sqs_setup \
  --region ap-northeast-2 \
  --prefix tile
```

출력된 queue URL을 환경변수로 등록합니다.

```bash
export SQS_TILE_HIGH_URL="..."
export SQS_TILE_MEDIUM_URL="..."
export SQS_TILE_LOW_URL="..."
export SQS_TILE_NO_BUILDING_URL="..."
```

CSV 또는 JSONL job을 분류한 뒤 priority queue로 보냅니다.

```bash
python3 -m core-lightvision-experiment.sqs_producer \
  --input jobs.csv \
  --region ap-northeast-2
```

worker는 `high → medium → low → no_building` 순서로 queue를 확인합니다.

```bash
python3 -m core-lightvision-experiment.priority_worker \
  --region ap-northeast-2 \
  --process-command "python3 process_tile.py"
```

## 전체 흐름

```text
xBD images + labels
    │
    ├─ build_index.py
    │   └─ no_building / low / medium / high index CSV 생성
    │
    ├─ train.py
    │   └─ ResNet18 priority classifier 학습
    │
    ├─ eval.py
    │   └─ confusion matrix, macro F1, priority recall 확인
    │
    └─ sqs_producer.py + priority_worker.py
        └─ tile priority 기반 scheduling
```
