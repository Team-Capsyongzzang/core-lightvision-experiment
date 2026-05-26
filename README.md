# core-lightvision-experiment

xBD pre/post disaster satellite tile을 보고 재난 분석 작업의 처리 우선순위를 분류하는 경량 vision pipeline.

## 태스크 정의

```
입력: pre-disaster RGB 이미지 + post-disaster RGB 이미지
      → 224/320 크기로 resize 후 6ch tensor로 결합

출력:
  기본 3-class priority  → low / medium / high
  building-aware priority → no_building / low / medium / high
```

이 저장소는 xBD label JSON에서 건물 피해 정보를 읽어 tile별 risk score를 만들고, 그 score를 기준으로 이미지 분류 모델을 학습합니다. 학습된 모델은 단일 tile 분류 또는 SQS 기반 priority scheduler 입력으로 사용할 수 있습니다.

## 레이블 생성 방식

`build_index.py`가 xBD split 디렉터리(`tier1`, `tier3`, `hold`, `test`)를 순회하면서 이미지/라벨 쌍을 찾습니다.

```
{stem}_pre_disaster.png
{stem}_post_disaster.png
{stem}_pre_disaster.json
{stem}_post_disaster.json
```

라벨 JSON의 building polygon과 damage subtype을 이용해 다음 값을 계산합니다.

```
damage subtype:
  no-damage     → 0
  minor-damage  → 1
  major-damage  → 2
  destroyed     → 3

severity_score = 0.7 * mean_damage_norm + 0.3 * max_damage_norm
impact_score   = 0.5 * building_coverage_norm + 0.5 * building_count_norm
risk_score     = severity_weight * severity_score + impact_weight * impact_score
```

기본 threshold는 `low < 0.35`, `medium < 0.65`, 그 이상은 `high`입니다. `building_threshold` 모드에서는 건물이 없는 tile을 `no_building` 클래스로 따로 분리합니다. `quantile` 모드는 train split 기준 분위수로 low/high cutoff를 정합니다.

## 프로젝트 구조

```
core-lightvision-experiment/
├── constants.py             ← damage score, split, risk threshold 기본값
├── labels.py                ← xBD JSON parsing, tile risk score 계산
├── build_index.py           ← 이미지/라벨 pair 탐색 및 index CSV 생성
├── dataset.py               ← 6ch pre/post Dataset, split/task 변환
├── models.py                ← ResNet18 / MobileNetV3-Small 입력 채널 확장
├── train.py                 ← 학습, class weight, sampler, metric 저장
├── eval.py                  ← 저장된 checkpoint 단일 모델 평가
├── eval_two_stage.py        ← high 판별 + low/medium 판별 2-stage 평가
├── priority_classifier.py   ← no_building → high → low/medium 3-stage 추론기
├── sqs_setup.py             ← priority별 SQS queue/DLQ 생성
├── sqs_producer.py          ← tile job 분류 후 priority SQS queue로 publish
├── priority_worker.py       ← high→medium→low→no_building 순서로 SQS polling
└── sqs_config.py            ← priority별 queue URL 환경변수 이름
```

## 실행 준비

현재 코드는 package relative import를 사용하므로 저장소 안에서 파일을 직접 실행하지 말고, 상위 디렉터리(`/home/mglee`)에서 `python3 -m core-lightvision-experiment.<module>` 형태로 실행합니다.

필요한 주요 패키지:

```bash
pip install torch torchvision pillow numpy tqdm
pip install boto3  # SQS 기능을 쓸 때만 필요
```

## Index CSV 생성

```bash
cd /home/mglee

python3 -m core-lightvision-experiment.build_index \
  --dataset-root /path/to/xBD \
  --output runs/index_threshold.csv \
  --label-mode threshold
```

building-aware 4-class label을 만들 때:

```bash
python3 -m core-lightvision-experiment.build_index \
  --dataset-root /path/to/xBD \
  --output runs/index_buildingaware.csv \
  --label-mode building_threshold
```

quantile 기반 label을 만들 때:

```bash
python3 -m core-lightvision-experiment.build_index \
  --dataset-root /path/to/xBD \
  --output runs/index_quantile.csv \
  --label-mode quantile \
  --quantile-low-ratio 0.30 \
  --quantile-high-ratio 0.30 \
  --quantile-reference-splits tier1 tier3
```

## 학습

```bash
python3 -m core-lightvision-experiment.train \
  --index-csv runs/index_threshold.csv \
  --output-dir runs/mobilenet_three_class \
  --model mobilenet_v3_small \
  --task three_class \
  --image-size 224 \
  --epochs 10 \
  --batch-size 16 \
  --lr 1e-3 \
  --use-weighted-sampler
```

지원 모델:

| 옵션 | 설명 |
|---|---|
| `mobilenet_v3_small` | 기본값, lightweight classifier |
| `resnet18` | 더 큰 baseline |

지원 task:

| task | 클래스 |
|---|---|
| `three_class` | low / medium / high |
| `four_class_building_aware` | no_building / low / medium / high |
| `high_vs_non_high` | non_high / high |
| `medium_vs_low` | low / medium |

학습 결과는 `output-dir`에 저장됩니다.

```
best_model.pt
last_model.pt
best_val_metrics.json
test_metrics.json
history.json
train_config.json
```

## 평가

단일 checkpoint 평가:

```bash
python3 -m core-lightvision-experiment.eval \
  --index-csv runs/index_threshold.csv \
  --checkpoint runs/mobilenet_three_class/best_model.pt \
  --model mobilenet_v3_small \
  --task three_class \
  --split test \
  --output-json runs/mobilenet_three_class/eval_test.json
```

2-stage 평가:

```bash
python3 -m core-lightvision-experiment.eval_two_stage \
  --index-csv runs/index_threshold.csv \
  --split test \
  --stage1-checkpoint runs/stage1_high_vs_non_high/best_model.pt \
  --stage1-model resnet18 \
  --stage1-task high_vs_non_high \
  --stage2-checkpoint runs/stage2_medium_vs_low/best_model.pt \
  --stage2-model resnet18 \
  --output-json runs/two_stage_eval.json
```

## 3-stage Priority Classifier

`priority_classifier.py`는 scheduler에 넣기 좋은 4단계 priority를 반환합니다.

```
Stage 0: no_building vs has_building
Stage 1: non_high vs high
Stage 2: low vs medium

priority:
  0 → no_building
  1 → low
  2 → medium
  3 → high
```

단일 tile 추론:

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

출력은 priority, label, stage별 label, stage별 softmax probability를 담은 JSON입니다.

## SQS Priority Scheduling

priority별 queue와 DLQ 생성:

```bash
python3 -m core-lightvision-experiment.sqs_setup \
  --region ap-northeast-2 \
  --prefix tile
```

출력되는 환경변수를 shell에 export합니다.

```
SQS_TILE_HIGH_URL
SQS_TILE_MEDIUM_URL
SQS_TILE_LOW_URL
SQS_TILE_NO_BUILDING_URL
```

CSV 또는 JSONL job을 분류한 뒤 priority queue로 publish:

```bash
python3 -m core-lightvision-experiment.sqs_producer \
  --input jobs.csv \
  --region ap-northeast-2
```

SQS 전송 없이 message만 확인:

```bash
python3 -m core-lightvision-experiment.sqs_producer \
  --input jobs.csv \
  --dry-run \
  --limit 5
```

worker는 `high → medium → low → no_building` 순서로 queue를 polling합니다.

```bash
python3 -m core-lightvision-experiment.priority_worker \
  --region ap-northeast-2 \
  --process-command "python3 process_tile.py"
```

`--process-command`를 넘기면 SQS message JSON이 `TILE_JOB_JSON` 환경변수로 전달됩니다. 넘기지 않으면 처리된 tile id와 priority만 출력합니다.

## 파이프라인 요약

```
xBD images + labels
    │
    ├─→ build_index.py
    │     └─ 건물 피해 기반 risk_score / risk_class CSV 생성
    │
    ├─→ train.py
    │     └─ pre/post 6ch vision classifier 학습
    │
    ├─→ eval.py / eval_two_stage.py
    │     └─ confusion matrix, macro F1, high-risk recall 확인
    │
    └─→ priority_classifier.py + sqs_producer.py
          └─ tile priority 산출 후 SQS priority queue로 스케줄링
```
