# autolabel

검사 모델 자동라벨링의 **판정 신호 · 판정 규칙 · 거리 통계 · 평가 · 임계값 보정**을 담은 라이브러리다.
서버(`app.labeling`)에서 계산 부분만 떼어 냈다. 서버 구조를 바꾸거나 새로 짜도 이 라이브러리를 그대로 가져다 쓰고,
결과가 같은지는 시험으로 확인한다.

- **다루지 않는 것:** DB, HTTP, 작업 큐, 파일 저장 위치, 환경 변수. 이것들은 서버의 몫이다.
- **입력:** 이미지 배열(numpy), 모델(예측기·분류 어댑터), 설정 객체(`config.py`)
- **출력:** 서버 DB 에 그대로 저장하는 dict. 형식은 검출 신호 v4·판정 v3, 분류 신호 v1·판정 v1 이다.

## 구조

| 위치 | 내용 | 의존 |
|---|---|---|
| `constants.py` | 입력 크기 640, 작업 종류, 신호 이름(`SignalName`), 거리 방식, 규격 버전 | — |
| `config.py` | 설정 묶음: `TtaConfig` · `RuleConfig` · `StageConfig` · `CalibrationConfig` · `StageBConfig` · `ClassifyConfig` | — |
| `geometry.py` | 사각형·회전 박스 표현, 각도 규칙, 정확한 회전 IoU, 검수 꼭짓점 정리 | numpy, cv2 |
| `features.py` | 검출 모델 neck 특징맵 → 이미지 임베딩·박스 특징(RoIAlign·회전 샘플링) | numpy (특징맵 함수는 torch) |
| `distance/` | ③ 이미지 거리(`ImageDistance`, 모든 작업 종류), ④ 박스 거리(`BoxDistance`, 검출) | numpy, scikit-learn |
| `metrics.py` | AUROC, 순위 결합, AP, 오류율 상한(Clopper-Pearson), 감사 표본 추첨 | scipy |
| `stages.py` | 클래스별 자동라벨링 단계(0 보조 · 1 부분 자동 · 2 안정)와 감사 비율 | — |
| `labels.py` | 작업 종류별 확정 라벨 형식과 검사(`label_error`) | — |
| `images.py` | 이미지 읽기(`decode="color"`·`"unchanged"`) | cv2 |
| `tasks/` | 작업 종류별 모듈 등록표(`task_module`, `available_tasks`) | — |
| `tasks/detect/` | **검출(detect·obb)**: 예측기(YOLO), 신호 ①~⑤, 판정, 평가·보정. 서버에서 운영 중이다 | (예측기만 torch·ultralytics) |
| `tasks/classify/` | **분류 참조 구현**: 어댑터 점수·임베딩·TTA 선언으로 신호를 계산하고, 판정·평가·보정한다 | numpy, cv2 |
| `plugins.py` | 신호 플러그인 규격: 외부(제이비소프트)가 모델 고유 신호를 붙이는 지점 | — |
| `conformance.py` | 연동 확인 도구: 분류 어댑터·신호 플러그인 검사, 명령줄 실행 | — |

**경계:** 라이브러리를 import 해도 `app` · fastapi · sqlalchemy · torch · ultralytics 를 불러오지 않아야 한다.

## 서버가 쓰는 방식 (검출)

```python
from autolabel.tasks import detect
from autolabel.config import TtaConfig, RuleConfig
from autolabel.stages import class_stages

setup = detect.SignalSetup(
    task="detect", enabled=True, variants=detect.tta_variants(TtaConfig()), match_iou=0.3,
    conf_thresholds=conf,                                   # 신호 상태(보정 결과)의 클래스별 ① 임계값
    rules=detect.rule_params(conf, tta_iou, agree_iou, class_stages(states, model_version, stage_cfg), RuleConfig()),
    distance=image_distance, box=box_distance, box_active=box_active, stats_id=stats_id, production=production_ref)
for item in detect.analyze(predictor, arrays, {"conf": 0.05, "iou": 0.7, "max_det": 300}, cmap, names, setup,
                           production=production_predictor):
    judgment = detect.judge(item["boxes"], item["signals"], setup.rules, f"{batch_id}:{sha256}")
```

평가셋 평가는 `tasks.detect.evaluation.evaluate_image` 가 이미지마다 평가 행·이미지 정보·클래스별 정답/맞은 수를 만든다.
그 결과를 `summarize`(AUROC) · `decide`(④ B단계) · `calibrate`(임계값 보정)에 넘긴다.

지금 서버에서는 `app.labeling.autolabel.SignalContext` 가 `SignalSetup` 을 상속한다. DB 의 신호 상태·거리 통계·단계를 읽어
위 인자를 채운다. `app.labeling.*` 의 신호·규칙·기하·특징 모듈은 이 라이브러리를 다시 내보내기만 한다(예전 import 경로 호환).

## 분류 (참조 구현)

```python
from autolabel.tasks import classify

info = classify.ModelInfo.from_adapter(adapter)             # 분류 어댑터 규격 1.0 의 Adapter
model = adapter.load(weights, "cuda:0", {})                 # LoadedModel: predict → (N, C), embed → (N, D)
setup = classify.ClassifySetup(names=["OK", "NG"], info=info, distance=image_distance,
                               plugins=plugins.registered("classify", info.name))
for item in classify.analyze(model, arrays, setup, production=production_model):
    judgment = classify.judge(item["label"], item["signals"], setup.rules, audit_key)
```

- **판정 결과 키:** 검출과 같다(`decision` · `audit` · `priority` · `gate` · `held` · `boxes`). 그래서 서버 집계·검수 순서·루프가
  그대로 동작한다. 분류의 사유는 모두 이미지 단위라 `gate` 에 들어가고, `boxes` 는 빈 목록이다.
- **서버 연동은 다음 단계다.** 서버 쪽 작업(자동라벨링·평가 작업, 분류 클래스 구성, 검수 묶음 형식)과는 아직 이어지지 않았다.

## 신호 플러그인 (외부 개발 확장)

라벨링 확장 규격 1.0 은 문서 `검사모델_라벨링_확장규격_Y-Map_Dotting_초안_20261030.docx` 에 있다.
플러그인은 이미지마다 `PluginSignal(value, detail)` 을 낸다. `value` 는 클수록 틀렸을 가능성이 높은 값이다.
판정·보정·감사는 플랫폼이 한다. 플러그인이 실패하면 그 묶음의 이미지들만 검수로 보내고(`plugin_error:<이름>`), 작업은 계속한다.

```bash
python -m autolabel.conformance classifier --adapter dotting --weights dotting.keras --images samples/ \
    --plugin dotting_attention
```

## 새 작업 종류 추가 (분할·이상탐지)

1. `labels.py` 에 라벨 형식을 정한다. 분할 `polygons_norm` 과 이상탐지 `judge` 는 이미 정해 두었다.
2. `tasks/<종류>/` 에 다섯 가지를 만든다.
   - `Setup`: 신호 조건 객체
   - `analyze`: 모델 → 라벨·신호
   - `judge`: 판정. 결과 키는 위와 같게 한다.
   - `evaluation`: 정답 대조·AUROC·보정
   - `SIGNALS_VERSION` · `RULES_VERSION`
3. `tasks/__init__.py` 에 `TaskModule` 을 만들어 `register_task` 로 등록한다.
4. 단계(`stages`)·감사 표본(`metrics.audit_pick`)·오류율 상한(`metrics.error_bound`)·③ 거리(`distance.image`)는 공통으로 쓴다.

## 버전

| 항목 | 값 | 바뀌는 때 |
|---|---|---|
| 라이브러리 `LIBRARY_VERSION` | 0.1.0 | 공개 함수·인자가 바뀔 때 |
| 검출 신호·판정 형식 | v4 · v3 | DB 에 저장되는 dict 형식이 바뀔 때. 예전 결과와 구분하는 데 쓴다 |
| 분류 신호·판정 형식 | v1 · v1 | 같음 |
| 라벨링 확장 규격 `LABELING_SPEC_VERSION` | 1.0 | 플러그인 규격. 주 버전이 다르면 등록을 거부한다 |

## 시험

```bash
python -m pytest                                       # 서버 전체: 검출은 서버 시험이 같은 코드를 지나간다
```
