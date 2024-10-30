TensorBoard를 활용하면 모델 학습 과정에서 생성된 로그를 시각적으로 확인할 수 있습니다. 특히 YOLO 모델을 파인튜닝할 때 학습 진행 상황을 모니터링하는 데 유용합니다. 다음은 Jupyter Notebook이나 터미널 환경에서 TensorBoard를 설정하고 실행하는 방법입니다.

### 1. TensorBoard 로그 저장 설정
YOLO 모델 학습 시, model.train() 메서드의 project와 name 파라미터를 설정하면 지정된 경로에 로그가 저장됩니다. 예를 들어 project="runs"와 name="yolo_finetune"을 지정하면 runs/yolo_finetune 폴더에 로그가 생성됩니다.


```python
from ultralytics import YOLO

# YOLO 모델 불러오기
model = YOLO("yolov8n.pt")

# 학습과정에서 TensorBoard 로그 저장 설정
model.train(data="path/to/data.yaml", epochs=50, imgsz=640, project="runs", name="yolo_finetune")
```

### 2. TensorBoard 실행 및 시각화
TensorBoard를 통해 학습 과정에서 기록된 로그를 확인하려면 다음 단계를 따릅니다.

(1) 터미널에서 TensorBoard 실행
프로젝트 폴더에서 터미널을 열고 다음 명령어를 입력합니다.

```bash
tensorboard --logdir=runs
```

- 여기서 --logdir 파라미터에는 로그 파일이 저장된 디렉토리(runs)를 지정합니다.

- 명령어를 실행하면 TensorBoard가 기본적으로 http://localhost:6006에서 실행되며, 브라우저를 통해 접속할 수 있습니다.

(2) Jupyter Notebook에서 실행
Jupyter Notebook 환경에서 TensorBoard를 열 수 있습니다. 다음 명령어를 노트북 셀에 입력하여 TensorBoard 인터페이스를 표시합니다.

```python
%load_ext tensorboard
%tensorboard --logdir=runs
```
이렇게 설정하면 학습이 진행되는 동안 손실값 변화, 정밀도, 재현율 등의 지표 변화를 실시간으로 확인할 수 있습니다. TensorBoard는 학습 진행 상황을 모니터링하고, 파인튜닝의 성능 향상을 시각적으로 확인하는 데 유용한 도구입니다.