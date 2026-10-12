# Fashion MNIST Anomaly Detection Project

본 리포지토리는 컴퓨터비전 수업 팀 프로젝트인 **'Fashion MNIST 데이터셋을 활용한 이상 탐지(Anomaly Detection)'** 과제를 수행하기 위한 워크스페이스입니다.

## 1. Project Overview

* **목표:** Fashion MNIST 데이터셋을 활용하여 정상(Normal) 샘플과 비정상(Abnormal) 샘플을 구별하는 이상 탐지 모델 개발
* 데이터셋 규칙:


* **Train Data:** 오직 `Label 2 (Pullover)`만 정상 샘플로 사용하여 모델 학습
* **Test Data:** `Label 0 (T-shirt)`, `Label 2 (Pullover)`, `Label 4 (Coat)`, `Label 6 (Shirt)`를 사용하여 평가 진행


* **핵심 평가 지표:** AUROC 및 Kaggle Leaderboard 성능



---

## 2. Ground Rules (수업 필수 제약 사항)

팀원 모두 모델링 및 코드 작성 시 아래 사항을 **반드시 엄수**해야 합니다. (위반 시 0점 처리)

1. **Reconstruction 기반 모델 필수:** 모델의 골자는 반드시 Autoencoder 등 Reconstruction 기반이어야 합니다. (Reconstruction을 중심으로 다른 기법을 추가하는 것은 허용). GAN, Diffusion 등을 사용해도 되나, 결과적으로 Reconstruction 방식을 따라야 합니다.


2. **테스트 과정의 공정성 (매우 중요):** Test 과정에서 라벨(Label) 정보를 활용하여 Anomaly Score를 인위적으로 조작하거나(예: 가중치 곱하기, 특정 라벨에만 노이즈 추가), Test 데이터를 Train에 사용하는 등 인위적인 개입이 적발되면 0점 처리됩니다.


3. **프레임워크 및 환경:** 반드시 **PyTorch**를 사용해야 하며(Keras, Tensorflow 불가), 훈련 시 필요 VRAM은 **10GB**를 초과할 수 없습니다. Mac 환경은 채점 제약으로 인해 불가하므로 Windows/Linux 또는 본 레포지토리의 Docker 환경을 사용합니다.



---

## 3. Repository Structure

본 프로젝트는 재현성과 모듈화를 위해 아래와 같은 구조로 설정했습니다(추후 변경 가능).

```text
.
├── .devcontainer/        # VS Code Dev Container 설정 파일
├── notebooks/            # 실험 및 데이터 분석용 Jupyter Notebooks
│   ├── EDA.ipynb         # 데이터 탐색 및 시각화
│   └── demo.ipynb        # 모델 추론 테스트
├── scripts/              # 실행 자동화 쉘 스크립트
│   └── baseline.sh       # 베이스라인 모델 학습/평가 스크립트
├── src/                  # 재사용 가능한 핵심 모듈 (여기에 코드 작성)
│   ├── __init__.py
│   ├── data.py           # Dataset, DataLoader 및 전처리 로직
│   ├── main.py           # Argument 기반의 통합 실행 파이프라인
│   ├── models.py         # Autoencoder 등 네트워크 아키텍처 정의
│   ├── trainer.py        # 훈련(Train) 및 검증(Val) 루프 로직
│   └── utils.py          # AUROC 계산, Loss 함수 등 유틸리티
├── tests/                # src/ 내부 모듈에 대한 Unit Test
├── Dockerfile            # PyTorch 기반 프로젝트 환경 설정 파일
├── .dockerignore
├── .gitignore
└── README.md

```

---

## 4. Getting Started (환경 세팅)

### Windows 로컬 Python 가상환경

현재 노트북용 구성은 **Python 3.12 (64-bit) + PyTorch 2.14.1 + torchvision 0.29.1 + CUDA 13.0 wheel**입니다.

| 항목 | 사양 / 설정 |
| --- | --- |
| CPU | AMD Ryzen 7 260 (사용자 제공) |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU, VRAM 8GB, compute capability 12.0 |
| GPU 드라이버 | 616.92 |
| RAM | 32GB (사용자 제공) |
| 가상환경 위치 | `03 AD_Git/.venv` |
| 프로젝트 마감 | 2026-10-26 |

RTX 50 시리즈는 Blackwell 지원 빌드가 필요하므로 기존 `cu124` 안내 대신 `cu130`을 사용합니다.
`nvidia-smi`의 CUDA 표시는 드라이버 지원 정보이며, 가상환경에 설치한 PyTorch의 CUDA 버전과 다를 수 있습니다.
일반적인 PyTorch wheel 실행에는 CUDA Toolkit/cuDNN을 별도로 설치하지 않아도 됩니다.
공식 배포 파일: [PyTorch CUDA 13.0](https://download.pytorch.org/whl/cu130/torch/),
[torchvision CUDA 13.0](https://download.pytorch.org/whl/cu130/torchvision/),
[Blackwell 및 드라이버 안내](https://pytorch.org/blog/pytorch-2-12-release-blog/).

**설치 상태:** 아래는 설치 절차입니다. 현재 Python 3.10.11만 확인했으며, Python 3.12 및 PyTorch 설치와 GPU 연산 검증은 아직 완료하지 않았습니다.

1. Python 3.12 64-bit를 준비합니다. 공식 Windows 설치 프로그램은
   [Python 3.12.10](https://www.python.org/downloads/release/python-31210/)에서 받을 수 있습니다.
   이미 3.12가 설치되어 있다면 해당 실행 파일을 사용하면 됩니다.
2. PowerShell에서 `03 AD_Git` 폴더를 열고 다음 명령을 순서대로 실행합니다.
   `py -3.12`를 찾지 못하면 첫 줄의 실행 파일을 설치된 Python 3.12의 절대 경로로 지정합니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .hardware_requirements/gpu_requirements.txt
.\.venv\Scripts\python.exe -m pip install -r .hardware_requirements/common_requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/check_environment.py --device cuda
```

GPU/CPU별 파일은 PyTorch 전용 저장소를 사용하고, 공통 패키지는 별도 명령으로 PyPI에서 설치합니다.
기존 요구사항의 비전 패키지는 `common_requirements.txt`로 옮겼으며, 실제 코드가 사용하는
NumPy, pandas, matplotlib, scikit-learn, tqdm과 Jupyter 관련 패키지를 추가했습니다.
CPU만 사용할 경우 별도 `.venv-cpu`를 만들고 `cpu_requirements.txt`를 사용한 뒤 `--device cpu`로 검증합니다.

`check_environment.py`는 데이터 다운로드 없이 난수 이미지로 기존 Autoencoder와 CNN의
순전파, 역전파, optimizer 갱신을 검사합니다. CUDA 모드에서는 GPU를 사용할 수 없으면 실패합니다.
이 검사는 환경 확인용이며, 실제 데이터 학습 성능이나 최종 모델의 VRAM 사용량을 보장하지 않습니다.
설치에 성공하면 다음 명령으로 정확한 패키지 버전을 기록할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe -m pip freeze | Out-File -Encoding utf8 .hardware_requirements/windows-gpu.lock.txt
```

재설치는 GPU 패키지를 위 절차대로 설치한 뒤 `windows-gpu.lock.txt`를 사용합니다.
이 lock 파일은 해당 Windows/Python/GPU 환경용입니다.

VS Code에서 `Python: Select Interpreter`로 `.venv\Scripts\python.exe`를 선택하고,
노트북의 커널도 같은 가상환경으로 선택합니다. Jupyter Lab 실행은 다음과 같습니다.

```powershell
.\.venv\Scripts\python.exe -m jupyter lab
```

가상환경 활성화 없이 위처럼 실행 파일을 직접 지정할 수 있습니다.
로컬 학습 파이프라인 실행 전에는 `src/main.py`와 `src/data.py`의 `/workspace` 고정 경로,
`src/trainer.py`의 무조건적인 `model.compile()` 호출을 Windows 환경에 맞게 조정해야 합니다.
초기 환경 검증 스크립트는 이 학습 파이프라인을 호출하지 않습니다.

### Docker / Dev Containers

로컬 환경의 충돌을 막기 위해 **Docker 및 VS Code Dev Containers** 사용을 권장합니다.
(기반 이미지: `nvcr.io/nvidia/pytorch:26.09-py3`)

**방법 1: VS Code Dev Containers (권장)**

1. VS Code 확장 프로그램에서 `Dev Containers`를 설치합니다.
2. 프로젝트 폴더를 열고 `Ctrl + Shift + P` (Mac: `Cmd + Shift + P`)를 누릅니다.
3. `Dev Containers: Reopen in Container`를 선택하면 자동으로 Docker 환경이 빌드되고 연결됩니다.

**방법 2: Manual Docker Build**

```bash
# 이미지 빌드
docker build -t cv-anomaly-detection .

# 컨테이너 실행 (Jupyter Lab 포트 연결 포함)
docker run --gpus all -it -p 8888:8888 -v $(pwd):/workspace cv-anomaly-detection

```

---

## 5. Deliverables & Deadlines

우리가 최종적으로 제출해야 할 항목들은 다음과 같습니다.

1. Kaggle 제출용 CSV 파일:


* 모델 추론을 통해 얻은 dataframe (`.csv`). (하루 최대 10회 제출 가능, 팀 대표 1인이 제출)




2. 최종 모델 코드 및 가중치 파일 (i-campus 제출):


* `code submission form.pdf` 양식에 맞춘 주피터 노트북(`.ipynb`) 파일. (주의: `src/` 폴더에서 개발하더라도, 최종 제출 시에는 노트북 하나로 실행되도록 코드를 통합해야 함)


* 훈련된 가중치 파일 (`.pt` 또는 `.pth`). 학습 없이 즉시 inference가 가능해야 함.




3. 결과 리포트:


* IEEE Conference Template (A4 양식, 2 column), Reference 포함 최대 2페이지 제한.


* 영어/한국어 모두 가능.




4. 발표 자료:


* 팀원 역할, 전체 흐름도, 우리 팀만의 아이디어, 결과(AUROC 그래프 및 수치 분석) 포함.


* 생성형 AI 사용 시 모델명 및 Prompt 명시 필수.
