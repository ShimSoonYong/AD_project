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
├── docs/
│   └── center_loss.md    # 중심 규제 loss(recon + λ||z−c||²) 실험 설명과 결과
├── scripts/              # 실행 자동화 쉘 스크립트
│   ├── baseline.sh       # 베이스라인 모델 학습/평가 스크립트
│   ├── center_loss_experiment.py # 중심 규제 loss 실험 (docs/center_loss.md 결과)
│   └── make_submission.py # i-campus 제출 노트북 생성 및 검증
├── src/                  # 재사용 가능한 핵심 모듈 (여기에 코드 작성)
│   ├── __init__.py
│   ├── data.py           # Dataset, DataLoader 및 전처리 로직
│   ├── main.py           # Argument 기반의 통합 실행 파이프라인
│   ├── models.py         # Autoencoder 등 네트워크 아키텍처 정의
│   ├── scoring.py        # SSIM / perceptual loss와 이상 점수 함수
│   ├── trainer.py        # 훈련(Train) 및 검증(Val) 루프 로직
│   └── utils.py          # AUROC 계산, 그래프, 결과 CSV 저장 (Scorer)
├── tests/                # src/ 내부 모듈에 대한 Unit Test
├── Dockerfile            # PyTorch 기반 프로젝트 환경 설정 파일
├── .dockerignore
├── .gitignore
└── README.md

```

---

## 4. Getting Started (환경 세팅)

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

**방법 3: 로컬 venv (Windows/Linux, Python 3.12)**

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate / Linux: source .venv/bin/activate

# GPU 없음
pip install -r cpu_requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
# NVIDIA GPU (CUDA 12.6+). 설치 후 torch.cuda.is_available()가 True인지 확인
pip install -r gpu_requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126

# 학습 (device는 cuda가 있으면 cuda, 없으면 cpu로 자동 선택)
python src/main.py --models Autoencoder CNN --epochs 20
```


## 5. Deliverables & Deadlines

우리가 최종적으로 제출해야 할 항목들은 다음과 같습니다.

1. Kaggle 제출용 CSV 파일:


* 모델 추론을 통해 얻은 dataframe (`.csv`). (하루 최대 10회 제출 가능, 팀 대표 1인이 제출)




2. 최종 모델 코드 및 가중치 파일 (i-campus 제출):


* `code submission form.pdf` 양식에 맞춘 주피터 노트북(`.ipynb`) 파일. (주의: `src/` 폴더에서 개발하더라도, 최종 제출 시에는 노트북 하나로 실행되도록 코드를 통합해야 함)


* 훈련된 가중치 파일 (`.pt` 또는 `.pth`). 학습 없이 즉시 inference가 가능해야 함.


* 제출 노트북 생성: 아래 명령은 `AD_project_sample/code submission form.ipynb` 양식의 빈칸을 채운 노트북(모델 코드는 `src/models.py`에서 복사)과 weight를 `submission/`에 만들고, 노트북을 실행해 Kaggle에 낸 CSV와 결과가 같은지 검사합니다. Linux에서 학습했다면 `--train_os Linux`를 붙입니다(FAQ: Linux 사용 시 주석 표기).

```bash
python scripts/make_submission.py --model Autoencoder --latent_dim 4 --hidden_dim 128 \
    --score perceptual --threshold 0 \
    --weight logs/final_C/Autoencoder/checkpoint/epoch15_weight.pth \
    --check_csv logs/final_C/Autoencoder/record/epoch15_result_team_2.csv \
    --members "이름 (학번)" "이름 (학번)" "이름 (학번)" "이름 (학번)"
```

모델 구조 옵션(`--latent_dim`, `--hidden_dim`)과 `--score`는 학습 때와 같은 값을 넣어야 합니다. `--train_note "..."`를 붙이면 어떤 loss로 학습했는지가 노트북 첫 셀 주석에 들어갑니다(학습 코드는 제출 대상이 아니므로 설명용).

* i-campus에는 `submission/code_submission_team2.ipynb`와 `submission/model_team2.pth` 두 파일만 올립니다(`.data/`, 이미지, CSV 제외).




3. 결과 리포트:


* IEEE Conference Template (A4 양식, 2 column), Reference 포함 최대 2페이지 제한.


* 영어/한국어 모두 가능.




4. 발표 자료:


* 팀원 역할, 전체 흐름도, 우리 팀만의 아이디어, 결과(AUROC 그래프 및 수치 분석) 포함.


* 생성형 AI 사용 시 모델명 및 Prompt 명시 필수.
