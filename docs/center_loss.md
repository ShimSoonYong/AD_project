# 학습 Loss 실험: 재구성 오차 + 중심 규제 (DASVDD 계열)

Autoencoder의 학습 loss에 **"정상 데이터의 latent를 한 점(중심 c)으로 모으는 항"**을 더해 본 실험입니다. 결과적으로 성능이 오르지 않았고, 그 이유를 분석한 내용까지 정리합니다.

## 1. Loss 정의

```
loss = recon_error + λ · ||z − c||²
       └ 재구성 항 ┘     └ 중심 규제 항 ┘
```

| 기호 | 의미 |
|---|---|
| x, x̂ | 입력 이미지, Autoencoder가 복원한 이미지 |
| recon_error | 재구성 오차. 픽셀별 MSE의 평균 = mean((x̂ − x)²) |
| z | encoder가 만든 latent 벡터 |
| c | **학습 데이터(Pullover 6000장)의 latent 평균** |
| λ | 중심 규제 항의 세기 |
| ‖z − c‖² | latent 차원별 제곱 차이의 합 |

### 각 항이 학습시키는 것

| 항 | 업데이트되는 부분 | 역할 |
|---|---|---|
| recon_error | encoder + decoder | 입력을 잘 복원하도록 (튜토리얼과 같음) |
| λ · ‖z − c‖² | **encoder만** | 정상 이미지의 latent를 중심 c 쪽으로 당김 |

중심 규제 항이 encoder만 업데이트하는 이유는 두 가지입니다. 계산에 decoder가 들어가지 않고, c는 기울기를 받지 않는 고정값이기 때문입니다.

### c를 정하는 방법

1. 매 에폭 시작 때, 그 시점의 encoder로 학습 데이터 6000장의 latent를 모두 계산해 평균을 냅니다.
2. 그 에폭 동안은 c를 고정하고 학습합니다.
3. 학습 데이터(2번 라벨)만 쓰고, 테스트 데이터는 쓰지 않습니다.

## 2. 왜 시도했나

튜토리얼의 지적처럼 Autoencoder는 일반화를 너무 잘해서 비정상 이미지도 잘 복원해 버립니다. 정상 latent를 c 근처의 좁은 영역으로 모으면, decoder는 그 영역만 복원하도록 학습됩니다. 그러면 비정상 이미지가 들어와도 Pullover처럼 복원되어 재구성 오차가 커질 것이라고 기대했습니다.

아이디어는 Deep SVDD(Ruff et al., ICML 2018)의 "정상 데이터를 한 점 주위로 모으기"에서 왔습니다. 이를 Autoencoder와 결합한 DASVDD(Hojjati & Armanfard, IEEE TKDE 2023)와 같은 계열입니다. 이 실험은 두 가지 점에서 DASVDD와 다릅니다. c를 매 에폭 다시 계산했고, 이상 점수로는 재구성 오차를 썼습니다.

## 3. 실험 설정

| 항목 | 값 |
|---|---|
| 모델 | 튜토리얼과 같은 MLP Autoencoder (784 → 128 → latent → 128 → 784) |
| latent | 2, 8 |
| λ | 0 (규제 없음, 비교 기준), 0.01, 0.1, 1 |
| 학습 | 튜토리얼 설정: 5에폭, batch 1, Adam lr 0.005, 시드 0 |
| 이상 점수 | 재구성 오차 (MSE) |
| 표의 AUROC | 1~5에폭 AUROC의 평균 |

## 4. 결과

| latent | λ=0 (규제 없음) | λ=0.01 | λ=0.1 | λ=1 |
|---|---|---|---|---|
| 2 | **0.7291** | 0.7188 | 0.6808 | 붕괴 (3에폭에서 0.686) |
| 8 | 0.7235 | **0.7256** | 0.6840 | 붕괴 (1에폭에서 0.649) |

- λ를 키울수록 AUROC가 떨어졌고, λ=1에서는 학습이 붕괴했습니다.
- latent 8, λ=0.01만 규제 없음보다 0.002 높지만, 에폭마다 AUROC가 ±0.03씩 흔들리는 것에 비하면 의미 없는 차이입니다.
- **결론: 중심 규제 loss는 성능을 올리지 못했습니다.**

## 5. 왜 안 됐나: scale collapse

걱정했던 scale collapse가 실제로 일어났습니다.

| latent 2 | λ=0 | λ=0.01 | λ=0.1 | λ=1 |
|---|---|---|---|---|
| latent 퍼짐 (std) | 2.93 | 0.23 | 0.05 | 0.02 → 0 |
| train 재구성 오차 | 0.028 | 0.031 | 0.037 | 0.04~0.06 |

1. **encoder가 latent를 모은 게 아니라 크기 자체를 줄여서 규제를 피해 갔습니다.** 모든 latent를 작게 만들고 decoder가 다시 키우면, 거리 항은 0에 가까워지지만 실제로는 아무것도 모이지 않습니다. 거리 항 ‖z − c‖²가 latent의 **크기(scale)에 의존**하기 때문에 생기는 문제입니다.
2. **그러다 보니 정상 이미지 복원까지 나빠졌습니다.** λ가 커질수록 train 재구성 오차가 0.028에서 0.037 이상으로 올랐습니다. 정상의 오차가 커지면 정상과 비정상의 오차 차이가 줄어서 AUROC가 떨어집니다.
3. **λ=1에서는 완전히 붕괴했습니다.** latent 8, λ=1에서는 std가 0이 되면서 모든 입력이 같은 출력(평균 이미지)으로 복원됐습니다. 이 경우 거리 점수가 전부 0이라 거리 점수를 표준화하는 계산에서 0으로 나누게 되어, 실험 스크립트가 중간에 멈췄습니다.

### 이 실험에서 얻은 교훈

- latent를 모으는 규제는 **크기를 줄여서는 피할 수 없는 형태**여야 합니다.
- 규제 때문에 **정상 복원 능력이 망가지면 안 됩니다.**

이후 실험에서 이 두 원칙을 바탕으로 VAE(KL 규제)를 시도했습니다. 최종적으로는 latent를 직접 규제하는 대신, 모델 용량을 줄이고 점수와 학습 기준을 맞추는 방향(SSIM + perceptual loss)으로 성능을 올렸습니다.

## 6. 코드

### 결과를 만든 원래 실험 코드

[scripts/center_loss_experiment.py](../scripts/center_loss_experiment.py)가 위 표를 만든 코드입니다.

```bash
python scripts/center_loss_experiment.py 8 0.01   # latent 8, λ = 0.01
```

### 학습 파이프라인 옵션

같은 loss를 `src/` 학습 파이프라인에서도 `--center_weight` 옵션으로 쓸 수 있습니다. 구현 위치는 다음과 같습니다.

- [src/trainer.py](../src/trainer.py): `Trainer.latent_center`가 c를 계산하고, `Trainer.train`에서 `loss = criterion + center_weight × ||z − c||²`를 만듭니다.
- [src/models.py](../src/models.py): latent를 꺼내는 `encode`와 복원하는 `decode`를 씁니다.

```bash
python src/main.py --models Autoencoder --latent_dim 8 --criterion MSELoss --center_weight 0.01 \
  --optimizer Adam --lr 0.005 --batch_size 1 --epochs 5 --score mse --threshold 0.01 \
  --seed 0 --log_dir logs/center_loss
```

파이프라인은 데이터를 섞는 방식이 원래 실험 코드와 달라서, 같은 시드라도 수치가 ±0.03 정도 다르게 나올 수 있습니다.

### 제출 노트북과의 관계

이 loss는 **학습할 때만** 쓰입니다. 이상 점수는 재구성 오차(MSE)라서, 이 방법으로 학습한 weight도 기존 제출 노트북 생성기로 바로 제출 양식을 만들 수 있습니다(FAQ Q10: 학습 코드는 제출하지 않아도 됨).

```bash
python scripts/make_submission.py --model Autoencoder --latent_dim 8 --score mse --threshold 0.01 \
    --weight logs/center_loss/Autoencoder/checkpoint/epoch5_weight.pth \
    --check_csv logs/center_loss/Autoencoder/record/epoch5_result_team_2.csv \
    --train_note "Training loss = MSE + 0.01 * ||z - c||^2 (c = mean train latent, recomputed each epoch)"
```

**규정:** 학습에는 2번 라벨 train 데이터만 쓰고, c도 train 데이터로만 계산합니다. 점수는 재구성 오차이므로 재구성 기반 규칙(FAQ Q6)에 맞습니다.

## 참고 문헌

- Ruff et al., "Deep One-Class Classification", ICML 2018 (Deep SVDD)
- Hojjati & Armanfard, "DASVDD: Deep Autoencoding Support Vector Data Descriptor for Anomaly Detection", IEEE TKDE 2023
