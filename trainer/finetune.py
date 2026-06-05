"""
============================================================
  플라스틱 병 재활용 분류기 — Fine-tuning
  ✅ InceptionResNetV2 전이학습 — Fine-tuning (상위 블록 해동)

[프로젝트 개요]
  플라스틱 PET 병을 3클래스로 분류하는 CNN 분류기
  - bad       : 색상·오염·비정형 등으로 재활용 어려운 병
  - no_label  : 투명 순수 PET 병 → 즉시 재활용 가능
  - with_label: 투명 PET + 유색 스티커 → 라벨 제거 후 재활용

[전략]
  Feature Extraction: InceptionResNetV2 전체 동결 → 분류기만 학습
  Fine-tuning       : InceptionResNetV2 상위 ~40% 해동 → 미세조정

[왜 상위 블록만 해동하나?]
  하위 블록: 선, 엣지, 색상 등 범용적 저수준 특징 → 동결 유지
  상위 블록: 복잡한 형태, 질감 등 고수준 특징     → 해동하여 재조정
            PET 병 라벨, 오염, 형태 특징에 맞게 조정 가능

[핵심: 학습률을 반드시 낮춰야 함]
  Feature Extraction 학습률: 1e-4
  Fine-tuning 학습률       : 1e-5  (10배 낮춤)
  이유: 사전학습 가중치를 우리 데이터로 덮어쓰면 안 됨

[변경 사항 (이전 VGG16 대비)]
  - 베이스 모델    : VGG16 → InceptionResNetV2 (299×299)
  - 데이터         : 균형화 데이터셋 (17,343장) 사용
  - 학습률 스케줄  : ReduceLROnPlateau → Warmup + Cosine Decay
  - 손실 함수      : Label Smoothing (0.1) 적용
  - 사전처리       : rescale=1/255 → InceptionResNetV2 preprocess_input
============================================================
"""

import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.losses import CategoricalCrossentropy
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ─────────────────────────────────────────────────────
# [폰트 설정] D2Coding 폰트 (한글 깨짐 방지)
# ─────────────────────────────────────────────────────
_D2CODING_CANDIDATES = [
    '/usr/share/fonts/truetype/naver-d2coding/D2Coding-Ver1.3.2-20180524-all.ttc',
    '/usr/local/share/fonts/d/D2CodingLigatureNerdFont_Regular.ttf',
    '/usr/local/share/fonts/d/D2CodingLigatureNerdFontPropo_Regular.ttf',
    '/usr/share/fonts/truetype/d2coding/D2Coding.ttf',
    '/usr/share/fonts/D2Coding.ttf',
]
_D2CODING_PATH = None
for _candidate in _D2CODING_CANDIDATES:
    if os.path.exists(_candidate):
        _D2CODING_PATH = _candidate
        break

if _D2CODING_PATH:
    fm.fontManager.addfont(_D2CODING_PATH)
    _d2coding_font = fm.FontProperties(fname=_D2CODING_PATH)
    _font_name = _d2coding_font.get_name()
    plt.rcParams['font.family'] = _font_name
    print(f"✅ D2Coding 폰트 로드: {_D2CODING_PATH}")
else:
    print("⚠️ D2Coding 폰트 미발견 — 기본 폰트 사용")
plt.rcParams['axes.unicode_minus'] = False


# ═══════════════════════════════════════════════════════
#   단계 1: 하이퍼파라미터
# ═══════════════════════════════════════════════════════
IMG_SIZE    = 299       # InceptionResNetV2 표준 입력 크기
BATCH_SIZE  = 32
EPOCHS      = 40        # Fine-tuning은 Feature Extraction보다 짧게

# ── 경로 ──
BALANCED_DIR = os.path.join(os.path.dirname(__file__), 'dataset-balanced')
PREPARE_WEIGHTS = 'best_prepare.keras'       # Feature Extraction 가중치
SAVE_PATH       = 'best_finetune.keras'
LOG_PATH        = 'finetune_log.csv'

# ── 클래스 ──
NUM_CLASSES = 3
CLASS_NAMES = ['bad', 'no_label', 'with_label']

# ── 균형화된 데이터 총량 ──
TOTAL_BALANCED = 17343   # 5,781 × 3

# ── 해동 설정 ──
# InceptionResNetV2: 총 618개 레이어
# 상위 ~40% 해동 → 약 250개 레이어 해동 (약 370번 레이어부터)
UNFREEZE_RATIO = 0.60    # 하위 60% 동결, 상위 40% 해동


# ═══════════════════════════════════════════════════════
#   단계 2: Warmup + Cosine Decay 학습률 스케줄러
# ═══════════════════════════════════════════════════════
class WarmupCosineDecay(tf.keras.callbacks.Callback):
    """
    학습 초기 warmup → cosine decay 스케줄러.

    Fine-tuning에서는 초기 학습률이 낮으므로 warmup을 짧게 설정.
    """

    def __init__(self, warmup_epochs, initial_lr, min_lr=1e-8):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_epochs = None

    def on_train_begin(self, logs=None):
        if self.total_epochs is None:
            self.total_epochs = self.params.get('epochs', 40)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        print(f"  → 학습률: {lr:.2e}")


# ═══════════════════════════════════════════════════════
#   단계 3: 데이터 제너레이터 (균형화 데이터셋 사용)
# ═══════════════════════════════════════════════════════
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest',
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
)

train_generator = train_datagen.flow_from_directory(
    BALANCED_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42,
)

val_generator = val_datagen.flow_from_directory(
    BALANCED_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42,
)

print(f"\n📂 클래스 인덱스: {train_generator.class_indices}")
print(f"   Train: {train_generator.samples:,}장")
print(f"   Val  : {val_generator.samples:,}장\n")


# ═══════════════════════════════════════════════════════
#   단계 4: Feature Extraction 모델 로드 + 해동
# ═══════════════════════════════════════════════════════
print("📥 Feature Extraction 최적 가중치 로드 중...")
model = tf.keras.models.load_model(PREPARE_WEIGHTS)
print("✅ 모델 로드 완료\n")

# Sequential 모델에서 InceptionResNetV2 베이스 추출
base_model = model.layers[0]

# 레이어 수 확인
total_layers = len(base_model.layers)
freeze_from = int(total_layers * UNFREEZE_RATIO)
print(f"   InceptionResNetV2 총 레이어: {total_layers}개")
print(f"   해동 기준: 인덱스 {freeze_from} 이후 (상위 ~{(1-UNFREEZE_RATIO)*100:.0f}%)")

# 해동 설정
base_model.trainable = True
for i, layer in enumerate(base_model.layers):
    if i < freeze_from:
        layer.trainable = False    # 하위 블록 동결 유지
    else:
        layer.trainable = True     # 상위 블록 해동

# 해동 결과 확인
frozen   = sum(1 for l in base_model.layers if not l.trainable)
unfrozen = sum(1 for l in base_model.layers if l.trainable)
print(f"   동결 레이어: {frozen}개")
print(f"   해동 레이어: {unfrozen}개\n")


# ═══════════════════════════════════════════════════════
#   단계 5: 재컴파일
# ═══════════════════════════════════════════════════════
INITIAL_LR = 1e-5   # Feature Extraction(1e-4)의 1/10

model.compile(
    optimizer=optimizers.Adam(learning_rate=INITIAL_LR),
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy'],
)

# 파라미터 수 확인
trainable_params     = sum(tf.size(w).numpy() for w in model.trainable_weights)
non_trainable_params = sum(tf.size(w).numpy() for w in model.non_trainable_weights)

print(f"{'=' * 52}")
print(f"  Fine-tuning 파라미터 구성")
print(f"  동결 파라미터 (백본 하위) : {non_trainable_params:>12,}개")
print(f"  학습 파라미터 (백본 상위  ")
print(f"                    + 분류기): {trainable_params:>12,}개")
print(f"  Feature Extraction LR: 1e-4  →  Fine-tuning LR: 1e-5")
print(f"{'=' * 52}\n")


# ═══════════════════════════════════════════════════════
#   단계 6: 콜백
# ═══════════════════════════════════════════════════════
lr_scheduler = WarmupCosineDecay(
    warmup_epochs=3,
    initial_lr=INITIAL_LR,
    min_lr=1e-8,
)

callbacks = [
    lr_scheduler,
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1,
    ),
    ModelCheckpoint(
        filepath=SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
    ),
    CSVLogger(
        LOG_PATH,
        append=False,
        separator=',',
    ),
]


# ═══════════════════════════════════════════════════════
#   단계 7: 학습
# ═══════════════════════════════════════════════════════
print("🏋️ Fine-tuning 학습 시작!")
print(f"   백본    : InceptionResNetV2 (하위 {frozen}개 동결 / 상위 {unfrozen}개 해동)")
print(f"   분류기  : GAP → Dense(1024) → Drop(0.5) → Dense(512) → Drop(0.4) → Dense(3)")
print(f"   스케줄  : Warmup(3에포크) + Cosine Decay")
print(f"   손실    : CategoricalCrossentropy (label_smoothing=0.1)")
print(f"   데이터  : 균형화 {TOTAL_BALANCED:,}장 (Train {train_generator.samples:,} / Val {val_generator.samples:,})\n")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
)


# ═══════════════════════════════════════════════════════
#   단계 8: 시각화
# ═══════════════════════════════════════════════════════
actual_epochs = len(history.history['accuracy'])
best_val_acc  = max(history.history['val_accuracy'])
best_epoch    = history.history['val_accuracy'].index(best_val_acc) + 1

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f'InceptionResNetV2 Fine-tuning (상위 {(1-UNFREEZE_RATIO)*100:.0f}% 해동)\n'
    f'데이터: 균형화 {TOTAL_BALANCED:,}장 | '
    f'학습: {actual_epochs}에포크 | '
    f'최고 Val Acc: {best_val_acc:.4f} (Epoch {best_epoch})',
    fontsize=12,
)

axes[0].plot(history.history['accuracy'],     label='Train', color='steelblue', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Val',   color='tomato',    linewidth=2)
axes[0].axvline(x=best_epoch - 1, color='gray', linestyle='--', alpha=0.5, label=f'Best epoch {best_epoch}')
axes[0].set_title('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'],     label='Train', color='steelblue', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Val',   color='tomato',    linewidth=2)
axes[1].axvline(x=best_epoch - 1, color='gray', linestyle='--', alpha=0.5, label=f'Best epoch {best_epoch}')
axes[1].set_title('Loss (label_smoothing=0.1)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('finetune.png', dpi=150)
plt.show()


# ═══════════════════════════════════════════════════════
#   단계 9: 학습 결과 요약
# ═══════════════════════════════════════════════════════
print(f"\n{'=' * 52}")
print(f"  ✅ Fine-tuning 완료!")
print(f"  백본              : InceptionResNetV2")
print(f"  해동 레이어       : {unfrozen}개 (상위 {(1-UNFREEZE_RATIO)*100:.0f}%)")
print(f"  데이터 (균형화)   : {TOTAL_BALANCED:,}장")
print(f"  실제 학습 에포크  : {actual_epochs} / {EPOCHS}")
print(f"  최고 Val Accuracy : {best_val_acc:.4f} (Epoch {best_epoch})")
print(f"  모델 저장         : {SAVE_PATH}")
print(f"  학습 로그         : {LOG_PATH}")
print(f"{'=' * 52}\n")

print(f"""
📋 Fine-tuning 결과 체크리스트

  ✅ Val Acc가 Feature Extraction(81%)보다 높다
       → Fine-tuning 성공! 최종 모델로 확정

  ⚠️  Val Acc가 Feature Extraction과 비슷하다
       → Fine-tuning 효과 미미. 현재 성능이 한계일 수 있음
          Feature Extraction 모델을 최종 모델로 사용 권장

  ❌ Val Acc가 Feature Extraction보다 낮아졌다
       → 학습률이 너무 높아 사전학습 가중치 손상
          learning_rate=1e-5 → 1e-6으로 낮추고 재시도
""")