"""
============================================================
  플라스틱 병 재활용 분류기 — Prepare (Feature Extraction)
  ✅ InceptionResNetV2 전이학습 + 데이터 증강 균형화

[프로젝트 개요]
  플라스틱 PET 병을 3클래스로 분류하는 CNN 분류기
  - bad       : 색상·오염·비정형 등으로 재활용 어려운 병
  - no_label  : 투명 순수 PET 병 → 즉시 재활용 가능
  - with_label: 투명 PET + 유색 스티커 → 라벨 제거 후 재활용

[데이터셋 구조 (서브클래스 포함)]
  dataset/bad/colored  : 1,927장  ← 기준치 (증강 불필요)
  dataset/bad/dirty    :   493장  ← 1,927장까지 증강
  dataset/bad/not_bottle:  401장  ← 1,927장까지 증강
  dataset/no_label     : 2,530장  ← 5,781장까지 증강 (1,927 × 3)
  dataset/with_label   :16,703장  ← 5,781장만 사용 (1,927 × 3)

[균형화 전략]
  bad 클래스 총합   : 1,927 × 3 = 5,781장
  no_label          : 5,781장 (증강으로 확보)
  with_label        : 5,781장 (랜덤 샘플링으로 제한)
  → 전체 학습 데이터: 5,781 × 3 = 17,343장

[변경 사항 (VGG16 대비)]
  - 베이스 모델    : VGG16 → InceptionResNetV2 (299×299 입력)
  - 데이터 증강    : 소수 클래스 증강 + 다수 클래스 샘플링
  - 학습률 스케줄  : ReduceLROnPlateau → Warmup + Cosine Decay
  - 손실 함수      : CategoricalCrossentropy → Label Smoothing (0.1)
  - Dropout        : 0.3 → 0.5 / 0.4 (과적합 억제 강화)
  - 사전처리       : rescale=1/255 → InceptionResNetV2 preprocess_input
============================================================
"""

import os
import math
import random
import shutil
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, save_img
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
IMG_SIZE    = 299      # InceptionResNetV2 표준 입력 크기 — 변경 금지
BATCH_SIZE  = 32
EPOCHS      = 60       # 데이터 증가(17,343장)에 따라 충분한 에포크 확보

# ── 데이터셋 경로 ──
DATA_DIR     = os.path.join(os.path.dirname(__file__), 'dataset')
BALANCED_DIR = os.path.join(os.path.dirname(__file__), 'dataset-balanced')

# ── 저장 경로 ──
SAVE_PATH = 'best_prepare.keras'
LOG_PATH  = 'prepare_log.csv'

# ── 클래스 정보 ──
NUM_CLASSES = 3
CLASS_NAMES = ['bad', 'no_label', 'with_label']

# ── 데이터 균형화 기준 (bad/colored = 1,927장) ──
TARGET_PER_SUBCLASS = 1927                # bad 서브클래스별 목표치
TARGET_BAD          = TARGET_PER_SUBCLASS * 3   # 5,781
TARGET_NO_LABEL     = TARGET_BAD                 # 5,781
TARGET_WITH_LABEL   = TARGET_BAD                 # 5,781
TOTAL_BALANCED      = TARGET_BAD + TARGET_NO_LABEL + TARGET_WITH_LABEL  # 17,343


# ═══════════════════════════════════════════════════════
#   단계 2: 데이터 균형화 (증강 + 샘플링)
# ═══════════════════════════════════════════════════════
print("=" * 58)
print("  📊 원본 데이터셋 현황")
print("=" * 58)

# 원본 파일 수 확인
ORIG_COUNTS = {}
for cls in ['bad/colored', 'bad/dirty', 'bad/not_bottle', 'no_label', 'with_label']:
    src = os.path.join(DATA_DIR, cls)
    count = len(glob.glob(os.path.join(src, '*.jpg')))
    ORIG_COUNTS[cls] = count
    print(f"  {cls:<20s}: {count:,}장")

print(f"\n  총 {sum(ORIG_COUNTS.values()):,}장")
print(f"  목표 총량         : {TOTAL_BALANCED:,}장\n")

# ── 증강용 제너레이터 (원본 품질 보존을 위한 적절한 증강) ──
aug_datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=[0.75, 1.25],
    shear_range=0.15,
    fill_mode='nearest',
)


def copy_originals(src_dir, dst_dir, limit=None, shuffle=False, prefix=''):
    """원본 이미지를 복사한다."""
    files = sorted(glob.glob(os.path.join(src_dir, '*.jpg')))
    if shuffle:
        random.shuffle(files)
    if limit:
        files = files[:limit]

    for i, f in enumerate(files):
        dst_name = f"{prefix}{i:05d}.jpg"
        shutil.copy2(f, os.path.join(dst_dir, dst_name))
    return len(files)


def augment_to_target(src_dir, dst_dir, target_count, prefix=''):
    """원본 + 증강으로 목표 수량을 확보한다."""
    files = sorted(glob.glob(os.path.join(src_dir, '*.jpg')))
    n_originals = len(files)

    # ① 원본 복사
    for i, f in enumerate(files):
        dst_name = f"{prefix}{i:05d}.jpg"
        shutil.copy2(f, os.path.join(dst_dir, dst_name))

    # ② 증강 이미지 생성
    needed = target_count - n_originals
    if needed <= 0:
        return n_originals

    print(f"     → {n_originals}장 복사 + {needed}장 증강 = {target_count}장")
    for i in range(needed):
        src_file = random.choice(files)
        img = load_img(src_file, target_size=(IMG_SIZE, IMG_SIZE))
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        aug_iter = aug_datagen.flow(x, batch_size=1, seed=random.randint(0, 2**31))
        aug_img = next(aug_iter)[0].astype(np.uint8)

        dst_name = f"{prefix}{n_originals + i:05d}.jpg"
        save_img(os.path.join(dst_dir, dst_name), aug_img)

    return target_count


def create_balanced_dataset():
    """균형화된 데이터셋 디렉토리를 생성한다."""
    if os.path.exists(BALANCED_DIR):
        shutil.rmtree(BALANCED_DIR)
    os.makedirs(os.path.join(BALANCED_DIR, 'bad'), exist_ok=True)
    os.makedirs(os.path.join(BALANCED_DIR, 'no_label'), exist_ok=True)
    os.makedirs(os.path.join(BALANCED_DIR, 'with_label'), exist_ok=True)

    counts = {}

    # ── bad/colored: 원본 1,927장 그대로 ──
    print("[bad/colored] 원본 복사 (증강 불필요)")
    src = os.path.join(DATA_DIR, 'bad', 'colored')
    dst = os.path.join(BALANCED_DIR, 'bad')
    counts['bad/colored'] = copy_originals(
        src, dst, limit=TARGET_PER_SUBCLASS, prefix='colored_'
    )

    # ── bad/dirty: 493장 → 1,927장까지 증강 ──
    print("[bad/dirty] 증강")
    src = os.path.join(DATA_DIR, 'bad', 'dirty')
    dst = os.path.join(BALANCED_DIR, 'bad')
    counts['bad/dirty'] = augment_to_target(
        src, dst, target_count=TARGET_PER_SUBCLASS, prefix='dirty_'
    )

    # ── bad/not_bottle: 401장 → 1,927장까지 증강 ──
    print("[bad/not_bottle] 증강")
    src = os.path.join(DATA_DIR, 'bad', 'not_bottle')
    dst = os.path.join(BALANCED_DIR, 'bad')
    counts['bad/not_bottle'] = augment_to_target(
        src, dst, target_count=TARGET_PER_SUBCLASS, prefix='bottle_'
    )

    # ── no_label: 2,530장 → 5,781장까지 증강 ──
    print("[no_label] 증강")
    src = os.path.join(DATA_DIR, 'no_label')
    dst = os.path.join(BALANCED_DIR, 'no_label')
    counts['no_label'] = augment_to_target(
        src, dst, target_count=TARGET_NO_LABEL, prefix='nl_'
    )

    # ── with_label: 16,703장 중 5,781장 랜덤 샘플링 ──
    print("[with_label] 랜덤 샘플링")
    src = os.path.join(DATA_DIR, 'with_label')
    dst = os.path.join(BALANCED_DIR, 'with_label')
    counts['with_label'] = copy_originals(
        src, dst, limit=TARGET_WITH_LABEL, shuffle=True, prefix='wl_'
    )

    # ── 결과 출력 ──
    bad_total = counts['bad/colored'] + counts['bad/dirty'] + counts['bad/not_bottle']
    print(f"\n{'=' * 58}")
    print(f"  📊 균형화된 데이터셋")
    print(f"{'=' * 58}")
    print(f"  bad (서브클래스 합) : {bad_total:>7,}장")
    print(f"    - colored         : {counts['bad/colored']:>7,}장")
    print(f"    - dirty (증강)    : {counts['bad/dirty']:>7,}장")
    print(f"    - not_bottle (증강): {counts['bad/not_bottle']:>7,}장")
    print(f"  no_label (증강)     : {counts['no_label']:>7,}장")
    print(f"  with_label (샘플링)  : {counts['with_label']:>7,}장")
    print(f"  {'─' * 40}")
    print(f"  총합                : {sum(counts.values()):>7,}장")
    print(f"{'=' * 58}\n")

    return counts


# ── 균형화 실행 ──
print("\n🔄 데이터 균형화 시작...\n")
_ = create_balanced_dataset()
print("✅ 데이터 균형화 완료\n")


# ═══════════════════════════════════════════════════════
#   단계 3: 데이터 제너레이터
# ═══════════════════════════════════════════════════════
#  ⚠️ InceptionResNetV2는 preprocess_input으로 [-1, 1] 스케일링 필요
#     rescale=1./255 대신 preprocessing_function 사용

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
#   단계 4: Warmup + Cosine Decay 학습률 스케줄러
# ═══════════════════════════════════════════════════════
class WarmupCosineDecay(tf.keras.callbacks.Callback):
    """
    학습 초기에는 warmup으로 천천히 학습률을 올리고,
    이후 cosine 함수로 서서히 줄이는 스케줄러.

    Args:
        warmup_epochs: warmup 기간 (에포크 수)
        initial_lr    : 최대 학습률
        min_lr        : 최소 학습률 (cosine decay 끝점)
    """

    def __init__(self, warmup_epochs, initial_lr, min_lr=1e-7):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_epochs = None

    def set_total_epochs(self, total_epochs):
        self.total_epochs = total_epochs

    def on_train_begin(self, logs=None):
        if self.total_epochs is None:
            self.total_epochs = self.params.get('epochs', 60)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # 선형 warmup
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            # cosine decay
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        print(f"  → 학습률: {lr:.2e}")


# ═══════════════════════════════════════════════════════
#   단계 5: InceptionResNetV2 모델 구성
# ═══════════════════════════════════════════════════════
print("📥 InceptionResNetV2 사전학습 가중치 로드 중...")
base_model = InceptionResNetV2(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)

# 동결 (Feature Extraction 단계)
base_model.trainable = False
frozen_count = sum(1 for l in base_model.layers if not l.trainable)
print(f"✅ InceptionResNetV2 레이어 {frozen_count}개 전체 동결 완료")

# ── 커스텀 분류기 ──
#   InceptionResNetV2 출력: (8, 8, 1536) → GAP → 1536 벡터
#   Dropout 0.5/0.4: 이전 VGG16(0.3) 대비 강화
#     → InceptionResNetV2는 파라미터가 더 많아 과적합 위험 큼
#     → 이전 학습 결과(train 77% > val 75%)에서도 과적합 경향 확인
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),          # (8, 8, 1536) → (1536,)

    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),                      # 과적합 방지 (강화)

    layers.Dense(512, activation='relu'),
    layers.Dropout(0.4),                      # 과적합 방지

    layers.Dense(3, activation='softmax'),     # 3클래스 출력
])

# ── 컴파일 ──
#   - Label Smoothing: 0.1 → hard target의 과신 방지, generalization 향상
#   - Adam: RMSprop 대비 안정적 수렴
INITIAL_LR = 1e-4

model.compile(
    optimizer=optimizers.Adam(learning_rate=INITIAL_LR),
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy'],
)

model.summary()

# 파라미터 수 확인
trainable_params     = sum(tf.size(w).numpy() for w in model.trainable_weights)
non_trainable_params = sum(tf.size(w).numpy() for w in model.non_trainable_weights)

print(f"\n{'=' * 52}")
print(f"  InceptionResNetV2 파라미터 구성")
print(f"  동결된 파라미터 (백본)    : {non_trainable_params:>12,}개")
print(f"  학습할 파라미터 (분류기)  : {trainable_params:>12,}개")
print(f"  → 전체의 {trainable_params / (trainable_params + non_trainable_params) * 100:.1f}%만 학습")
print(f"{'=' * 52}\n")


# ═══════════════════════════════════════════════════════
#   단계 6: 콜백
# ═══════════════════════════════════════════════════════
lr_scheduler = WarmupCosineDecay(
    warmup_epochs=5,
    initial_lr=INITIAL_LR,
    min_lr=1e-7,
)

callbacks = [
    lr_scheduler,
    EarlyStopping(
        monitor='val_loss',
        patience=12,
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
print("🏋️ Feature Extraction 학습 시작!")
print(f"   백본    : InceptionResNetV2 (동결)")
print(f"   분류기  : GAP → Dense(1024) → Drop(0.5) → Dense(512) → Drop(0.4) → Dense(3)")
print(f"   스케줄  : Warmup(5에포크) + Cosine Decay")
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
    f'InceptionResNetV2 Feature Extraction\n'
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
plt.savefig('prepare.png', dpi=150)
plt.show()


# ═══════════════════════════════════════════════════════
#   단계 9: 학습 결과 요약
# ═══════════════════════════════════════════════════════
print(f"\n{'=' * 52}")
print(f"  ✅ Feature Extraction 완료!")
print(f"  백본              : InceptionResNetV2")
print(f"  데이터 (균형화)   : {TOTAL_BALANCED:,}장")
print(f"  실제 학습 에포크  : {actual_epochs} / {EPOCHS}")
print(f"  최고 Val Accuracy : {best_val_acc:.4f} (Epoch {best_epoch})")
print(f"  모델 저장         : {SAVE_PATH}")
print(f"  학습 로그         : {LOG_PATH}")
print(f"{'=' * 52}\n")

print(f"""
📋 체크리스트
  ✅ 데이터 균형화 완료 (bad 5,781 / no_label 5,781 / with_label 5,781)
  ✅ InceptionResNetV2 Feature Extraction 학습 완료
  ✅ Warmup + Cosine Decay 스케줄 적용
  ✅ Label Smoothing (0.1) 적용
  ✅ Dropout 0.5 / 0.4 강화

→ 다음 단계: Fine-tuning (InceptionResNetV2 상위 블록 해동)
""")


# ═══════════════════════════════════════════════════════
#   단계 10: 균형화 데이터셋 정리
# ═══════════════════════════════════════════════════════
#  학습 완료 후 균형화된 데이터는 더 이상 필요 없으므로 정리
#  (디스크 공간 절약: ~1.5GB)
#  ※ 주석을 해제하면 자동으로 삭제됨
# shutil.rmtree(BALANCED_DIR, ignore_errors=True)
# print("🗑️  균형화 데이터셋 정리 완료")