"""
============================================================
  플라스틱 병 재활용 분류기 — Revised Confusion Matrix
  최종 모델(VGG16 Fine-tuning) 성능 분석

[프로젝트 개요]
  플라스틱 PET 병을 3클래스로 분류하는 CNN 분류기
  - bad       : 색상·오염·비정형 등으로 재활용 어려운 병
  - no_label  : 투명 순수 PET 병 → 즉시 재활용 가능
  - with_label: 투명 PET + 유색 스티커 → 라벨 제거 후 재활용

[원본 대비 변경 사항]
  - DATA_DIR   : Windows 경로 → Linux 로컬 dataset 경로로 수정
  - MODEL_PATH : revised 파이프라인의 최종 모델 참조
  - CLASS_NAMES: color_label → bad (실제 데이터셋 폴더명 반영)
  - 폰트       : Malgun Gothic → D2Coding (Linux 환경)
  - 주석·문서  : 현재 프로젝트 상황에 맞게 전면 업데이트
============================================================
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    print(f"   폰트명: {_font_name}")
else:
    print("⚠️ D2Coding 폰트 미발견 — 기본 폰트 사용")
    print("   설치: sudo apt install fonts-d2coding")
plt.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────────────────
# [단계 1] 설정
# ─────────────────────────────────────────────────────
IMG_SIZE   = 224      # VGG16 표준 입력 크기
BATCH_SIZE = 32

# 현재 프로젝트의 로컬 데이터셋 경로
DATA_DIR = os.path.join(os.path.dirname(__file__), 'dataset', 'dataset-matched')

# Fine-tuning 단계에서 저장된 최종 모델
MODEL_PATH = 'best_revised_vgg16_finetuned.keras'

# 클래스 정보 (실제 데이터셋 폴더명 기준)
CLASS_NAMES = ['bad', 'no_label', 'with_label']
CLASS_KOR   = ['재활용 불가', '즉시 재활용', '라벨 제거 후 재활용']


# ─────────────────────────────────────────────────────
# [단계 2] 모델 및 데이터 로드
#
#  ⚠️ Confusion Matrix는 반드시 Val 데이터로 평가
#     - shuffle=False : 이미지 순서 고정 (예측값과 정답 순서 맞추기 위해)
#     - augmentation 없음 : 실제 성능을 있는 그대로 측정
# ─────────────────────────────────────────────────────
print("📥 최종 모델 로드 중...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ 모델 로드 완료\n")

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

val_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,    # ⚠️ 반드시 False (순서 고정)
    seed=42
)

print(f"📂 클래스 인덱스: {val_generator.class_indices}")
print(f"   Val 샘플 수: {val_generator.samples}장\n")


# ─────────────────────────────────────────────────────
# [단계 3] 예측 수행
# ─────────────────────────────────────────────────────
print("🔍 Val 데이터 전체 예측 중...")
y_pred_prob = model.predict(val_generator, verbose=1)

# 확률 → 클래스 인덱스 변환
y_pred = np.argmax(y_pred_prob, axis=1)   # 예측 클래스
y_true = val_generator.classes             # 실제 클래스

print(f"\n예측 완료: {len(y_pred)}장")


# ─────────────────────────────────────────────────────
# [단계 4] Confusion Matrix 계산
# ─────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)

# 정규화된 Confusion Matrix (비율, %)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100


# ─────────────────────────────────────────────────────
# [단계 5] 시각화
# ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Confusion Matrix — VGG16 Fine-tuning 최종 모델', fontsize=13, y=1.02)

# ── 왼쪽: 실제 개수 ──────────────────────────────────
ax1 = axes[0]
im1 = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar(im1, ax=ax1)

ax1.set_title('실제 개수', fontsize=12, pad=12)
ax1.set_xticks(range(3))
ax1.set_yticks(range(3))
ax1.set_xticklabels(CLASS_KOR, fontsize=10)
ax1.set_yticklabels(CLASS_KOR, fontsize=10)
ax1.set_xlabel('예측 클래스', fontsize=11)
ax1.set_ylabel('실제 클래스', fontsize=11)

thresh1 = cm.max() / 2
for i in range(3):
    for j in range(3):
        color = 'white' if cm[i, j] > thresh1 else 'black'
        ax1.text(j, i, f'{cm[i, j]}',
                 ha='center', va='center',
                 color=color, fontsize=13, fontweight='bold')

# ── 오른쪽: 퍼센트 ───────────────────────────────────
ax2 = axes[1]
im2 = ax2.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=100)
plt.colorbar(im2, ax=ax2, format='%.0f%%')

ax2.set_title('비율 (%)', fontsize=12, pad=12)
ax2.set_xticks(range(3))
ax2.set_yticks(range(3))
ax2.set_xticklabels(CLASS_KOR, fontsize=10)
ax2.set_yticklabels(CLASS_KOR, fontsize=10)
ax2.set_xlabel('예측 클래스', fontsize=11)
ax2.set_ylabel('실제 클래스', fontsize=11)

thresh2 = 50
for i in range(3):
    for j in range(3):
        color = 'white' if cm_norm[i, j] > thresh2 else 'black'
        ax2.text(j, i, f'{cm_norm[i, j]:.1f}%',
                 ha='center', va='center',
                 color=color, fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('revised_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()


# ─────────────────────────────────────────────────────
# [단계 6] 클래스별 상세 성능 출력
#
#  Precision : 모델이 "이 클래스다"라고 예측했을 때 실제로 맞은 비율
#  Recall    : 실제 그 클래스인 것 중 모델이 맞게 찾아낸 비율
#  F1-score  : Precision과 Recall의 조화 평균
# ─────────────────────────────────────────────────────
print(f"\n{'='*58}")
print(f"  클래스별 상세 성능 리포트")
print(f"{'='*58}")
print(classification_report(
    y_true, y_pred,
    target_names=CLASS_KOR,
    digits=4
))

# 클래스별 정확도 개별 출력
print(f"{'='*58}")
print(f"  클래스별 개별 정확도")
print(f"{'='*58}")
for i, (eng, kor) in enumerate(zip(CLASS_NAMES, CLASS_KOR)):
    correct = cm[i, i]
    total   = cm[i].sum()
    acc     = correct / total * 100
    wrong   = total - correct
    print(f"  {kor:<14} ({eng})")
    print(f"    맞음: {correct:4d}장  |  틀림: {wrong:4d}장  |  정확도: {acc:.2f}%")
    if wrong > 0:
        # 어디로 잘못 분류했는지 출력
        for j in range(3):
            if i != j and cm[i, j] > 0:
                print(f"    └ '{CLASS_KOR[j]}'(으)로 잘못 분류: {cm[i, j]}장")
    print()

# 전체 요약
overall_acc = np.trace(cm) / cm.sum() * 100
print(f"{'='*58}")
print(f"  전체 Val Accuracy: {overall_acc:.2f}%")
print(f"  올바르게 분류:     {np.trace(cm)}장 / {cm.sum()}장")
print(f"  저장 완료:         revised_confusion_matrix.png")
print(f"{'='*58}")