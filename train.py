import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# -------------------------------
# المسارات
# -------------------------------
base_path = r"C:\Users\lenovo\Desktop\wall_def"
train_dir = os.path.join(base_path, "train")
val_dir = os.path.join(base_path, "val")
test_dir = os.path.join(base_path, "test")

# -------------------------------
# Data Augmentation (قوي 🔥)
# -------------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3]
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=16,  # صغرناها 🔥
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    shuffle=False
)

# -------------------------------
# Calculate Class Weights
# -------------------------------
num_samples = train_generator.samples
num_classes = train_generator.num_classes
class_weights_dict = {}
for cls_idx in range(num_classes):
    cls_count = np.sum(train_generator.classes == cls_idx)
    weight = num_samples / (num_classes * max(1, cls_count))
    class_weights_dict[cls_idx] = weight
print(f"Computed Class Weights: {class_weights_dict}")

# -------------------------------
# MobileNetV2
# -------------------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# -------------------------------
# طبقات إضافية
# -------------------------------
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)  # طبقة إضافية 🔥
x = Dropout(0.2)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# -------------------------------
# Compile
# -------------------------------
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# Callbacks 🔥
# -------------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=3
)

# -------------------------------
# التدريب (المرحلة الأولى)
# -------------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weights_dict
)

# -------------------------------
# Fine-tuning 🔥🔥
# -------------------------------
base_model.trainable = True

# Unfreeze the last 50 layers instead of just 20 to let the model learn more specific patterns
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.00001),  # أصغر 🔥
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weights_dict
)

# -------------------------------
# التقييم
# -------------------------------
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# -------------------------------
# حفظ
# -------------------------------
model.save(r"C:\Users\lenovo\Desktop\wall_def\wall_defects_model.keras")