import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import os

# -------------------------------
# إعداد مسارات الصور حسب مجلدك الجديد
# -------------------------------
base_path = r"C:\Users\lenovo\Desktop\wall_def"
train_dir = os.path.join(base_path, "train")
val_dir = os.path.join(base_path, "val")
test_dir = os.path.join(base_path, "test")

# -------------------------------
# توليد بيانات الصور مع augmentation للترين
# -------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# -------------------------------
# تحميل MobileNetV2 بدون الطبقة النهائية
# -------------------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # نجمد الطبقات الأساسية أولاً

# -------------------------------
# إضافة طبقات مخصصة للتصنيف
# -------------------------------
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# -------------------------------
# تجميع الموديل
# -------------------------------
model.compile(optimizer=Adam(learning_rate=0.0001),  loss='categorical_crossentropy', metrics=['accuracy'])

# -------------------------------
# تدريب الموديل
# -------------------------------
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=25
)

# -------------------------------
# تقييم على مجموعة الاختبار
# -------------------------------
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# -------------------------------
# حفظ الموديل بعد التدريب
# -------------------------------
model.save(r"C:\Users\lenovo\Desktop\wall_def\wall_defects_model.h5")