from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import os

# -------------------------------
# تحميل الموديل المدرب
# -------------------------------
model_path = r"C:\Users\lenovo\Desktop\wall_def\wall_defects_model.keras"
model = load_model(model_path)

# -------------------------------
# مسار الصورة الجديدة داخل مجلد wall_def
# -------------------------------
img_path = r"C:\Users\lenovo\Desktop\wall_def\test_image\ww.png"

# تجهيز الصورة
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# التنبؤ بالفئة
pred = model.predict(img_array)
# أسماء الفئات بالترتيب نفسه اللي اتدربت عليه
class_labels = ['crack', 'moisture', 'normal']  
class_idx = np.argmax(pred[0])

print(f"Prediction: {class_labels[class_idx]} OK")