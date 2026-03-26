from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# -------------------------------
# مسار الموديل المدرب
# -------------------------------
model_path = r"C:\Users\lenovo\Desktop\wall_def\wall_defects_model.h5"

# التأكد من وجود الموديل
if not os.path.exists(model_path):
    raise FileNotFoundError("الموديل غير موجود 😅 شغلي train.py أولاً")

# تحميل الموديل
model = load_model(model_path)

# -------------------------------
# مسار الصورة الجديدة داخل wall_def
# -------------------------------
img_name = "try1.jpg"  # غيري الاسم حسب صورتك
img_path = os.path.join(r"C:\Users\lenovo\Desktop\wall_def", img_name)

# التأكد من وجود الصورة
if not os.path.exists(img_path):
    raise FileNotFoundError(f"الصورة '{img_name}' غير موجودة داخل wall_def 😅")

# تجهيز الصورة
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# التنبؤ بالفئة
pred = model.predict(img_array)

# أسماء الفئات بالترتيب نفسه اللي اتدربت عليه
class_labels = ['crack', 'moisture', 'normal']  # هذا نفس ترتيب مجلداتك
class_idx = np.argmax(pred[0])

print(f"Prediction: {class_labels[class_idx]} ✅")