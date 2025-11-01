import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from vit_keras import vit
from PIL import Image
import numpy as np
import streamlit as st # Thêm st để dùng @st.cache_resource

# Định nghĩa Hằng số
IMG_SHAPE = (224, 224, 3)
IMG_SIZE = (224, 224)
# CẬP NHẬT ĐÚNG TÊN CÁC LỚP CỦA BẠN THEO THỨ TỰ
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
NUM_CLASSES = len(CLASS_NAMES)

# Hàm tiền xử lý
# ViT rescale 0-1
def vit_preprocess(image_np):
    """Tiền xử lý ảnh numpy cho ViT."""
    image_np = tf.image.resize(image_np, IMG_SIZE)
    image_np = image_np / 255.
    return image_np

# VGG
def vgg_preprocess_image(image_np):
    """Tiền xử lý ảnh numpy cho VGG."""
    image_np = tf.image.resize(image_np, IMG_SIZE)
    image_np = vgg_preprocess(image_np)
    return image_np

# ResNet
def resnet_preprocess_image(image_np):
    """Tiền xử lý ảnh numpy cho ResNet."""
    image_np = tf.image.resize(image_np, IMG_SIZE)
    image_np = resnet_preprocess(image_np)
    return image_np

# Hàm xây dựng lại mô hình
def build_model(base_model_func, model_name):
    """Xây dựng lại mô hình từ base và thêm các lớp FC."""
    base_model = base_model_func(weights=None, include_top=False, input_shape=IMG_SHAPE)
    base_model.trainable = False # Không cần thiết khi predict, nhưng để cho an toàn

    x = base_model.output
    if len(x.shape) > 2:
        x = GlobalAveragePooling2D()(x)
        
    x = Dense(256, activation='relu')(x)
    preds = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=preds)
    
    # Load trọng số đã lưu
    try:
        model.load_weights(f'models/{model_name}.weights.h5')
    except Exception as e:
        st.error(f"Lỗi khi tải trọng số cho {model_name}: {e}")
        st.error(f"Đảm bảo file 'models/{model_name}.weights.h5' tồn tại.")
        return None
        
    return model

def build_vit_model(model_name="vit_b16"):
    """Hàm xây dựng riêng cho ViT."""
    base_model = vit.vit_b16(
        image_size=IMG_SIZE[0],
        activation='softmax',
        pretrained=False, # Không cần tải lại từ imagenet
        include_top=False,
        pretrained_top=False
    )
    base_model.trainable = False

    x = base_model.output
    x = Dense(256, activation='relu')(x)
    preds = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=preds)
    
    try:
        model.load_weights(f'models/{model_name}.weights.h5')
    except Exception as e:
        st.error(f"Lỗi khi tải trọng số cho {model_name}: {e}")
        st.error(f"Đảm bảo file 'models/{model_name}.weights.h5' tồn tại.")
        return None
        
    return model

# Hàm Load mô hình (với cache)
# @st.cache_resource giúp Streamlit không cần load lại mô hình mỗi lần
# người dùng tương tác với UI, tiết kiệm rất nhiều thời gian.
@st.cache_resource
def load_all_models():
    """Tải 3 mô hình và cache lại."""
    print("Đang tải mô hình VGG16...")
    model_vgg = build_model(VGG16, "vgg16")
    
    print("Đang tải mô hình ResNet50...")
    model_resnet = build_model(ResNet50, "resnet50")
    
    print("Đang tải mô hình ViT_B16...")
    model_vit = build_vit_model("vit_b16") # Giả sử tên file là vit_b16.weights.h5
    
    return model_vgg, model_resnet, model_vit

# Hàm dự đoán chính
def predict_image(image_bytes, model_vgg, model_resnet, model_vit):
    """
    Nhận ảnh (dưới dạng bytes), tiền xử lý và trả về dự đoán từ 3 mô hình.
    """
    
    # Đọc ảnh từ bytes
    img = Image.open(image_bytes).convert('RGB')
    img_np = np.array(img) # Chuyển sang numpy array
    
    # Tiền xử lý riêng biệt cho từng mô hình
    # Thêm batch dimension (1, 224, 224, 3)
    img_vgg = np.expand_dims(vgg_preprocess_image(img_np), axis=0)
    img_resnet = np.expand_dims(resnet_preprocess_image(img_np), axis=0)
    img_vit = np.expand_dims(vit_preprocess(img_np), axis=0)

    # Dự đoán
    pred_vgg = model_vgg.predict(img_vgg, verbose=0)
    pred_resnet = model_resnet.predict(img_resnet, verbose=0)
    pred_vit = model_vit.predict(img_vit, verbose=0)
    
    # Định dạng kết quả
    results = {
        "VGG16": {
            "class": CLASS_NAMES[np.argmax(pred_vgg)],
            "confidence": np.max(pred_vgg) * 100
        },
        "ResNet50": {
            "class": CLASS_NAMES[np.argmax(pred_resnet)],
            "confidence": np.max(pred_resnet) * 100
        },
        "ViT-B16": {
            "class": CLASS_NAMES[np.argmax(pred_vit)],
            "confidence": np.max(pred_vit) * 100
        }
    }
    
    return img, results
