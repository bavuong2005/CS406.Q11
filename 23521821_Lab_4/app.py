import streamlit as st
from PIL import Image
import utils # Import file utils.py của chúng ta

# Cấu hình trang
st.set_page_config(
    page_title="Demo Phân loại Cảnh vật",
    page_icon="🏞️",
    layout="wide"
)

# Tải mô hình
# Sử dụng cache để không phải tải lại mô hình
with st.spinner('Đang tải các mô hình AI, vui lòng chờ...'):
    model_vgg, model_resnet, model_vit = utils.load_all_models()

st.title("Ứng dụng Demo Phân loại Cảnh vật")
st.write("""
Upload một ảnh cảnh vật tự nhiên (biển, núi, rừng, v.v.) 
để so sánh kết quả dự đoán từ 3 mô hình Deep Learning.
""")

# Giao diện Upload
uploaded_file = st.file_uploader(
    "Chọn một file ảnh", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Xử lý khi có ảnh
    
    # Hiển thị ảnh gốc
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh bạn đã upload.", use_column_width=True)
    
    # Tạo nút bấm để dự đoán
    if st.button("Bắt đầu Dự đoán"):
        
        # Gọi hàm dự đoán từ utils.py
        with st.spinner('Đang phân tích ảnh...'):
            # Cần lấy 'bytes' của file đã upload
            image_bytes = uploaded_file.getvalue()
            _, results = utils.predict_image(
                image_bytes, model_vgg, model_resnet, model_vit
            )
        
        st.subheader("Kết quả Dự đoán:")
        
        # Hiển thị kết quả
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"**Mô hình: VGG16**")
            st.metric(
                label="Dự đoán", 
                value=results["VGG16"]["class"].capitalize()
            )
            st.write(f"Độ tự tin: {results['VGG16']['confidence']:.2f}%")

        with col2:
            st.success(f"**Mô hình: ResNet50**")
            st.metric(
                label="Dự đoán", 
                value=results["ResNet50"]["class"].capitalize()
            )
            st.write(f"Độ tự tin: {results['ResNet50']['confidence']:.2f}%")

        with col3:
            st.warning(f"**Mô hình: ViT-B16**")
            st.metric(
                label="Dự đoán", 
                value=results["ViT_B16"]["class"].capitalize()
            )
            st.write(f"Độ tự tin: {results['ViT_B16']['confidence']:.2f}%")
            
        # Hiển thị kết luận (mô hình nào tốt nhất)
        st.subheader("Phân tích:")
        if (results["VGG16"]["class"] == results["ResNet50"]["class"] == results["ViT_B16"]["class"]):
            st.balloons()
            st.success(f"Cả 3 mô hình đều đồng thuận dự đoán là: **{results['ViT_B16']['class'].capitalize()}**")
        else:
            st.warning("Các mô hình cho ra kết quả khác nhau. Đây là một ca khó!")
