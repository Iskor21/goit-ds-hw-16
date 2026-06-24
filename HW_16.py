import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps
import tensorflow as tf

# ---------------- Завантаження моделей ----------------
cnn_model = load_model("cnn_model.keras")

# VGG16 через TensorFlow Lite Interpreter
vgg_interpreter = tf.lite.Interpreter(model_path="vgg16_quantized.tflite")
vgg_interpreter.allocate_tensors()
vgg_input_details = vgg_interpreter.get_input_details()
vgg_output_details = vgg_interpreter.get_output_details()

# ---------------- Завантаження історій ----------------
cnn_history = np.load("cnn_history.npz", allow_pickle=True)["history"].item()
vgg_history = np.load("vgg_history.npz", allow_pickle=True)
# доступ до масивів
fe_acc = vgg_history["fe_accuracy"]
ft_acc = vgg_history["ft_accuracy"]

labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

st.title("Класифікація одягу")

# ---------------- 1. Вибір моделі ----------------
model_choice = st.radio("Оберіть модель:", ["CNN", "VGG16"])

# ---------------- 2. Завантаження зображення ----------------
uploaded_file = st.file_uploader("Завантажте зображення одягу", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    pil_img = Image.open(uploaded_file)

    if model_choice == "CNN":
        # CNN очікує 28×28 grayscale
        pil_img = ImageOps.grayscale(pil_img.resize((28,28)))
        img_array = image.img_to_array(pil_img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = cnn_model.predict(img_array)

    else:  # VGG16 очікує 96×96×3
        pil_img = pil_img.resize((96, 96))
        img_array = image.img_to_array(pil_img) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

        # Використовуємо TFLite Interpreter замість .predict()
        vgg_interpreter.set_tensor(vgg_input_details[0]['index'], img_array)
        vgg_interpreter.invoke()
        prediction = vgg_interpreter.get_tensor(vgg_output_details[0]['index'])

    st.image(pil_img, caption="Оброблене зображення", use_column_width=True)

    # ---------------- 3. Результати класифікації ----------------
    st.subheader("Ймовірності по класах")
    for i, label in enumerate(labels):
        st.write(f"{label}: {prediction[0][i]:.4f}")

    st.markdown("ПЕРЕДБАЧЕНИЙ КЛАС:")
    st.write(labels[np.argmax(prediction)])

    # ---------------- 4. Графіки історії ----------------
    if model_choice == "CNN":
        history = cnn_history
        st.subheader("Графік точності")
        plt.figure()
        plt.plot(history['accuracy'], label='train acc')
        plt.plot(history['val_accuracy'], label='val acc')
        plt.legend()
        st.pyplot(plt)

        st.subheader("Графік втрат")
        plt.figure()
        plt.plot(history['loss'], label='train loss')
        plt.plot(history['val_loss'], label='val loss')
        plt.legend()
        st.pyplot(plt)

    else:  # VGG16
        history = vgg_history
        st.subheader("Графік точності (FE + FT)")
        plt.figure()
        plt.plot(history['fe_accuracy'], label='FE train acc')
        plt.plot(history['fe_val_accuracy'], label='FE val acc')
        plt.plot(history['ft_accuracy'], label='FT train acc')
        plt.plot(history['ft_val_accuracy'], label='FT val acc')
        plt.legend()
        st.pyplot(plt)

        st.subheader("Графік втрат (FE + FT)")
        plt.figure()
        plt.plot(history['fe_loss'], label='FE train loss')
        plt.plot(history['fe_val_loss'], label='FE val loss')
        plt.plot(history['ft_loss'], label='FT train loss')
        plt.plot(history['ft_val_loss'], label='FT val loss')
        plt.legend()
        st.pyplot(plt)
