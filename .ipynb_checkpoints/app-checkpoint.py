import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from streamlit_drawable_canvas import st_canvas

# -------------------------
# Load Model (Cached)
# -------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fashion_mnist_cnn.keras")

model = load_model()

# -------------------------
# Class Labels
# -------------------------
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

st.title("👕 Fashion MNIST CNN Application")

# Sidebar Menu
menu = st.sidebar.selectbox(
    "Select Feature",
    [
        "Draw & Predict",
        "Upload Single Image",
        "Upload Multiple Images",
        "Evaluate Test Dataset",
        "Confusion Matrix"
    ]
)

# -------------------------
# 1️⃣ Draw & Predict
# -------------------------
if menu == "Draw & Predict":

    st.subheader("Draw Clothing Item")

    canvas = st_canvas(
        fill_color="white",
        stroke_width=10,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas"
    )

    if st.button("Predict Drawing"):
        if canvas.image_data is not None:
            img = canvas.image_data[:, :, 0]
            img = Image.fromarray(img.astype('uint8'))
            img = img.resize((28, 28))
            img = np.array(img) / 255.0
            img = img.reshape(1, 28, 28, 1)

            prediction = model.predict(img)
            predicted_class = np.argmax(prediction)

            st.success(f"Prediction: {class_names[predicted_class]}")
            st.write("Confidence:", float(np.max(prediction)))
            st.bar_chart(prediction[0])


# -------------------------
# 2️⃣ Upload Single Image
# -------------------------
elif menu == "Upload Single Image":

    uploaded_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("L")
        image = image.resize((28, 28))

        st.image(image, width=150)

        img_array = np.array(image) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)

        st.success(f"Prediction: {class_names[predicted_class]}")
        st.write("Confidence:", float(np.max(prediction)))
        st.bar_chart(prediction[0])


# -------------------------
# 3️⃣ Upload Multiple Images
# -------------------------
elif menu == "Upload Multiple Images":

    uploaded_files = st.file_uploader(
        "Upload Multiple Images",
        type=["png","jpg","jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for file in uploaded_files:
            image = Image.open(file).convert("L")
            image = image.resize((28, 28))

            img_array = np.array(image) / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)

            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)

            st.image(image, width=100)
            st.write(f"Prediction: {class_names[predicted_class]}")
            st.write("Confidence:", float(np.max(prediction)))
            st.write("---")


# -------------------------
# 4️⃣ Evaluate Test Dataset
# -------------------------
elif menu == "Evaluate Test Dataset":

    st.subheader("Live Test Dataset Evaluation")

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    x_test = x_test / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1)
    y_test_cat = tf.keras.utils.to_categorical(y_test, 10)

    loss, accuracy = model.evaluate(x_test, y_test_cat, verbose=0)

    st.success(f"Test Accuracy: {accuracy:.4f}")
    st.write(f"Test Loss: {loss:.4f}")


# -------------------------
# 5️⃣ Confusion Matrix
# -------------------------
elif menu == "Confusion Matrix":

    st.subheader("Confusion Matrix")

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    x_test = x_test / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1)

    predictions = model.predict(x_test)
    y_pred = np.argmax(predictions, axis=1)

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=class_names,
                yticklabels=class_names,
                cmap="Blues")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)