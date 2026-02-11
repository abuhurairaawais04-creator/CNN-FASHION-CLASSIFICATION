# 👕 Fashion MNIST CNN Classification App

A Convolutional Neural Network (CNN) based image classification system built using TensorFlow/Keras and deployed with Streamlit.

This application classifies clothing items from the Fashion MNIST dataset and provides an interactive web interface for prediction and evaluation.

---

## 🚀 Features

- 🎨 Draw & Predict using interactive canvas
- 🖼 Upload single image prediction
- 📂 Upload multiple image prediction
- 📊 Live test dataset evaluation
- 🔎 Confusion matrix visualization
- 📈 Probability distribution chart
- ⚡ Cached model loading for optimized performance

---

## 🧠 Model Details

- Dataset: Fashion MNIST
- Input Shape: 28x28 grayscale images
- Architecture:
  - Conv2D + ReLU
  - MaxPooling
  - Dropout
  - Dense layers
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Final Test Accuracy: **~93%**

---

## 🗂 Project Structure
├── app.py
├── fashion_mnist_cnn.keras
├── requirements.txt
└── README.md



---


## ⚙️ Installation


Clone the repository:


```bash
git clone https://github.com/yourusername/repository-name.git
cd repository-name

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py
📦 Requirements

streamlit

tensorflow

numpy

pillow

matplotlib

seaborn

scikit-learn

streamlit-drawable-canvas

📊 Classes

The model predicts the following 10 categories:

T-shirt/top

Trouser

Pullover

Dress

Coat

Sandal

Shirt

Sneaker

Bag

Ankle boot

🔍 Confusion Matrix

The application includes a full confusion matrix visualization to analyze classification performance and misclassifications.

📌 Future Improvements

Grad-CAM visualization

Transfer learning implementation

Model comparison dashboard

Cloud deployment (Streamlit Cloud / Hugging Face)

👨‍💻 Author

Abu Huraira Awais
BS Computer Science

📜 License

This project is for educational and academic purposes.



---


# ✅ 3️⃣ requirements.txt (Final Version)


Create:



requirements.txt



Add:



streamlit
tensorflow
numpy
pillow
matplotlib
seaborn
scikit-learn
streamlit-drawable-canvas



---


# ✅ 4️⃣ Optional: Add Badges (Makes Project Look Professional)


Add this at the very top of README:


```markdown
![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)