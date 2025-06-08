# Image Classifier with PyTorch

This project trains a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset using PyTorch.

## 🧠 Model
- 2 Conv layers + MaxPooling
- 1 Hidden Dense layer
- Output: 10 classes (CIFAR-10)

## 🚀 How to Run

```bash
pip install -r requirements.txt
python train.py
python predict.py # Make sure test_image.png is present
```

## 📁 Project Structure
```
project_root/
├── data/
├── models/
├── notebooks/
├── utils.py
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

## 📓 Notebook Demo
A Jupyter Notebook demonstrating inference and visualization with the trained CIFAR-10 classifier.  
[View Notebook](notebooks/Image_Classifier_Demo.ipynb)

## ✅ Results
After 10 epochs, achieves ~70–75% accuracy on CIFAR-10 test set (can improve with tuning).

## 🙋‍♂️ Author
Built by Adam — aspiring AI researcher and backend developer.
