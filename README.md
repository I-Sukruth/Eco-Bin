# Eco-Bin
Automatic Sorting of Recyclable Materials using CNN

Project Overview
This project addresses the problem of waste management by using AI to automate the sorting of recyclable materials. The system employs a Convolutional Neural Network (CNN) based on the pre-trained VGG16 model, fine-tuned to classify 12 different categories of waste. The automation of this process can significantly reduce manual labor, errors, and contribute to more sustainable waste management practices.

Dataset
The dataset used for training consists of images across 12 categories, which include:
Battery
Biological
Brown Glass
Cardboard
Clothes
Green Glass
Metal
Paper
Plastic
Shoes
Trash
White Glass
The images were obtained from Kaggle, resized to 128x128 pixels, and preprocessed using normalization and data augmentation techniques such as rotation and zooming to improve the generalization of the model.

Model Architecture
The architecture for this project uses the VGG16 model pre-trained on the ImageNet dataset. The top layers are replaced to suit the classification of our 12 recyclable categories. The model structure consists of:

VGG16 Base Model: For feature extraction.
Flatten Layer: To convert the feature maps into a 1D vector.
Dense Layer (256 units): For classification using ReLU activation.
Dropout Layer (0.5 probability): To reduce overfitting by randomly turning off neurons during training.
Final Dense Layer (12 units): The output layer with Softmax activation to predict the category of waste.
Technologies Used:
TensorFlow & Keras: Deep learning framework used to build and train the model.
VGG16: Pre-trained model used for transfer learning.
Python: Primary programming language.
Pillow (PIL): For image preprocessing.
NumPy: For array manipulation.
Streamlit: For creating a simple web interface (Optional)

Practical Implementation
The model can be integrated into real-world systems like recycling plants or automated sorting machines. By classifying waste in real-time, the model helps streamline the sorting process, reduces contamination in recycling streams, and improves the recycling rate.

Contributing
Contributions are welcome! To contribute:

Fork this repository.
Create a new feature branch (git checkout -b feature/your-feature).
Commit your changes (git commit -m 'Add your feature').
Push to your branch (git push origin feature/your-feature).
Submit a pull request.
