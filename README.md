# AgriWaste-Classifier
# Automated detection of spoiled agriculture Produce using CNN
# Problem Statement
A large amount of agricultural produce is wasted due to inefficient post-harvest handling and distribution.
-> Farmers and distributors often lack real-time tools to assess the freshness and quality of produce, leading to premature spoilage, poor storage management, and unnecessary losses.
-> This project seeks to design an AI-driven system that utilizes Convolutional Neural Networks (CNNs), Machine Learning analytics, and a Chatbot assistant to:
   1.Detect and classify spoiled produce automatically,
   2.Provide waste analytics and composting recommendations, and
   3.Interact with users through a bot for guidance on produce quality and composting practices.
# Proposed Solution
Integrates computer vision, machine learning, and conversational AI into a single system:
Image Classification using CNN
   Capture images of fruits and vegetables.
   Use a CNN model (like ResNet50, EfficientNet, or a custom CNN) to classify produce into categories:
   Fresh,Borderline,Spoiled
   This helps automatically separate good produce from waste.
Machine Learning Analytics
   Use the classification data to: Calculate percentage of spoiled produce. Predict spoilage trends (e.g., based on temperature, storage time, etc.).
   Estimate compostable residue quantity.
-> ML algorithms like Random Forest or Linear Regression can be applied for these predictions.
# Chatbot Integration A simple bot interface (using Python, Streamlit, or Flask) will:
Allow users (farmers/distributors) to upload an image and get instant quality feedback. Answer basic questions about composting and best practices,Display system analytics (like weekly waste reports or suggestions).
# Technology Stack
Python
Deep Learning	CNN 
Machine Learning	Scikit-learn
Chatbot	Streamlit
Database	MySQL
Image Processing	OpenCV
Visualization	Matplotlib, Seaborn
Platform	VS Code
Version Control	GitHub
----------------------------------------------------------------------------------------------------
# Week-1
Installed dependencies,Requirements Libraries
Project setup files in VS code
Downloaded Data set from Kaggle
Database created in MYSQL for this Project
Used a Pretrained model like EfficientNetB0 or ResNet50 for better accuracy.
Image classification Using CNN model and analytics.
