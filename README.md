🌾 AgriWaste-Classifier
Automated Detection of Spoiled Agricultural Produce using CNN
🧩 Problem Statement

A large amount of agricultural produce is wasted due to inefficient post-harvest handling and distribution.
Farmers and distributors often lack real-time tools to assess the freshness and quality of produce, leading to:

Premature spoilage

Poor storage management

Unnecessary financial and resource losses

This project aims to design an AI-driven system that utilizes Convolutional Neural Networks (CNNs), Machine Learning analytics, and a Chatbot assistant to:

🔍 Automatically detect and classify spoiled produce

📊 Provide waste analytics and composting recommendations

🤖 Interact with users through a chatbot for produce-quality guidance and composting practices

💡 Proposed Solution

The AgriWaste-Classifier integrates Computer Vision, Machine Learning, and Conversational AI into a unified system.

🖼️ 1. Image Classification (CNN)

Capture images of fruits and vegetables.

Use a CNN model (e.g., ResNet50, EfficientNetB0, or a custom CNN) to classify produce into categories:

Fresh 🍏

Borderline 🍊

Spoiled 🍅

Automatically separate usable produce from waste.

📈 2. Machine Learning Analytics

Analyze classification data to:

Calculate percentage of spoiled produce

Predict spoilage trends (based on temperature, storage duration, etc.)

Estimate compostable residue quantity

ML algorithms like Random Forest or Linear Regression are used for these predictive analytics.

💬 3. Chatbot Integration

A user-friendly chatbot interface (built using Python + Streamlit/Flask) that allows:

Uploading an image for instant freshness detection

Receiving quality feedback and composting tips

Viewing weekly waste reports and optimization suggestions

🛠️ Technology Stack
Category	Tools/Frameworks
Programming Language	Python
Deep Learning	CNN (ResNet50, EfficientNetB0, Custom CNN)
Machine Learning	Scikit-learn
Image Processing	OpenCV
Visualization	Matplotlib, Seaborn
Web Interface	Streamlit / Flask
Database	MySQL
Version Control	Git & GitHub
IDE	VS Code
📅 Project Timeline
Week 1

✅ Installed dependencies and required Python libraries

✅ Project setup completed in VS Code

✅ Downloaded dataset from Kaggle

✅ Created a MySQL database for produce and classification storage

✅ Implemented CNN model using EfficientNetB0 / ResNet50 for image classification

✅ Generated initial analytics and visualizations for produce quality

🚀 Future Enhancements

Integrate IoT sensors for real-time temperature and humidity tracking

Add voice-based interaction to the chatbot

Expand dataset to include regional produce varieties

Deploy system on cloud for farmer access via mobile app

📊 Sample Output (Expected)

Classification output: Fresh / Borderline / Spoiled

Analytics dashboard: Pie chart of spoilage percentage, line graph for trend prediction

Chatbot interface: Image upload + composting advice

🧠 Model Architecture
Input Image → Preprocessing (OpenCV)
→ CNN Model (EfficientNetB0 / ResNet50)
→ Classification Layer (Softmax)
→ Output: [Fresh, Borderline, Spoiled]

📦 Installation & Usage
1. Clone Repository
git clone https://github.com/yourusername/AgriWaste-Classifier.git
cd AgriWaste-Classifier

2. Install Dependencies
pip install -r requirements.txt

3. Run Streamlit App
streamlit run app.py

4. Use the Application

Upload fruit/vegetable images

Get freshness classification

View analytics

Chat with the assistant for composting guidance
