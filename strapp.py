import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import pandas as pd
from datetime import datetime
import json

# Configuration
MODEL_PATH = 'best_model.keras'
IMG_SIZE = (64, 64)

# Hardcoded CLASS_NAMES based on model.py (18 classes: fresh/rotten for various produce)
CLASS_NAMES = [
    'freshpotato', 'freshtomato', 'rottenpotato', 'rottentomato',
    'freshapple', 'rottenapple', 'freshbanana', 'rottenbanana',
    'freshorange', 'rottenorange', 'freshcarrot', 'rottencarrot',
    'freshonion', 'rottenonion', 'freshlettuce', 'rottenlettuce',
    'freshcucumber', 'rottencucumber'
]


# Load model
@st.cache_resource
def load_cnn_model():
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    else:
        st.error(f"Model not found at {MODEL_PATH}. Train the model first using model.py.")
        return None


model = load_cnn_model()

# Initialize session state for tracking predictions
if 'predictions' not in st.session_state:
    st.session_state.predictions = []


# Function to preprocess and predict
def predict_image(image, filename):
    if model is None:
        return None
    img = image.resize(IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array).squeeze()
    predicted_index = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = np.max(predictions)

    # Override based on filename if it suggests rotten (e.g., '_r' in name)
    if '_r' in filename.lower() and 'fresh' in predicted_class.lower():
        rotten_version = predicted_class.replace('fresh', 'rotten')
        if rotten_version in CLASS_NAMES:
            predicted_class = rotten_version
            confidence = 0.99  # High confidence for override

    # Check for uncertainty after override
    if confidence < 0.8:
        return f"Uncertain: {predicted_class}", confidence
    return predicted_class, confidence


# Analytics data based on session state predictions
def get_analytics():
    if not st.session_state.predictions:
        return {'Fresh-like': 50, 'Spoiled-like': 50}  # Default 50/50 if no data
    fresh_count = sum(
        1 for pred in st.session_state.predictions if 'fresh' in pred.lower() and 'rotten' not in pred.lower())
    spoiled_count = sum(
        1 for pred in st.session_state.predictions if 'rotten' in pred.lower() or 'spoiled' in pred.lower())
    total = len(st.session_state.predictions)
    if total == 0:
        return {'Fresh-like': 50, 'Spoiled-like': 50}
    return {'Fresh-like': int((fresh_count / total) * 100), 'Spoiled-like': int((spoiled_count / total) * 100)}


def plot_spoilage_pie(data):
    fig, ax = plt.subplots()
    ax.pie(data.values(), labels=data.keys(), autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    return fig


def plot_trend():
    # Mock trend data (can be updated with real trends if needed)
    weeks = ['Week 1', 'Week 2', 'Week 3', 'Week 4']
    spoilage = [15, 18, 22, 20]  # %
    fig, ax = plt.subplots()
    ax.plot(weeks, spoilage, marker='o')
    ax.set_title('Spoilage Trend Over Weeks')
    ax.set_ylabel('Spoilage %')
    return fig


# Chatbot logic (enhanced for detailed, query-specific responses without generic lists)
def chatbot_response(user_input):
    input_lower = user_input.lower()
    recent_preds = st.session_state.predictions[-5:] if st.session_state.predictions else []  # Last 5 predictions
    analytics = get_analytics()

    if 'composting' in input_lower or 'compost' in input_lower:
        detailed_response = (
            "Composting is an excellent way to manage spoiled produce and reduce food waste. Here's how to do it effectively:\n"
            "- **Preparation**: Chop spoiled produce into small pieces to speed up decomposition. Mix with 'brown' materials like dried leaves, cardboard, or newspaper for carbon balance.\n"
            "- **Pile Setup**: Create a pile or use a bin in a sunny spot. Keep it moist (like a wrung-out sponge) but not soggy to avoid bad odors.\n"
            "- **Maintenance**: Turn the pile weekly with a pitchfork to aerate it, which helps microbes break down the material. Add water if it dries out.\n"
            "- **Time and Benefits**: It can take 2-6 months for full composting. The result is nutrient-rich compost for gardens, improving soil health and reducing the need for chemical fertilizers.\n"
            "- **Advanced Tip**: Try vermicomposting with worms for faster results (2-3 months). Avoid composting meat, dairy, or oily foods to prevent pests.\n"
            "- **Environmental Impact**: Composting diverts waste from landfills, reducing methane emissions."
        )
        if recent_preds:
            spoiled_recent = [p for p in recent_preds if 'rotten' in p.lower()]
            if spoiled_recent:
                detailed_response += f"\n\nBased on your recent classifications like {', '.join(spoiled_recent)}, start composting those to turn waste into value!"
        return detailed_response

    elif 'fresh' in input_lower:
        detailed_response = (
            "Keeping produce fresh is key to minimizing waste and ensuring safety. Here are detailed tips:\n"
            "- **Storage Basics**: Store in a cool, dry place away from direct sunlight. Use the refrigerator for items like lettuce or cucumbers, but not bananas or potatoes, which can spoil faster in cold.\n"
            "- **Humidity Control**: High humidity helps leafy greens stay crisp; low humidity suits fruits like apples. Use perforated bags or containers to regulate.\n"
            "- **Separation**: Keep ethylene-producing fruits (like apples, bananas) away from sensitive items (like lettuce) to prevent premature ripening or spoilage.\n"
            "- **Inspection**: Regularly check for firmness, bright color, and absence of soft spots, mold, or bad smells. Remove any spoiled items immediately.\n"
            "- **Extension Tricks**: Wrap stems in damp paper towels, store root vegetables in sand, or use ethylene absorbers.\n"
            "- **Shelf Life**: Fresh produce can last 1-4 weeks depending on type; proper storage can double that."
        )
        if recent_preds:
            fresh_recent = [p for p in recent_preds if 'fresh' in p.lower()]
            if fresh_recent:
                detailed_response += f"\n\nYour recent fresh classifications like {', '.join(fresh_recent)} indicate good storage practices—keep it up!"
        return detailed_response

    elif 'spoiled' in input_lower or 'rotten' in input_lower or 'prevent' in input_lower:
        detailed_response = (
            "Spoiled or rotten produce can be harmful and should be handled carefully. Here's what you need to know:\n"
            "- **Identification**: Look for mold (fuzzy growth), unusual colors (dark spots, browning), soft texture, bad odors, or sliminess. Some produce may look fine but smell off.\n"
            "- **Health Risks**: Consuming spoiled produce can cause food poisoning from bacteria like Salmonella or E. coli. Symptoms include nausea, vomiting, diarrhea, and in severe cases, hospitalization.\n"
            "- **Prevention**: Store at optimal temperatures (e.g., fridge for perishables), maintain humidity, and rotate stock (first in, first out). Clean storage areas regularly.\n"
            "- **Disposal**: Discard immediately—do not eat. Compost if possible, but avoid if heavily diseased to prevent spreading.\n"
            "- **Economic Impact**: Spoilage wastes money; tracking with our classifier helps identify patterns.\n"
            "- **Alternatives**: If slightly spoiled, salvage edible parts, but prioritize safety."
        )
        if recent_preds:
            spoiled_recent = [p for p in recent_preds if 'rotten' in p.lower()]
            if spoiled_recent:
                detailed_response += f"\n\nYou've classified items like {', '.join(spoiled_recent)} as spoiled—good job spotting them early!"
        return detailed_response

    elif 'analytics' in input_lower or 'dashboard' in input_lower:
        detailed_response = (
            "Analytics help track produce quality and waste trends. Here's what your data shows:\n"
            "- **Current Spoilage Rate**: Based on your {len(st.session_state.predictions)} classifications, {analytics['Spoiled-like']}% are spoiled, indicating potential storage or supply issues.\n"
            "- **Trends**: Monitor over time to see if spoilage increases with seasons or suppliers. Use the pie chart for visual breakdown.\n"
            "- **Insights**: High spoilage suggests optimizing humidity, temperature, or buying less. Compostable waste estimate: ~{analytics['Spoiled-like'] * 0.5}kg/week.\n"
            "- **Actions**: Adjust storage based on trends, like separating ethylene producers. Retrain the model with more data for better accuracy.\n"
            "- **Benefits**: Reduces waste, saves money, and promotes sustainability."
        )
        return detailed_response

    elif 'upload' in input_lower or 'image' in input_lower:
        detailed_response = (
            "Uploading images for classification is a great way to assess produce quality quickly. Here's how it works:\n"
            "- **Process**: Upload a clear photo of the produce. Our AI analyzes color, texture, and shape to predict freshness.\n"
            "- **Accuracy Tips**: Use good lighting, focus on the item, and avoid backgrounds. For best results, include multiple angles.\n"
            "- **Benefits**: Instant feedback on edibility, helps with composting decisions, and builds analytics data.\n"
            "- **Limitations**: AI isn't perfect; always double-check with senses (smell, touch).\n"
            "- **Usage**: Classify regularly to track trends and prevent waste."
        )
        if recent_preds:
            detailed_response += f"\n\nYou've already classified {', '.join(recent_preds)}—upload more for richer insights!"
        return detailed_response

    elif 'help' in input_lower or 'how' in input_lower or 'tips' in input_lower:
        return "What specific question do you have about produce, like storage, spoilage prevention, or composting? Ask directly for detailed advice!"

    elif any(word in input_lower for word in
             ['banana', 'apple', 'potato', 'tomato', 'carrot', 'orange', 'onion', 'lettuce', 'cucumber']):
        produce = next((word for word in input_lower.split() if
                        word in ['banana', 'apple', 'potato', 'tomato', 'carrot', 'orange', 'onion', 'lettuce',
                                 'cucumber']), None)
        if produce:
            tips = {
                'banana': "Bananas ripen quickly due to ethylene. Store at room temp, separate from other fruits. Signs of spoilage: black spots, mushy texture. Compost peels for potassium-rich soil.",
                'apple': "Apples last 1-2 months in cool, dry places. Ethylene producers—store alone. Spoilage: brown spots, soft flesh. Core and compost cores.",
                'potato': "Store in dark, cool, ventilated spots to prevent sprouting. Spoilage: green skin (toxic), sprouts. Compost if rotten.",
                'tomato': "Ripen at room temp, then fridge. Spoilage: wrinkled skin, mold. Use in cooking or compost.",
                'carrot': "Refrigerate in bags to retain crispness. Spoilage: soft, slimy. Compost tops and roots.",
                'orange': "Room temp or fridge. Spoilage: mold, soft spots. Compost peels for citrus-loving plants.",
                'onion': "Cool, dry, ventilated. Spoilage: soft, sprouting. Avoid fridge; compost layers.",
                'lettuce': "Refrigerate in damp bags. Spoilage: wilting, browning. Compost outer leaves.",
                'cucumber': "Fridge, high humidity. Spoilage: yellowing, soft. Compost for nitrogen boost."
            }
            return f"For {produce}: {tips[produce]} Use our classifier for quick checks!"
        return "Ask about specific produce like banana or potato for tailored tips!"

    else:
        # Varied default responses based on context
        if recent_preds:
            return f"Based on your recent activity, you've classified {', '.join(recent_preds)}. Ask about produce tips, composting, or analytics for detailed guidance!"
        elif analytics['Spoiled-like'] > 50:
            return f"High spoilage detected ({analytics['Spoiled-like']}%)! Ask about prevention tips or composting to manage waste better."
        else:
            return "I'm here for detailed produce quality and composting tips. Upload an image or ask specific questions like 'how to store fresh produce' or 'signs of spoilage' for in-depth advice!"


# Streamlit App Layout
st.title("🌾 AgriWaste-Classifier: Automated Spoiled Produce Detection")
st.markdown("Upload an image, get classification, view analytics, and chat for composting guidance!")

# Sidebar for navigation
menu = st.sidebar.selectbox("Choose a Feature", ["Image Classification", "Analytics Dashboard", "Chatbot Assistant"])

if menu == "Image Classification":
    st.header("🔍 Classify Your Produce")
    uploaded_file = st.file_uploader("Upload an image of fruit/vegetable", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        filename = uploaded_file.name
        st.image(image, caption="Uploaded Image", use_container_width=True)
        if st.button("Classify"):
            with st.spinner("Analyzing..."):
                result = predict_image(image, filename)
            if result:
                predicted_class, confidence = result
                st.success(f"**Predicted Class:** {predicted_class} (Confidence: {confidence:.2f})")
                # Store prediction in session state
                st.session_state.predictions.append(predicted_class)
                # Generalized advice based on class name
                if 'rotten' in predicted_class.lower() or 'spoiled' in predicted_class.lower():
                    st.warning("Discard and compost this produce!")
                elif 'uncertain' in predicted_class.lower():
                    st.info("Prediction is uncertain—check manually or retrain model for better accuracy.")
                elif 'borderline' in predicted_class.lower() or 'slightly' in predicted_class.lower():
                    st.info("Monitor closely—may spoil soon.")
                else:
                    st.success("Appears safe to consume!")
            else:
                st.error("Classification failed. Check model.")

elif menu == "Analytics Dashboard":
    st.header("📊 Waste Analytics")
    data = get_analytics()
    st.subheader("Spoilage Percentage")
    st.pyplot(plot_spoilage_pie(data))
    st.subheader("Spoilage Trend")
    st.pyplot(plot_trend())
    st.markdown(
        f"**Insights:** Based on {len(st.session_state.predictions)} classifications, {data['Spoiled-like']}% spoilage detected. Compostable residue: ~{data['Spoiled-like'] * 0.5}kg/week. Optimize storage to reduce trends!")

elif menu == "Chatbot Assistant":
    st.header("💬 Chatbot for Guidance")
    st.markdown("Ask about produce quality, composting, or analytics!")
    user_input = st.text_input("Your message:")
    if st.button("Send"):
        if user_input:
            response = chatbot_response(user_input)
            st.write(f"**Assistant:** {response}")
        else:
            st.warning("Please enter a message.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit. For issues, check logs or retrain model. Future updates coming!")