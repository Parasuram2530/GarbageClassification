🗑️ Garbage Classification using Transfer Learning (EfficientNetV2B2)
This project aims to build an intelligent garbage classification system using Transfer Learning with the EfficientNetV2B2 architecture. It classifies garbage images into various categories like plastic, glass, cardboard, metal, paper, etc., enabling smarter waste segregation and promoting sustainability.

📌 Problem Statement
Manual garbage sorting is inefficient, error-prone, and poses health risks. Automating this process using AI can improve waste management, recycling, and environmental health.

🎯 Objectives
Develop a robust deep learning model using EfficientNetV2B2.

Classify garbage images into their respective categories.

Optimize the model using hyperparameter tuning.

Deploy the model using Streamlit for real-time inference.

🧠 Key Features
✅ Transfer learning with pre-trained EfficientNetV2B2

✅ Achieves 90%+ validation accuracy

✅ Real-time image prediction via web app

✅ Supports 6+ garbage categories

✅ Lightweight model for deployment

🗂️ Dataset
Source: Kaggle Garbage Classification Dataset

Classes: Cardboard, Glass, Metal, Paper, Plastic, Trash

Format: Images organized in subfolders per class

🛠️ Tools & Technologies
Category	Tools/Frameworks
Language	Python 3.10
Deep Learning	TensorFlow 2.10 (with GPU support)
Model	EfficientNetV2B2 (Keras Applications)
Visualization	Matplotlib, Seaborn
Image Handling	OpenCV, PIL
Deployment	Streamlit
Others	Scikit-learn, NumPy, Pandas

🧪 Model Architecture
text
Copy
Edit
EfficientNetV2B2
└── Input Layer (224x224x3)
└── EfficientNetV2B2 Base (pre-trained on ImageNet)
└── Global Average Pooling
└── Dense Layer (ReLU)
└── Dropout
└── Output Layer (Softmax – 6 classes)
🔧 Training Details
Optimizer: Adam

Loss Function: Categorical Crossentropy

Metrics: Accuracy

Epochs: 25+

Augmentation: Yes (ImageDataGenerator)

Validation Accuracy: ~90%

📊 Results
Metric	Value
Train Accuracy	~95%
Val Accuracy	~90%
Test Accuracy	~89%
Confusion Matrix	✅
Classification Report	✅

🚀 How to Run
1. Clone the repository
git clone https://github.com/yourusername/garbage-classification.git
cd garbage-classification

2. Install dependencies
conda activate py310
pip install -r requirements.txt

3. Train the model
# Open model.ipynb and run all cells

4. Run Streamlit app
streamlit run app.py



🤝 Acknowledgements
Kaggle Dataset Contributors

TensorFlow Team

Edunet Foundation Internship

🧑‍💻 Author
Mannem Parasuram (Parasuu)
B.Tech CSE (AI & ML), Mohan Babu University
LinkedIn • GitHub

