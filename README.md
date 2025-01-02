
![output](https://github.com/user-attachments/assets/15909cc0-ab7c-4aad-b554-032b79741576)

**Sentiment Analysis for Mental Health Classification**

*Project Overview*
This project leverages sentiment analysis and machine learning techniques to classify mental health statuses (Normal, Depression, Anxiety, Suicidal, Stress, Bipolar, and Personality Disorder) from a dataset of over 52,000 labeled text statements. The goal is to develop a predictive model that can assess mental health conditions based on textual data, providing early intervention tools and support mechanisms for individuals in need.

*Technologies Used*
Python
Scikit-learn: For machine learning algorithms
NLTK: Natural Language Toolkit for text preprocessing
TensorFlow & PyTorch: Deep learning frameworks
Matplotlib & Seaborn: For data visualization
TF-IDF: For feature extraction
BERT: For advanced text classification (exploratory)
Pandas & NumPy: For data manipulation
RandomOversampler (from imbalanced-learn): For addressing class imbalance

*Dataset*
The dataset consists of 52,681 text statements, each labeled with one of the following mental health statuses:

Normal
Depression
Suicidal
Anxiety
Stress
Bipolar
Personality Disorder
These statements were sourced from diverse platforms, including Reddit, Twitter, and curated datasets available on Kaggle.

*Project Goals*
Classify mental health statuses based on textual data using machine learning algorithms.
Address class imbalance through techniques like Random Oversampling to ensure equal representation for underrepresented categories.
Develop predictive models that assist in early mental health detection and support tools like chatbots or early intervention systems.

*Methodology*
Data Preprocessing:

Text cleaning (removal of hyperlinks, user handles, punctuation, etc.)
Stopword removal using NLTK
Stemming (using SnowballStemmer from NLTK)
TF-IDF vectorization to transform raw text into structured numerical features.

Modeling:

Multiple machine learning models were trained:
Logistic Regression
Naive Bayes (Multinomial)
Random Forest
Support Vector Machines (SVM)
Hyperparameter optimization and model evaluation using metrics like accuracy, precision, recall, and F1-score.
Addressed class imbalance using Random Oversampling.
Exploration of Advanced Models:

For improved performance, BERT (Bidirectional Encoder Representations from Transformers) was explored to capture contextual nuances in the text.


Evaluation:

Evaluation metrics included accuracy, precision, recall, and F1-score.
Performance visualized through confusion matrices and classification reports.
