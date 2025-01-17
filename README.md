# IMDb Movie Review Sentiment Classification using BERT

## Project Overview

This project aims to classify IMDb movie reviews as positive or negative using a pre-trained BERT model for sentiment analysis. BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model that captures the contextual meaning of words in a sentence, making it highly effective for natural language processing (NLP) tasks such as text classification. The dataset used in this project is the IMDb movie review dataset, which contains a large collection of labeled movie reviews, allowing the model to learn patterns in sentiment classification.

The project focuses on:
- Text pre-processing (removing special characters, converting to lowercase, and removing stopwords)
- Tokenization and encoding using BERT's tokenizer
- Fine-tuning the BERT model for binary sentiment classification
- Evaluation of the model using accuracy and F1 score
- Performing inference on the test dataset


## Dataset Used

The dataset used in this project is the IMDb movie review dataset, which consists of:
- 50,000 reviews labeled as either positive or negative
- The reviews are split into training, validation, and test sets

The dataset is available on [Kaggle IMDb dataset](https://www.kaggle.com/), and it's ideal for sentiment analysis because of its balanced distribution of sentiments and the real-world nature of the movie reviews.

## Tools and Techniques

### Tools
- **PyTorch**: Used for model training and evaluation.
- **Transformers Library (Hugging Face)**: Used for accessing pre-trained BERT models and tokenizers.
- **Pandas**: Used for data manipulation and loading the dataset.
- **NLTK**: Used for text processing (stopword removal).
- **Scikit-learn**: Used for evaluation metrics like accuracy and F1 score.

### Techniques
- **Text Preprocessing**: Includes steps like converting text to lowercase, removing special characters, and stopwords, which helps in cleaning the data for better model performance.
- **Tokenization**: Text is tokenized using BERT's tokenizer, converting text into a format suitable for BERT.
- **Model Fine-tuning**: The BERT model is fine-tuned for sentiment analysis by training it on the IMDb dataset.
- **Evaluation**: The model is evaluated on accuracy and F1 score to assess its performance on unseen data.
- **Inference**: Once trained, the model is used to predict the sentiment of reviews in the test dataset.

