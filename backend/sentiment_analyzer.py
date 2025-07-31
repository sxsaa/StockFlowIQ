from transformers import pipeline

# Load pre-trained sentiment analysis model
sentiment_pipeline = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment")

def analyze_sentiment(text):
    """
    Analyzes the sentiment of the given text using RoBERTa.
    Returns:
        - 0 (Neutral)
        - 1 (Negative)
        - 2 (Positive)
    """
    result = sentiment_pipeline(text)[0]
    
    if result["label"] == "LABEL_0":  # Negative
        return 1
    elif result["label"] == "LABEL_1":  # Neutral
        return 0
    else:  # Positive
        return 2
