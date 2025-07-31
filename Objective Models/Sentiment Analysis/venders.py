# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from nltk.sentiment import SentimentIntensityAnalyzer
# from tqdm.notebook import tqdm
# import nltk 

# plt.style.use('ggplot')


# # nltk.download('vader_lexicon')

# # reading data
# df = pd.read_csv('Reviews.csv')
# print(df.head())
# df = df.head(500)
# print(df.shape)

# example = df['Text'][50]

# # Quick EDA
# ax = df['Score'].value_counts().sort_index().plot(kind='bar', title="Count of Reviews by Stars", figsize=(10,5))
# ax.set_xlabel('Review Stars')
# plt.show()

# sia = SentimentIntensityAnalyzer()

# print(sia.polarity_scores(example))

# # Run polaraty score in entire dataset
# res = {}
# for i, row in df.iterrows():
#     text = row['Text']    # Text column
#     myid = row['Id']      # Id column
    
#     # Calculate the polarity scores for the text
#     polarity_scores = sia.polarity_scores(text)
    
#     # Store the results in the dictionary with Id as the key
#     res[myid] = polarity_scores

# print(res)

# vaders = pd.DataFrame(res).T
# vaders = vaders.reset_index().rename(columns={'index':'Id'})
# vaders = vaders.merge(df, how='left')
# # print(vaders)


# # Plot vader results 
# ax = sns.barplot(data=vaders, x='Score', y='compound')
# ax.set_title('Compund Score by Amazon Star Review')
# plt.show()

# fig, axs = plt.subplots(1, 3, figsize=(15,5))
# sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
# sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
# sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
# axs[0].set_title('Positive')
# axs[1].set_title('Neutral')
# axs[2].set_title('Negative')
# plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk 

# Download VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

# Initialize Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

# Define the input text
text = "The store had a great variety of fresh vegetables, but the billing process was extremely slow and frustrating."

# Get sentiment scores
scores = sia.polarity_scores(text)

# Classify sentiment based on compound score
sentiment = "Positive" if scores['compound'] >= 0 else "Negative"

# Print results
print(f"Text: {text}")
print(f"Sentiment Scores: {scores}")
print(f"Predicted Sentiment: {sentiment}")








