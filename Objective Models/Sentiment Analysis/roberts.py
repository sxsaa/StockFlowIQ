# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from nltk.sentiment import SentimentIntensityAnalyzer
# from tqdm.notebook import tqdm
# import nltk 
# from transformers import AutoTokenizer
# from transformers import AutoModelForSequenceClassification
# from scipy.special import softmax

# plt.style.use('ggplot')

# # reading data
# df = pd.read_csv('Reviews.csv')
# print(df.head())
# df = df.head(500)
# print(df.shape)

# example = df['Text'][50]

# ###############################################################################

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

# ##############################################################################


# # Quick EDA
# ax = df['Score'].value_counts().sort_index().plot(kind='bar', title="Count of Reviews by Stars", figsize=(10,5))
# ax.set_xlabel('Review Stars')
# plt.show()

# MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
# tokenizer = AutoTokenizer.from_pretrained(MODEL)
# model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# # run for Roberta Model
# encoded_text = tokenizer(example, return_tensors='pt')
# output = model(**encoded_text)
# scores = output[0][0].detach().numpy()
# scores = softmax(scores)
# scores_dict = {
#     'roberta_neg':scores[0],
#     'roberta_neu':scores[1],
#     'roberta_pos':scores[2]
# }
# # print(scores_dict)

# def polarity_scores_roberta(example):
#     encoded_text = tokenizer(example, return_tensors='pt')
#     output = model(**encoded_text)
#     scores = output[0][0].detach().numpy()
#     scores = softmax(scores)
#     scores_dict = {
#         'roberta_neg':scores[0],
#         'roberta_neu':scores[1],
#         'roberta_pos':scores[2]
#     }
#     return (scores_dict)

# res = {}
# for i, row in df.iterrows():
#     try:
#         text = row['Text']
#         myid = row['Id']

#         roberta_scores_dict = polarity_scores_roberta(text)
#         res[myid] = roberta_scores_dict
#     except:
#         print(f'Broke for id {myid}')

        
# results_df = pd.DataFrame(res).T
# results_df = results_df.reset_index().rename(columns={'index': 'Id'})
# results_df = results_df.merge(df, how='left')

# results_df = results_df.merge(vaders[['Id', 'compound', 'pos', 'neu', 'neg']], how='left', on='Id')

# # Now, plot the pairplot with all necessary columns
# sns.pairplot(data=results_df,
#              vars=['vader_neg', 'vader_neu', 'vader_pos',
#                    'roberta_neg', 'roberta_neu', 'roberta_pos'],
#              hue='Score',
#              palette='tab10')
# plt.show()

# # print(res)

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from nltk.sentiment import SentimentIntensityAnalyzer
# from tqdm.notebook import tqdm
# import nltk 
# from transformers import AutoTokenizer
# from transformers import AutoModelForSequenceClassification
# from scipy.special import softmax

# plt.style.use('ggplot')

# # reading data
# df = pd.read_csv('Reviews.csv')
# print(df.head())
# df = df.head(500)
# print(df.shape)

# example = df['Text'][50]

# ###############################################################################

# # Plotting review count by stars
# ax = df['Score'].value_counts().sort_index().plot(kind='bar', title="Count of Reviews by Stars", figsize=(10,5))
# ax.set_xlabel('Review Stars')
# plt.show()

# # Initialize SentimentIntensityAnalyzer
# sia = SentimentIntensityAnalyzer()

# # Print polarity scores for the example review
# print(sia.polarity_scores(example))

# # Run polarity score in the entire dataset
# res = {}
# for i, row in df.iterrows():
#     text = row['Text']    # Text column
#     myid = row['Id']      # Id column
    
#     # Calculate the polarity scores for the text
#     polarity_scores = sia.polarity_scores(text)
    
#     # Store the results in the dictionary with Id as the key
#     res[myid] = polarity_scores

# print(res)

# # Convert the results into a DataFrame and merge with the original data
# vaders = pd.DataFrame(res).T
# vaders = vaders.reset_index().rename(columns={'index':'Id'})
# vaders = vaders.merge(df, how='left')

# # Plot Vader sentiment scores
# ax = sns.barplot(data=vaders, x='Score', y='compound')
# ax.set_title('Compound Score by Amazon Star Review')
# plt.show()

# fig, axs = plt.subplots(1, 3, figsize=(15,5))
# sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
# sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
# sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
# axs[0].set_title('Positive')
# axs[1].set_title('Neutral')
# axs[2].set_title('Negative')
# plt.show()

# ##############################################################################

# # Quick EDA - review count by stars
# ax = df['Score'].value_counts().sort_index().plot(kind='bar', title="Count of Reviews by Stars", figsize=(10,5))
# ax.set_xlabel('Review Stars')
# plt.show()

# # Load the RoBERTa model for sentiment analysis
# MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
# tokenizer = AutoTokenizer.from_pretrained(MODEL)
# model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# # Run for an example with the RoBERTa model
# encoded_text = tokenizer(example, return_tensors='pt')
# output = model(**encoded_text)
# scores = output[0][0].detach().numpy()
# scores = softmax(scores)
# scores_dict = {
#     'roberta_neg': scores[0],
#     'roberta_neu': scores[1],
#     'roberta_pos': scores[2]
# }

# # Define a function to calculate polarity scores using RoBERTa
# def polarity_scores_roberta(example):
#     encoded_text = tokenizer(example, return_tensors='pt')
#     output = model(**encoded_text)
#     scores = output[0][0].detach().numpy()
#     scores = softmax(scores)
#     scores_dict = {
#         'roberta_neg': scores[0],
#         'roberta_neu': scores[1],
#         'roberta_pos': scores[2]
#     }
#     return scores_dict

# # Process the entire dataset with RoBERTa
# res = {}
# for i, row in df.iterrows():
#     try:
#         text = row['Text']
#         myid = row['Id']
        
#         roberta_scores_dict = polarity_scores_roberta(text)
#         res[myid] = roberta_scores_dict
#     except:
#         print(f'Broke for id {myid}')

# # Convert the RoBERTa results into a DataFrame and merge with the original data
# results_df = pd.DataFrame(res).T
# results_df = results_df.reset_index().rename(columns={'index': 'Id'})
# results_df = results_df.merge(df, how='left')

# # Merge VADER scores with the results_df
# results_df = results_df.merge(vaders[['Id', 'compound', 'pos', 'neu', 'neg']], how='left', on='Id')

# # Plot pairplot using both VADER and RoBERTa scores
# sns.pairplot(data=results_df,
#              vars=['neg', 'neu', 'pos',  # Using the correct VADER columns
#                    'roberta_neg', 'roberta_neu', 'roberta_pos'],
#              hue='Score',
#              palette='tab10')
# plt.show()

from transformers import pipeline

# Load a pre-trained RoBERTa model for sentiment analysis
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# Example text
text = "I love this product! It's amazing."
result = sentiment_pipeline(text)
print(result)


