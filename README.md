# Group 3:
Xiaojing Shen/xshen16@depaul.edu   
Serena Yang/myang43@depaul.edu   
Yue Hou/yhou14@depaul.edu    
 
# < Sentiment-Analysis-Based Movie Recommender Systems>
## Description
There are three milestones in this project. Sentiment analysis was first implemented on an IMDb dataset from Kaggle, on which multiple models will be applied to select the ideal arithmetic model. Next, we applied a web crawler on IMDb to crawl data that fit in the following recommender systems. The last step was integrating crawled data with the selected sentiment model to build a personalized recommender system.
## Environment
We executed the code in MacOS Monterey (version 12.0.1) and MacOS Big Sur (version 11.1). The IDE that we used is Jupyter Notebook, and the programming language is Python 3.8. Also, the code is conducted 244MB Memory, 6.1% CPU, and 2.71GB Disk Space.

## Table of Contents
- [Stage 1: Sentiment Analysis]
- [Stage 2: Crawler Web]
- [Stage 3: Recommender System]

## Stage 1: Sentiment Analysis
File: Stage_1_Sentiment_Analysis.ipynb
Data Files: IMDB Dataset.csv (downloaded from Kaggle), reviews_top.csv (web crawler from stage 2), glove.6B.100d.txt (download from glove.6B.zip)
1.	loaded in file IMDB Dataset.csv and apply EDA to find some insights from the dataset.
2.	split the dataset into training and testing
3.	apply text normalization, including tokenization, removing stripe and special characters, text stemming, remove stop words by using the nltk package
4.	convert text-based feature – “review” to vectors by GloVe using glove.6B.100d.txt
5.	convert text-based feature – “review” to vectors by Bags of Word
6.	convert text-based feature – “review” to vectors by Term Frequency-Inverse Document Frequency (TFIDF)
7.	apply logistic regression /stochastic gradient descent / linear SVM/ multinomial Naïve Bayes on data, which is processed by BOW and TFIDF
8.	apply hybrid CNN-LSTM and DNN model for data, which is processed by GloVe
9.	compare the performance of all models; DNN with GloVe has the highest accuracy. It will be introduced in stage 3 to classify the review of movies
After stage 2,
1.	load in reviews_top.csv
2.	use sentiment analysis model to identify review
3.	save final sentiment analyzed dataset into dataset_review_1.csv for stage 3

## Stage 2: Crawler Web
File: Stage_2_Crawler_Web.ipynb
Data Files: movie_dic.csv, reviews_top.csv, df_IMDb_top25_info.csv 
1.	get movie list from IMDB playlist store as a dictionary: {movie title: sub-link} 
2.	save the movie list dictionary into movie_dic.csv for getting reviews and movie info
3.	get movie reviews from each movie sub-link: {Movie title: user name, rating score, rating time, num helpful, review}
4.	save the movie reviews dictionary as reviews_top.csv for build stage 3: collaborative & rating-based recommended model
5.	get movie features from each movie sub-link: [movie title Movie intro, Genre, rating, release year, review num]
6.	save the movie features into df_IMDb_top25_info.csv for build stage 3: content-based recommended model 

## Stage 3: Recommender System
File: Stage_3_Recommender_System.ipynb
Data Files: reviews_250.csv (web crawler from stage 2), dataset_review_1.csv (from stage 1 final step), df_IMDb.csv (from stage 2 movie info), glove.6B.100d.txt (download from glove.6B.zip). 
  
* Collaborative Filtering
1.	load in reviews_250.csv 
2.	create user_id and movie_dictionary
3.	transform review dataset into a set: {'user1’: [movie rated score, 0:no rating score for this movie]}
4.	calculate the cosine similarity
5.	get the most similar taste user
6.	get a mean rating score for each user
7.	predict the final score for the target user for one movie
8.	get a prediction for each non-rating movie and get top K movies with high predicted scores
9.	assume user rates "the Dark Knight" movie with a score of 10, get ten recommendation movies for the target user (sample output)  

* Collaborative Filtering with sentiment analysis
11.	load sentiment analyzed dataset from dataset_review_1.csv
12.	add sentiment_key into the reviews_250 table
13.	transform review dataset into a set: {'user1’: [movie rated score, 0:no rating score for this movie]}
14.	calculate the cosine similarity
15.	get the most similar taste user
16.	get a mean rating score for each user
17.	predict the final score for the target user for one movie
18.	get a prediction for each non-rating movie and get top K movies with high predicted scores
19.	assume user rate "the Dark Knight" movie with a score of 2, get ten recommendation movies for the target user (sample output)
   
* Content-Based Filtering
1.	load dataset df_IMDb.csv and apply EDA to find some insights from the dataset
2.	convert features moive_title and Movie_intro vectors through GloVe
3.	list all movie genres
4.	inquired user about the favorite genre of movie
5.	list all movie titles within the genre the user selects
6.	requested the name of the favorite movie from users
7.	calculate the cosine similarity between the favorite one and other movies
8.	assume user rate like "the Dark Knight" movie, get ten recommendation movies for the target user (sample output)

*Rule-Based Filtering
1.	load in dataset reviews_250.csv
2.	create a table - rows are user names, columns are movie titles, and values are rating scores
3.	get the correlation between users who recently watched movie vs. remaining all other movies, comparing the correlation matrix between movies
4.	create a data frame based on correlation
5.	filter with several ratings and sort the data frame with correlation column
6.	recommend movies with titles
7.	assume user rate like "the Dark Knight" movie, get ten recommendation movies for the target user (sample output)


