# Large Scale Computing Project: Learning Better Stock Sentiment Labels for StockTwit Messages

Author: Zihua Chen

## Introduction and Motivation
The use of social media for predicting stock market performance has gained popularity, but the noisy nature of messages on platforms like Twitter pose a challenge to such research. Training a sentiment analysis classifier requires a substantial amount of labeled data, which can be resource-intensive and expensive to obtain. StockTwits, a popular social media platform used by investors and traders, provides a valuable resource for stock market sentiment analysis. Users can post messages about various listed stocks and label their messages as "bullish" or "bearish", making it an ideal dataset for sentiment analysis. Since StockTwits is used by many professional investors/traders, there is little incentive to post nonsensical messages, because messages will be read by their fellow colleagues. This further strengthens the value of using such a dataset. 

However, not all StockTwit messages are labeled, as users have the option not to do so. To address this issue, I use a supervised sentiment classifier trained on the labeled data, and then use this classifier to predict labels for the unlabeled data. The goal of this project is to use a subset of the data that is labelled in order to predict the labels of the unlabelled dataset, in a semi-supervised fashion.

## Computation Bottlenecks

The StockTwits API only allows us to get up to 30 tweets at a time for a given stock ticker symbol. The API allows getting only tweets older than some tweet ID, which we need for repeatedly querying the server to get many recent tweets. Furthermore, there is an API rate limit of 200 requests an hour per client, so having a serial solution would significantly prolong and slowdown the data collection process. To address these challenges, I employ AWS services to enhance the speed and scalability of the scraping process. By implementing parallelization using Lambda and Step Functions, I construct a parallel scraping pipeline, reducing processing time and improving the efficiency of data collection.

After scraping the most recent tweets from the top 50 stock ticker symbols, I end up with a dataset of roughly 270,000 tweets. Doing natural language processing tasks and sentiment analysis on such a large dataset would require significant computing resources. Hence, I use AWS Spark to reduce computation time by leveraging parallel processing, in-memory caching, lazy evaluation, and advanced optimizations. These mechanisms make AWS Spark ideal for fast and scalable data processing, analytics, and machine learning on large-scale datasets.

## Workflow of Project

* Obtain the stock ticker symbols of the top 50 companies by market cap listed on NASDAQ [here](https://www.nasdaq.com/market-activity/stocks/screener). 
  * I focus on the 50 largest stock tickers, but a natural extension to this project is to expand to more stocks, and also other stock exchanges
  * Implement parallel web scraping to collect tweets using Stocktwits API
  * Notebook: [Scraping.ipynb](https://github.com/macs30123-s23/final-project-zihua/blob/main/Scraping.ipynb)

* After collecting tweets through my parallel web-scraping implementation, I run sentiment analysis on messages using data that have labels ("bullish" or "bearish" about a certain stock)
  * After training the classification model, I predict labels on unlabelled data, using the stock ticker symbol and natural language processing on tweet messages (Word2Vec)
  * Notebook: [Analysis.ipynb](https://github.com/macs30123-s23/final-project-zihua/blob/main/Analysis.ipynb)

## Data

I scrape the most recent tweets for the top 50 stock tickers on NASDAQ by market cap using the StockTwits API. The final dataset comprises around 270,000 tweets, with the following distribution of labeled observations:

Label | Count
:--- | :---
Bullish  | 55528
Bearish  | 22690
Unlabelled  | 191772

The data is also available in this GitHub repository:  [Data](https://github.com/macs30123-s23/final-project-zihua/tree/main/scraped-data)

I use the following variables in my analysis: 
* Label ("bullish" or "bearish" about a certain stock): Outcome variable
* Stock Ticker: Categorical independent variable representing the stock mentioned in the tweet
* Tweet message: Main predictor variable (independent variable) processed using Word2Vec

## Results
After fitting a logistic regression model on the labeled data (n = 78,218), the model achieves a performance of AUC = 0.728 and accuracy = 0.736 on the training dataset. Using this model, I make predictions on the unlabeled dataset.

## Conclusion and Next Steps

In this large-scale computing project, my goal was to improve the sentiment labels for StockTwit messages related to stock market sentiment analysis. By leveraging the capabilities of AWS, I successfully constructed a parallel scraping pipeline using AWS Lambda and Step Functions, making the solution highly scalable and efficient.

The motivation behind this project was due to the challenge of obtaining labeled data for sentiment analysis and the noisy nature of social media messages. With StockTwits being a valuable resource for stock market sentiment, I utilized a supervised sentiment classifier trained on labeled data to predict labels for the unlabeled data, through a semi-supervised approach.

To overcome the computation bottlenecks imposed by the StockTwits API and the large dataset size, I employed AWS Spark, a distributed computing framework. AWS Spark reduced computation time through parallel processing, in-memory caching, optimized data sharing, lazy evaluation, and advanced optimizations. This allowed for faster and more scalable data processing, analytics, and machine learning on the extensive dataset.

The workflow of the project involved collecting the most recent tweets for the top 50 stock ticker symbols from the NASDAQ stock exchange using web scraping techniques. Sentiment analysis was then performed on the labeled data, utilizing the stock ticker symbol and natural language processing techniques such as Word2Vec.

The dataset used in the analysis consisted of approximately 270,000 tweets, with varying numbers of observations for each label category. After training a logistic regression model on the labeled data, the model achieved an AUC of 0.728 and an accuracy of 0.736 on the training dataset. Using this trained model, I obtained predictions for the labels of the unlabelled dataset.

In conclusion, this project demonstrates the effectiveness of utilizing large-scale computing techniques and AWS services to enhance sentiment analysis for StockTwit messages. One natural further extension of this project would be to evaluate the predictions of the unlabelled data on downstream tasks, such as explaining stock market prices, or fluctuations is stock market prices. Another possible extension would be to explore the bias that people have in labelling their tweets. The number of "bearish" tweets is less than half of the number of "bullish" tweets, which may suggest that users who are "bearish" about a certain stock could be less willing to tweet messages than users who are "bullish". This could be an interesting topic because this bias affects how well machine learning models are able to learn and predict sentiments of stocks.

## References

Agrawal, S., Azar, P. D., Lo, A. W., & Singh, T. (2019). Momentum, mean-reversion and social media: evidence from StockTwits and Twitter. SSRN.

Batra, R., & Daudpota, S. M. (2018, March). Integrating StockTwits with sentiment analysis for better prediction of stock price movement. In 2018 international conference on computing, mathematics and engineering technologies (ICoMET) (pp. 1-5). IEEE.

Jaggi, M., Mandal, P., Narang, S., Naseem, U., & Khushi, M. (2021). Text mining of stocktwits data for predicting stock prices. Applied System Innovation, 4(1), 13.

Li, Q., & Shah, S. (2017, August). Learning stock market sentiment lexicon and sentiment-oriented word vector from stocktwits. In Proceedings of the 21st conference on computational natural language learning (CoNLL 2017) (pp. 301-310).

Oliveira, N., Cortez, P., & Areal, N. (2013). On the predictability of stock market behavior using stocktwits sentiment and posting volume. In Progress in Artificial Intelligence: 16th Portuguese Conference on Artificial Intelligence, EPIA 2013, Angra do Heroísmo, Azores, Portugal, September 9-12, 2013. Proceedings 16 (pp. 355-365). Springer Berlin Heidelberg.
