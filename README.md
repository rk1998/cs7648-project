# cs7648-project
Repo for CS 7648: Interactive Robot Learning Final Project: Applying Active Learning for Sentiment Analysis

This repo contains an implementation for an Active Learning Pipeline that allows a Sentiment Classification Model to learn how to label tweets using a small set of labeled data. 
The pipeline will query a set of unlabeled tweets to be labeled by a human annotator. We experimented with different acquisition functions that determine which tweets will help
the model learn most about the unlabeled dataset. We trained our models on the Sentiment-140 Twitter dataset which contains 1.5 million tweets with positive and negative sentiment labels. The data can be found here: https://www.kaggle.com/kazanova/sentiment140

Implemented a CNN-based Neural Network for Sentiment Classification using pytorch. The original paper for this architecture can be found here: https://arxiv.org/pdf/1408.5882.pdf



Link to Final Presentation: https://docs.google.com/presentation/d/16g83CSRMItJjRbkhx2ZDgRpDDaXu_Q9cr85_UnrvStY/edit?usp=sharing
https://docs.google.com/presentation/d/16g83CSRMItJjRbkhx2ZDgRpDDaXu_Q9cr85_UnrvStY/edit?usp=sharing
