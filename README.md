# jointMultitaskEmo

1. Introduction

This repo contains an implementation of JMTE described in the paper Emotion Detection and Classification in a 
Multigenre Corpus with Joint Multi-Task Deep Learning by Shabnam Tafreshi and Mona Diab, COLING 2018

Dependencies

2. Dependencies

	- Keras 2.0.8
	- Python 3.6
	- NumPy
	
				
3. Data

Located under directory "data", we evaluated our model on a multigenre emotion corpus consist of 4 genres as they are described in 
section 2 of our paper. 3 enres (blog post, movie review, news title) are available at the moment, and the 4th genre (Tweets) will 
be available by end of 2018. 

File/Folder structure is as follows: 

	- 3 available genres -> dir emo_multigenre
		The available genres in emo_multigenre folder has its own README.md file, if you use this set please cite the 
		following paper: Sentence and Clause Level Emotion Annotation, Detection, and Classification in a Multi-Genre Corpus
	- train.txt
		JMTE training set
	- test.txt
		JMTE test set
		
**Note: Due to Tweeter set not being available to us, the paper results are not verifiable at the moment. The training and test sets are just samples
However, the model is flexible enough to adopt a new genre of your own.

4. Usage

Our model is implemented in jmte.py. 
	

