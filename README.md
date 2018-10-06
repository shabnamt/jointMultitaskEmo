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

Our model is implemented in jmte.py located in jmte folder.

Follow the instruction in the python file to use it with your dataset. We proposed a multi-task model with auxiliary layer for mix genre dataset.
We also provide the multi-task model with no auxiliary layer in jmte_with_no_aux.py, this version has no auxiliary layer and can be used with any genre.

- Input training files: training_text.txt, training_label.txt, and training_clause.txt
-- We provide a sample of how your input text, label, and clause model should be formated. Numeric labels are stand for: 
     0: anger
	 1: anticipation
     2: disgust
     3: fear
     4: joy
     5: sadness
     6: surprise
     7: trust
- Input test files: test_text.txt, test_label_.txt, test_clause.txt; numeric labels are the same.

- Output predicted tags: depends on which code you run there will be output (predicted tags) per task:
-- example: task_1_noaux_results.txt

- There will be a pickle file, this file is the tokenizer at the prediction time

**** Note: In both python file look for handle "Important Note", this handle points you to simply change the name of the input files and indexes of the arrays based on the size of your input files
	

