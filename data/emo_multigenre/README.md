# Multigenre Corpus

## Introduction

Multigenre corpus is a combination of 3 genres: blog posts, news headlines, and movie reviews.
Blog posts and news headlines were annotated by 6 basic emotion from Ekman (happy, sad, surprise, fear, anger, disgust). 
movie review was annotated for sentiment and sentiment intensity tasks. The detail of the individual 
genres are available in the following paper:

	-Multigenre Corpus with Joint Multi-Task Deep Learning by Shabnam Tafreshi and Mona Diab, LREC 2018
 
Multigenre corpus is annotated with 8 emotion tags from Plutchik wheele of emotion, on sentence and clause level. 
Emotion tags are as follows:

	Tagset = {joy, trust, anticipation, surprise, fear, anger, sandness, disgust}
	 
## File structure

	- emotion_multigenre_corpus_setences.txt
	
		file structure: <sentence-id>	<sentence>	<emotion-tag>	<genre>
		 				234		I had such wonderful time learning about NLP, but I must stop	joy		blog
		
	- emotion_multigenre_corpus_clauses.txt
	
		file structure: <clause-id>	<sentence-id_clause#>	<clause>	<emotion-tag>	<genre>
						1546	234_1	I had such wonderful time learning about NLP	joy				blog
						1547	234_2	but now I must stop	                            no-emotion		blog

## Contact

You can send your emails to Shabnam Tafreshi, email: <shabnamt@gwu.edu>, if you have questions, suggestions, or concerns. 

## Citation

Please cite the following paper:

	-Multigenre Corpus with Joint Multi-Task Deep Learning by Shabnam Tafreshi and Mona Diab, LREC 2018
