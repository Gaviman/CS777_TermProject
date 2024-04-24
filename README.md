# CS777_TermProject

* Environment setup This program is run on google collab. This requires accessing the website at https://colab.research.google.com/ and uploading the code directly. Adding the dataset to collab directly works best instead of referencing a local cop.y Select the file icon on the left side of the screen to add the file directly.

* How to run the code Add the dataset file to the collab session. Right click the file in collab to get the collab pathway, copy and paste this over the existing text string on line 10. The execution button in the shape of a play button. the top left corner should be selected.

* Results of running the code with data The code will create an interface to use certain functions.
* The first option is to process a dataset, so that it will be usable with the two algorithms. The Small NYC dataset has already been processed, so will not need to be processed twice.
* The next two options will run the training and testing for each of the two processing  programs. They will output Accuracy, F1 Score, Weighted Precision, Weighted Recall, Weighted F1 Score, Time (mins).
*                       MLP			  LogReg
Accuracy: 		         	0.9987		0.9988
F1 Score: 		 	        0.9980		0.9982
Weighted Precision:  	  0.9974		0.9976
Weighted Recall: 	  	  0.9987		0.9988
Weighted F1 Score:   	  0.9980		0.9982

Time (mins)	:		          12 	      3
* The last button requires that you enter a zip code in the top text field. It will return the probability of lethality given the median property price of that zipcode.

* The full dataset is 2 million rows of traffic accident reports from NYC, with the median house values of the zip codes in which they occured.
* The small dataset is 1000 rows of same.
* The results show that overall probability of lethality is low per incident, with a probability of .0014.
* This probablity goes as low as .0012 in the wealthiest zip codes.
* In absolute terms, this is small, but it's a relative 16% difference in lethality per incident.
