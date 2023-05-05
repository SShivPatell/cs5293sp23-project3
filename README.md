# cs5293sp23-project3

Name: Shiv Patel

# Project Description (in your own words)

In this project we need to predict the cluster id of a city given its pdf. In order to do that, we need to first train our data on the given list of pdf and then store that model. Once, the model is stored it can be used to predict the cluster id of the new city.

# How to install

- All the required python package names are stored in Pipfile and Pipfile.lock.
- The modules required for this project are:
     - json
     - argparse (used to pass in the arguments when program is run from terminal)
     - sklearn (used to import Count Vectorizer and Cosine similarity)

- The following steps will help with the installation of the required packages:
    - Pipenv should be installed
    - To update the environment with the packages use "pipenv sync". This command will install all the required packages.

# How to run

- Download the zip of the project
- Follow the steps of installing the required packages which are described in How to install section
- Set the directory to cs5293-project3/
    - Run the command: pipenv run python project3.py --document 'docs/TX Austin.pdf'
    - This command takes in the pdf name and predicts the cluster it belongs to.
    
- To test the functions we have unit testing function available which can be run using the following command:
  Run the command: "pipenv run python -m pytest"
    - Details about the test functions are given in the Test Function section
  

# Functions

- I created the function readPdf() and Predict() which helps with the prediction. To clean the raw data I am using the same functions I used in project3.ipynb which are taken from text_normalizer.py of the textbook.

- The functions are:   
   - readPdf(fileName, document_Path)
       - This function takes in the file name which in our case is the city name and the document path of where it is located. We read from the pdf and store its content in a dataframe and then return it to the predict function.
       
   - predict(newData)
       - This function takes in the data returned by readPdf function and normalizes it to make it clean. Using the trained model which is stored as model.pkl and the vectorizer (vectorizer.pkl) which was used to predict the model is loaded to predict the cluster id of the new city. The cluster id is returned by the function along with the updated dataFrame which containes the city name, raw text, and clean text.
       
- Main function
    - It is used to read the argument from the command line.
    - It calls readPdf and predict function which returns the cluster id and the dataFrame of the new city.
    - The dataFrame is stored as smartcity_predict.tsv and the cluster is displayed with proper formatting.

# Test Function

- I have 2 tests for the 2 functions described above.
    - test_Read.py
        - This functions is used to test if the read function is working correctly. I read in the yummly file and then test if the number of dishes match the expected number of dishes present in the yummly file. If it matches then the test is passed successfully, else it fails.
        
    - test_TrainAndPredict.py
        - This function is used to test if the output of a given set of ingredients would give the correct prediction using cosine similarity. I check the dictionary returned by trainAndPredict() with the expected output to check if the test passes or not.
        

# Bugs and Assumptions

- I am assuming that the ingredients entered by the user would be a part of one of the dishes given in yummly.json file.
- If an ingredient is not a part of any of the dishes, then I am displaying "The ingredient is not a part of any of the cusine and dishes." so that the user knows that no dish contains the ingredients that were given as the input.
- If there is a tie in the similiraty score of the 2 cuisines or dishes then the order in which the differnt cuisines were read from the json file would be picked. For example, there are 39774 items in yummly.json. If I enter certain ingredients and the similarity score of item 1304 and 34922 are the same, then the cuisine of item 1304 will be display. 


# Video Link

- My video was bigger than what could be uploaded to Github, so uploaded it to mymedia and its link is as follows:
    - https://mymedia.ou.edu/media/t/1_7n8jip1y