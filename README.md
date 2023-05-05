# cs5293sp23-project3

Name: Shiv Patel

# Project Description (in your own words)

In this project we need to predict the cluster id of a city given its pdf. In order to do that, we need to first train our data on the given list of pdf and then store that model. Once, the model is stored it can be used to predict the cluster id of the new city.

# How to install

- All the required python package names are stored in Pipfile and Pipfile.lock.
- The modules required for this project are given in the Pipfile.
- The following steps will help with the installation of the required packages:
    - Pipenv should be installed
    - To update the environment with the packages use "pipenv sync". This command will install all the required packages.
    - The nltk stopwords needs to be manually downloaded. Following images shows what the error looks if the package is missing and the image below it shows the process to fix it.
    -
    <img width="1392" alt="Screenshot 2023-05-05 at 1 01 58 AM" src="https://user-images.githubusercontent.com/89544171/236387590-c489fa1f-8447-45da-aaa1-b3b038585166.png">
<img width="1356" alt="Screenshot 2023-05-05 at 1 04 15 AM" src="https://user-images.githubusercontent.com/89544171/236387615-dd69d752-77d9-4c88-89c7-18a42be519a2.png">


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
    - test_ReadPdf.py
        - This functions is used to test if the read function is working correctly. I read in a file and check if all the raw data is being read and concated in one row before being returned to predict().
        
    - test_Predict.py
        - This function is used to test if the 'AZ Tucson.pdf' is being given a cluster is of 1 or not. If it is then the test is successful since 1 is the cluster value for AZ Tucson in smartcity_eda.tsv.
        

# Bugs and Assumptions

- I only have 2 clusters as it was the optimal when I ran K-means. K-means uses random initialization of the centroids which may lead to different optimal k values every time the project3.ipynb file is run.
    - My predict test case won't pass since the clusters would be different.
- I am assuming optimal k to be 2.

# Video Link

- My video was bigger than what could be uploaded to Github, so uploaded it to mymedia and its link is as follows:
    - https://mymedia.ou.edu/media/t/1_fq8jpoyc
