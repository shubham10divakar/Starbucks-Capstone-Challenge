# Machine Learning Engineer Nanodegree
## Capstone Proposal
Ajitesh Sakaray  
December 30th, 2019

## Proposal

### Domain Background
Starbucks Corporation is an American coffee company and coffeehouse chain. Starbucks was founded in Seattle, Washington, in 1971. As of early 2019, the company operates over 30,000 locations worldwide.

Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks.

Not all users receive the same offer, and that is the challenge to solve with this data set.

Every offer has a validity period before the offer expires. As an example, a BOGO offer might be valid for only 5 days. You'll see in the data set that informational offers have a validity period even though these ads are merely providing information about a product; for example, if an informational offer has 7 days of validity, you can assume the customer is feeling the influence of the offer for 7 days after receiving the advertisement.

If the company sends more offers or irrelavent offers to a person then there is no use. That's why they should send only relavent offers to a person who is likely going to use this offer which benefits the company.We need to predict who are more likely going to use a certain offer and send that offer only to them to make company beneficial.

My motivation behind choosing this problem are, firstly this is a more releastic problem which I see in the day-to-day like where I will be receiving the offers which I don't really like. Second thing is felt this as challenging task to be completed.


### Problem Statement
Building a model that predicts whether or not someone will respond to an offer based on given demographics and offer type.Thus there will be two possible outputs from the model, making it **Binary Classification**: 
1) A person will respond to an offer (1)
2) A person will not respond to an offer(0)

The metrics that will be used in solving this problem is **Accuracy**. Based on this metric we will calculate how likely a person is going to respond to this offer making it as 0 or 1.


### Datasets and Inputs
The data is contained in three files:

1. portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
2. profile.json - demographic data for each customer
3. transcript.json - records for transactions, offers received, offers viewed, and offers completed

Here is the schema and explanation of each variable in the files:

**1. portfolio.json**

* id (string) - offer id
* offer_type (string) - type of offer ie BOGO, discount, informational
* difficulty (int) - minimum required spend to complete an offer
* reward (int) - reward given for completing an offer
* duration (int) - time for offer to be open, in days
* channels (list of strings)

**2. profile.json**

* age (int) - age of the customer
* became_member_on (int) - date when customer created an app account
* gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
* id (str) - customer id
* income (float) - customer's income

**3. transcript.json**

* event (str) - record description (ie transaction, offer received, offer viewed, etc.)
* person (str) - customer id
* time (int) - time in hours since start of test. The data begins at time t=0
* value - (dict of strings) - either an offer id or transaction amount depending on the record

Here we can see some data is in number form and some data is in characters. Since a Machine Learning Model doesn't take input in characters, we need to convert the field which are not numbers to numbers by using **one-hot-encoding** method. Since some of the values are high in number, it will be computationally expensive task. That's why need to reduce these values such as field **_income_** in **profile.json** using **PCA** method.
 

### Solution Statement
The solution for this problem will be to combine the three files and make a single file or table which will have all the demographics of the person and offer details and the predicted output (i.e., a person will or not respond to the offer given).

By using this table we will pass the input to the model and get the output as 0 or 1.

We need to take **transcript.json** file or table first. Here by using the field **_person_**, we can extract person details from **profile.json** file using **_id_** field.

By using the field **_value_** in **transcript.json** file which has a json object **_offer_id_**, we can extract the details of the offer from **potfolio.json** file using **_id_** field.


### Benchmark Model
Since it is a binary classification problem we need to use a Binary Classifier to solve this problem which will get output as 0 or 1. We can also try to use Deep Neural Networks to solve and check on which model it performs better by tuning the hyperparameters.

We get the output of this model as the percentage of which a person is likely to respond an given offer. Based on the percentage we can consider that if output is less than 50% then the person is likely to reject an offer.


### Evaluation Metrics
Evaluation  metrics for this model will be based on **Accuracy percentage**.

``Accuracy = No. of predicted correct / No. of actual correct``


### Project Design
The implementation workflow for this project will include three steps.
1. Data Preparation
2. Model Training
3. Model Tuning
4. Model Prediction

#### 1. Data Preparation :

* Building the Dataset : We have 3 datasets namely portfolio.json, profile.json and transcript.json. We need to combine these 3 datasets to make a single dataset using the common field in each dataset. For example we have field **_customer_id_** as common field between profile.json and transcript.json.

* Feature Exploration : Checking the co-relation between different fields in the table and understand the dataset. If any fields are more similar to each other then we can remove these redundant features and if require add more features from them.

* Preparing the Dataset : As the Machine Learning Model doesn't understand categorical data we need to use one-hot-encoding method
for the fields with categorial data. For example the field **_gender_** in profile.json has categorial values.So we need to do one-hot_encoding for this field also we need to normalize the data using PCA so that a model can be trained faster.

#### 2. Model Training :

As we are determing if the person will respond to an offer or not, it will be binary classification problem.We can use Binary Classifier or Deep Neural Networks to train the model as this will be ideal for solving this type of problem. After the model has been trained we can check the **accuracy** of the model and decide whether to train the model again or to stop training.

#### 3. Model Tuning :

This step will be done if the previous model is not trained enough or have less accuracy after model training step. Things involved when tuning the model is increasing/decreasing the epochs, hidden layers, learning rate etc and check for the accuracy. This step will be trail and error method as to guess what hyperparameters to change and check for the model improvement.

#### 4. Model Prediction :

This will be the final step of the process. If the model is successfully trained will good amount of accuracy then we are now ready to predict the data.

-----------

