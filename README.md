# symptom-tree
Medical symptom/diagnosis decision tree built with Python's scikit-learn

## Contributors:
James Jensen, Tammy Glazer, and Tim Hannifan

## Purpose: 
The purpose of this tree is to leverage historical symptom and diagnosis
patterns to facilitate self-diagnosis. To accomplish this task, we utilize a
non-parametric supervised learning method and build a Decision Tree that will
predict a diagnosis given a set of symptoms and demographic characteristics.
Our model makes these predictions using National Ambulatory Medical Care
Survey data, which includes information for 218,409 individuals who have
received outpatient medical care between 2012 and 2016, paired with diagnoses
scraped from an online ICD-10-CM Medical Coding Reference. Our model achieves
a 0.26 accuracy score for 755 unique symptoms and 866 unique diagnoses using
115,150 rows of data after cleaning.

## Usage: 
0) Download the data [here](https://s3.amazonaws.com/symptom-tree/symptom-tree-data.csv) and save it to the `data\` directory.

1) To initialize the tree, run the following in ipython:

`tree = symptomtree.buildtree(‘data/symptom-tree-data.csv’)`

This function reads and processes the data file, then initializes the SymptomTree class using this processed data. This class contains attributes for the DecisionTreeClassifier model (model), the cleaned NAMCS dataset (data), a dictionary mapping diagnoses to unique identifier codes (diagnosis dict), a dictionary  mapping unique codes to diagnosis strings (rev_diagnosis_dict), the x training dataset (x_train), the y training dataset (y_train), the x testing dataset (x_test), the y testing dataset (y_test), predicted diagnoses (y_hat), and a lookup attribute.

2) Once the model is built, run the following method to construct a user
form, which represents the exact demographic and symptom text that can
be passed into the predict_user_diagnosis method given the training data:

`tree.get_user_form()`

3) To obtain a prediction for a given set of demographic characteristics
and symptoms, run the following, entering a list of symptoms and/or 
demographic characteristics matching the options from the user_form:

`symptoms = [‘example_age’, ‘example_symptom’, ‘example_symptom’]`
`tree.predict_user_diagnosis(symptoms)`

NB: this step will only work if the symptoms/demographics are spelled
exactly as they appear in the user_form, including any prefixes
(ie. ‘KE_arm pain’)

If needed, you can create a JSON file with ICD codes and their corresponding
diagnoses. Call read_and_process_data() in process.py, passing in the csv filename. Then call 
make_dictionary() in icd.py, passing in the dataframe. This will write the
dictionary to a JSON file.

## Data Sources:
The National Ambulatory Medical Care Survey (NAMCS) is a national survey
designed to provide objective, reliable information about the provision and
use of ambulatory medical care services in the United States. Findings from
the survey are based on a sample of visits to non-federally employed
office-based physicians. The survey has been conducted annually since 1989. We
draw from data collected between 2012 and 2016 (2012 was the first year that
data was collected using an automated laptop-assisted data collection method).
The data is designed for consumption using SAS, SPSS, or Stata, and was
downloaded from a Center for Disease Control repository as a ‘.dta’ file.
The basic sampling unit for NAMCS community health centers (CHC) in the
provider-patient visit. 

The International Classification of Diseases, Ninth Revision, Clinical
Modification (ICD-10-CM) is a system used by physicians and other healthcare
providers to classify and code all diagnoses and procedures recorded in
conjunction with hospital care in the United States. It provides a level of
detail that is necessary for diagnostic specificity, and is based on the
International Classification of Diseases published by the World Health
Organization (WHO). This system uses alphanumeric codes to identify known
diseases and other health problems. All HIPAA-covered entities must adhere
to ICD-10-CM codes. The structure of an ICD-10-CM code is as follows:
the first three characters categorize an injury or illness, and the fourth
through sixth characters describe in greater detail the cause, anatomical
location, and severity of an injury or illness.

Because the NAMCS data includes diagnoses encoded as ICD-10-CM codes but lacks
the corresponding string representations, our implementation takes an
ICD-10-CM code as an input and scrapes a Medical Coding Reference website for
a string representation. The output of this scraping function is a string
which is then stored as a value in a dictionary associated with an ICD-10-CM
code (key). This dictionary is converted to a JSON file which is finally
merged with the NAMCS dataset.

## Data Cleaning:
Each year of NAMCS data is stored as a separate '.dta' file on the CDC online
repository. Because the data was prepared for SAS, SPSS, or Stata, the data
files for 2012-2016 were loaded and unioned using Stata. Over one thousand
irrelevant columns were dropped and the final dataset was exported to a CSV
and loaded into Python. In Python, data cleaning tasks include:

1) Split strings containing multiple symptoms into separate columns and apply
binary encoding to create a wide dataset with a dummy variable for each
2) Recode incorrect and null values as NaN 
(eg. the “tobacco use” column originally contained values: 3, Blank, Current,
Not current, Unknown; therefore, 3 was recoded to NaN)
3) Ensure that data in each column is in a uniform format
(eg. the 'height_inches' column originally contained numeric values encoded as
strings as well as '72 inches (capped value for females)', recoded to 72
4) Rows containing unhelpful diagnosis codes are dropped
(eg. V990 indicates a non-codable diagnosis, V991 indicates left before being
seen, V992 indicates left before being seen, and V997 indicates no diagnosis)
5) Set all table values that contain text, excluding headers, to lowercase
6) Replace all diagnosis codes in the target column with diagnosis text using
a diagnosis dictionary created from the scraped data
(eg. 338 becomes Central Pain Syndrome)

Multiple strategies were considered for encoding categorical variables to use
with skikit-learn’s Decision Tree module. Because Skikit-learn interprets
numerical features as continuous numeric variables, we identified methods to
avoid inducing order that does not exist in the data, and determined to use
the ‘get_dummies’ function in pandas to convert categorical variables into
dummy variables to accomplish this task efficiently.

## Methodology:
Our implementation utilizes the Decision Tree module from skikit learn to
train a tree using 3 demographic variables and a list of symptoms and predict
an individual’s diagnosis based on manually entered parameters (options
include age category, race, sex, and symptoms). Advantages of Decision Trees
include that they handle both numerical and categorical data, are simple to
interpret, the cost of using the tree is logarithmic in the number of data
points used to train the tree, and they are possible to validate using
statistical tests to account for the reliability of the model. With each
iteration of the model, we made improvements to how variables are encoded and
witnessed the tree’s accuracy score improve from 0.14 to 0.26.

## Additional Research:
In the second iteration of this model, we would be interested to explore the
effect on prediction accuracy of utilizing different machine learning
algorithms (eg. Random Forest, KNN) in addition to decision tree. In addition, while we completed a basic principal component analysis (PCA) to better understand which variables maximize the variance of
the data, we would like to further this analysis to explore the effects of
additional variables available in the NAMCS dataset such as preexisting
conditions, use of tobacco, and pregnancy. We would employ fuzzy matching
so that the end user does not need to enter the symptoms exactly to produce
an accurate output, and would like to design a user-friendly interface to
interact with the predictive feature of the tool.

## Citations
The `util.py` file is a taken from a programming Assignment for CAPP 30122.

## Dependencies
- Pandas v0.24.2
- Beautiful Soup v4.4.0
- Skikit-learn v0.20.3; modules used:
	- tree.DecisionTreeClassifier
	- metrics.accuracy_score
	- preprocessing.StandardScaler
	- decomposition.PCA
	- model_selection.train_test_split
- Graphviz v0.10.1
- Urllib v3.7.3rc1
- Requests v2.21.0