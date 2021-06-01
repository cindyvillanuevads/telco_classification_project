#### PROJECT DESCRIPTION
- Predict customer churn at Telco, a telecommunications company, using a machine learning classification model, u

#### GOALS 

- Find drivers for customer churn at Telco.

- Construct a ML classification model that accurately predicts customer churn.

- Document your process well enough to be presented or read like a report


#### DATA DICTIONARY
---
| Attribute | Definition | Data Type | Values|
| ----- | ----- | ----- | ----- |
|customer\_id|Alpha-numeric ID that identifies each customer| object | uniques ids|
senior_citizen|Indicates if the customer is 65 or older| int64 | 0/1 |
partner|If a customer is married| int64 | 0/1 |
dependents|Indicates if a customer lives with dependents| int64 | 0/1 |
tenure_months |The length of a customers  with Telco  in months|  int64 | numbers |
phone_service|If a customer has phone service| int64| 0/1 |
multiple_lines|If a customer has multiple phone lines| int64| 0 (no phone), 1(1 line), 2(mult. lines) |
online_security|Indicates if a customer has online security add-on| int64 | 0 (no), 1(yes), 3(no internet) |
online_backup|Indicates if a customer has online backups add-on| int64 | 0 (no), 1(yes), 3(no internet) |
device_protection|Indicates if a customer has a protection plan for Telco devices | int64 | 0 (no), 1(yes), 3(no internet) |
tech_support|Indicates whether a customer has technical support add-on| int64 | 0 (no), 1(yes), 3(no internet) |
streaming_tv|Indicates if a customer uses internet to stream tv| int64 | 0 (no), 1(yes), 3(no internet) |
streaming_movies|Indicates if a customer uses internet to stream movies| int64 | 0 (no), 1(yes), 3(no internet) |
paperless_billing|Indicates if a customer is enrolled in paperless billing| int64 | 0/1 |
monthly_charges|The amount a customer pays each month for services with Telco| float64 | float |
total_charges|The total amount a customer has paid for Telco™ services| float64 | float |
|payment\_type\_id | Indicates how a customer pays their bill each month | int64 | int |
female|Gender of the customer = female | unit8 | 0/1 |
male |Gender of the customer = male | unit8 | 0/1 |
month_to_month |contract type = month to month | unit8| 0/1 |
one_year |contract type = one year | unit8 | 0/1 |
two_year |contract type = two year | unit8 | 0/1 |
dsl |Indicates the type of internet service as dsl |  unit8 | 0/1 |
fiber_optic |Indicates the type of internet service as fiber optic |  unit8 | 0/1 |
has_internet|Indicates if a customer has internet |  unit8 | 0/1 |
bank_transfer_(automatic)|type of payment| unit8| 0/1 |
credit_card_(automatic) |type of payment| unit8 | 0/1 |
electronic_check |type of payment| unit8 | 0/1 |
mailed_check |type of payment| unit8 | 0/1 |

_____________________________________
| Target | Definition | Data Type |
| ----- | ----- | ----- |
|churn|Indicates whether a customer has terminated service| int64| 0/1 |

#### PROJECT PLANNIG

- Acquire data from the Codeup Database using my own function to automate this process. this function is saved in acquire.py file to import into the Final Report Notebook.
- Clean and prepare data for preparation step. Create a function to automate the process. The function is saved in a prepare.py module. Use the function in Final Report Notebook.
- Define two hypotheses, set an alpha, run the statistical tests needed,document findings and takeaways.
- Establish a baseline accuracy and document well.
- Train three different classification models.
- Evaluate models on train and validate datasets.
- Choose the model that performs the best.
- Evaluate the best model (only one) on the test dataset
- Create csv file with customer_id, probability of churn, and prediction of churn

#### AUDIENCE 
- Codeup Data Science team

#### INITIAL IDEAS/ HYPOTHESES STATED
- 𝐻𝑜 : There is no difference in churn rate between customers with 1 month of tenure  and the customers longer tenure
- 𝐻𝑎 : There is a difference in churn rate between customers with 1 month of tenure  and the customers with longer tenure

#### INSTRUCTIONS FOR RECREATING PROJECT

- [x] Read this README.md
- [ ] Create a env.py file that has (user, host, password) in order to  get the database 
- [ ] Download the aquire.py, prepare.py, model_func.py, explore.py and  final_notebook_project.ipynb into your working directory
- [ ] Run the final_notebook_project.ipynb notebook


#### DELIVER:
- a Jupyter Notebook Report showing process and analysis with goal of finding drivers for customer churn.
- a README.md file containing project description with goals, a data dictionary, project planning (lay out your process through the data science pipeline), instructions or an explanation of how someone else can recreate your project and findings (What would someone need to be able to recreate your project on their own?), and key findings and takeaways from your project.
- a CSV file with customer_id, probability of churn, and prediction of churn. 
- individual modules, .py files, that hold your functions to acquire and prepare your data.
- a notebook walkthrough presentation with a high-level overview of your project (5 minutes max).