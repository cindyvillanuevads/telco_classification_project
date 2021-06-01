
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def bar_plot (features, df):
    '''
    Take in a features (max 4) to plot  and churn_rate.
    '''
    churn_rate = df['churn'].mean()
    lc = len(features)
    _, ax = plt.subplots(nrows=1, ncols=lc, figsize=(16, 6), sharey=True)
    for i, feature in enumerate(features):
        sns.barplot(feature, 'churn', data=df, ax=ax[i], alpha=0.5, saturation=1)
        ax[i].set_xlabel('Churn')
        ax[i].set_ylabel(f'Churn Rate ={churn_rate:.2%}')
        ax[i].set_title(feature)
        ax[i].axhline(churn_rate, ls='--', color='red')


def plot_tenure (df, tenure):
    '''
    Take in a df and tenure and plot tenure and churn
    '''
    plt.figure(figsize=(15,10))
    # Distribution of Tenure
    sns.histplot(df,
                 x=df[tenure],
                 hue='churn',
                 multiple='stack',
                 binwidth=1          
                 )

    plt.title("Distribution of Tenure")
    plt.xlabel('Tenure (in months)')
    plt.xlim(1, df[tenure].median())

def report_tenure (train, tenure):
    '''
    This function create a report based on a specified tenure. Calculate total customers, churn customers, churn rate,
    % of phone service , % only internet,  % of type of contracts, and electronic cheks and paperless billing for
    the first months of the specified tenure, 
    train: dataframe
    tenure : number of the first month of tenure.
    Example:
    report_tenure (df, tenure)
    
    '''
    #add 1 so the numer given is included in the condiction ['tenure_months'] < tenure
    tenure=tenure+ 1
    #cols has all the columns that I want to check
    cols = ['monthly_charges','month_to_month','one_year', 'two_year','has_internet','dsl','fiber_optic','multiple_lines','electronic_check','paperless_billing', 'tenure_months', 'churn']
    #create a df with the fist months of tenure specified in the function
    df_t1 =train[cols][train['tenure_months'] < tenure ]
    #total of customers that have the specified tenure
    ct1 = train.tenure_months[train['tenure_months'] < tenure].count()
    #total customer with specified tenure and churn =1
    can = df_t1.churn[ df_t1['churn'] == 1].count()
    #create a df and churn =1
    churndf = df_t1[(df_t1['churn']== 1)]
    #df of customers who have service phone
    cols_m = ['multiple_lines','monthly_charges','dsl','fiber_optic', 'electronic_check', 'paperless_billing','month_to_month','one_year','two_year']
    ml_df = churndf[cols_m].groupby( 'multiple_lines').sum()
    #calculate the total of phone service
    phone = ml_df.iloc[1:3, 5:].sum().sum()
    #calculate tonly internet service
    only_int = (ml_df.iloc[0, 1:3].sum()).astype(int)
    #create report of  churn = 1 and tenure < tenure
    cols_1v = ['monthly_charges','electronic_check','paperless_billing','month_to_month', 'one_year','two_year','churn']
    res_df= pd.DataFrame((churndf[cols_1v].sum()),columns=['churn_counts'])
    #calculate nmontlhy charges
    m_char = res_df.iloc[0, 0]
    # let's see the churn rate
    churn_rate = train['churn'].mean()
    #customers with month_to month contracts and have canceled
    m2m =(res_df.loc['month_to_month'][0]).astype(int)
    #customers with mone year contract and churn
    oyc =(res_df.loc['one_year'][0]).astype(int)
    #customers with two year contract and churn
    tyc =(res_df.loc['two_year'][0]).astype(int)
    #customers with electronic_check and have canceled
    ec=(res_df.loc['electronic_check'][0]).astype(int)
    #customers with paperless billing and have canceled
    ppl=(res_df.loc['paperless_billing'][0]).astype(int)
    print(f'                      *** THE FIRST {tenure -1}  MONTH(S) OF TENURE *** ')
    print("")
    print(f'Total customers :     {ct1} ')
    print(f'Total cancellations : {can} ')
    print(f"Churn rate in the first {tenure -1 } month(s) of Tenure: {(can/ct1):.2%}")
    print("")
    print (f"****Overall churn rate: {churn_rate:.2%}******")
    print("")
    print("________________________________________________________________________________")
    print("")
    print(f'                    ** FIRST {tenure-1} MONTH(S) OF TENURE AND CHURN** ')
    print("")
    print(f'Customers with phone service:         {(phone/can):.2%} ')
    print(f'Customers with only internet service: {(only_int/can):.2%} ')
    print(f'Monlthy charges: $ {str(round(m_char, 3))} ')
    print("")

    print("")
    print(f'Month_to_month contracts: {(m2m/can):.2%} ')
    print(f'One year contract:        {(oyc/can):.2%} ')
    print(f'Two year contract:        {(tyc/can):.2%} ')
    print(f"Paperless_billing:        {(ppl/can):.2%}")
    print(f'Electronic_check payment type : {(ec/can):.2%} ')
