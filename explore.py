
import matplotlib.pyplot as plt
import seaborn as sns


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