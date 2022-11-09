import threading as t

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


##################
cat_vars = ['bedrooms', 'bathrooms', 'fips']
cont_vars = ['sq_feet', 'tax_value', 'year_built','tax_amount']
palettes = ['flare', 'Blues_r', 'PuRd_r', 'Accent']
colors_sns = sns.color_palette("flare")
##################

def pairplot_data(df):
    '''
    the function take a zillow data frame as an argument
    creates a random sample n=100_000 for faster visualiztions
    creates a pairplot with regression line for numeric variables
    '''
    # draw a sample
    #sample = df.sample(100_000, random_state=2912)
    
    #define columns
    col_pairplot = ['bedrooms', 'bathrooms', 'sq_feet', 'tax_value', 'year_built', 'tax_amount']
    
    # create a pairplot
    sns.pairplot(data=df[col_pairplot], kind='reg', plot_kws={'line_kws':{'color':'red'}}, corner=True)
    plt.show()

def correlation_and_heatmap(df):
    '''
    accepts a zillow data frame as an argument
    creates a correlation matrix and displays a heatmap
    '''
    
    # create a correlation matrix to see if there are liner correlations between variab
    tax_corr = df.drop(columns=['id', 'fips']).corr(method='spearman')
    display(tax_corr)
    
    # pass my correlation matrix to Seaborn's heatmap
    # trim the upper corner no remove duplicates
    sns.heatmap(tax_corr, cmap='Purples', annot=True, 
                mask=np.triu(tax_corr))
    plt.show()

########################

def show_categ_vars(df, cat_vars=cat_vars):
    print('Categorical Variables:')
    plt.figure(figsize=(16,4))
    plt.suptitle('Categorical variables')
    for i, col in enumerate(cat_vars):
        plt.subplot(1, 3, i+1)
        sns.histplot(data=df, x=col, bins=10, stat = 'percent', color=colors_sns[i])
        plt.vlines(df[col].mean(), 0, 70)
        plt.ylim(0,70)
        plt.title(col)
    plt.show()

def show_cont_vars(df, cont_var=cont_vars):
    #sample = df.sample(100_000, random_state=2912)
    print('Continuous Variables:')
    plt.figure(figsize=(20, 4))
    for i, col in enumerate(cont_vars):
        plt.subplot(1, 4, i+1)
        plt.title(col)
        sns.boxplot(x=col, data=df, color=colors_sns[i])
        
def show_cat_vs_cont(df, cat_vars=cat_vars, cont_vars=cont_vars):
    print('Categorical vs Continuous Variables:')
    #number = 1
    for j, cont in enumerate(cont_vars):
        plt.figure(figsize=(20,4))
        plt.suptitle(cont)
        for i, cat in enumerate(cat_vars):
            plt.subplot(1, 4, i+1)
            sns.barplot(data=df, x=cat, y=cont, palette=palettes[j])
            plt.title(cat + ' vs ' + cont)
        plt.show()

###########
#### combine 3 func with categorical and continuous vars #####
def plot_categorical_and_continuous_vars(df, cat_vars, cont_vars):
    t1 = t.Thread(target=show_categ_vars(df, cat_vars))
    t2 = t.Thread(target=show_cont_vars(df, cont_vars))
    t3 = t.Thread(target=show_cat_vs_cont(df, cat_vars, cont_vars))
    
    if __name__ =="__main__":
        #print('Categorical Variables:')
        t1.start()
        t1.join()
        #print('Continuous Variables:')
        t2.start()
        t2.join()
        #print('Categorical vs Continuous Variables:')
        t3.start()
        t3.join()