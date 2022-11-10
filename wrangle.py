import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer

from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression

from env import get_db_url

seed = 2912
target = 'home_vale'

def get_zillow():
    '''
    acuires data from codeup data base
    returns a pandas dataframe with
    'Single Family Residential' properties of 2017
    from zillow
    '''
    
    filename = 'zillow.csv'
    sql = '''
    SELECT *
    FROM properties_2017
    WHERE propertylandusetypeid = 261
    '''

    url = get_db_url('zillow')
    
    # if csv file is available locally, read data from it
    if os.path.isfile(filename):
        df = pd.read_csv(filename) 
    
    # if *.csv file is not available locally, acquire data from SQL database
    # and write it as *.csv for future use
    else:
        # read the SQL query into a dataframe
        df =  pd.read_sql(sql, url)
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index_label = False)
    
    # transform fips to integer
    df['fips'] = df.loc[:, 'fips'].astype(int)
    #remove null values   
    df = df.dropna()

    #dropped very small and unrealistic square feet
    df = df[df.calculatedfinishedsquarefeet > 100]

    df.rename(columns={
        'bedroomcnt':'bedrooms',
        'bathroomcnt':'bathrooms',
        'calculatedfinishedsquarefeet':'sq_feet',
        'taxvaluedollarcnt':'home_value',
        'taxamount':'tax_amount',
        'yearbuilt':'year_built'
    }, inplace=True)

    # add a new column with county names
    df['county_name'] = np.select([(df.fips == 6037), (df.fips == 6059), (df.fips == 6111)],
                             ['LA', 'Orange', 'Ventura'])
    # column to category data type
    df.loc[:, 'county_name'] = df.loc[:, 'county_name'].astype('category')

    #change the type of bedrooms to integer
    df.loc[:, 'bedrooms'] = df.loc[:, 'bedrooms'].astype('uint8')
 
    return df 

def handle_outliers(df):
        # remove outliers
    df = df[df.sq_feet < 15_000]
    df = df[df.bedrooms <= 10]
    df = df[df.bathrooms <= 10]
    df = df[df.home_value <= df.home_value.quantile(0.75)]

    return df

############### FEATURE ENGINEERING FUCNTIONS ########
def select_kbest(X, y, k):
    '''
    the function accepts the X_train data set, y_train array and k-number of features to select
    runs the SelectKBest algorithm and returns the list of features to be selected for the modeling
    !KBest doesn't depend on the model
    '''
    kbest = SelectKBest(f_regression, k=k)
    kbest.fit(X, y)
    return X.columns[kbest.get_support()].tolist()

def rfe(X, y, k):
    '''
    The function accepts the X_train data set, y_train array and k-number of features to select
    runs the RFE algorithm and returns the list of features to be selected for the modeling
    !RFE depends on the model.
    This function uses Linear regression
    '''
    model = LinearRegression()
    rfe = RFE(model, n_features_to_select=k)
    rfe.fit(X, y)
    return X.columns[rfe.get_support()].tolist()

def rfe_model(X, y, model, k):
    '''
    The function accepts the X_train data set, y_train array,
    model (created with hyperparameters) and k-number of features to select
    runs the RFE algorithm and returns the list of features to be selected for the modeling
    '''
    rfe = RFE(model, n_features_to_select=k)
    rfe.fit(X, y)
    return X.columns[rfe.get_support()].tolist()

############### SPLIT FUCNTIONS ########
def split_zillow(df):
    '''
    This function takes in a dataframe and splits it into 3 data sets
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    #split_db class verision with random seed
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed)
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed)
    return train, validate, test

def full_split3_zillow(train, validate, test, target):
    '''
    accepts train, validate, test data sets and the name of the target variable as a parameter
    splits the data frame into:
    X_train, X_validate, X_test, y_train, y_validate, y_test
    '''
    #train, validate, test = train_validate_test_split(df, target)

    #save target column
    y_train = train[target]
    y_validate = validate[target]
    y_test = test[target]

    #remove target column from the sets
    train.drop(columns = target, inplace=True)
    validate.drop(columns = target, inplace=True)
    test.drop(columns = target, inplace=True)

    return train, validate, test, y_train, y_validate, y_test

def full_split_zillow(df, target):
    '''
    the function accepts a zillow data frame a 
    '''
    train, validate, test = split_zillow(df)
    #save target column
    y_train = train[target]
    y_validate = validate[target]
    y_test = test[target]

    #remove target column from the sets
    train.drop(columns = target, inplace=True)
    validate.drop(columns = target, inplace=True)
    test.drop(columns = target, inplace=True)

    return train, validate, test, y_train, y_validate, y_test

#general split function for any data frame
def split_df(df):
    #split_db class verision with random seed
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed)
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed)
    return train, validate, test

#################### SCALING FUNCTIONS #####################

def scale_zillow(train, validate, test):
    '''
    accepts train, validate, test data sets
    scales the data in each of them
    returns transformed data sets
    '''
    #count_columns = ['bedroomcnt', 'bathroomcnt']
    
    #col = train.columns[1:-1]
    col = ['bedrooms',
        'bathrooms',
        'sq_feet',
        'year_built',
        'tax_amount']
    
    # create scalers
    #min_max_scaler = MinMaxScaler()    
    qt = QuantileTransformer(output_distribution='normal')
    qt.fit(train[col])
    train[col] = qt.transform(train[col])
    validate[col] = qt.transform(validate[col])
    test[col] = qt.transform(test[col])
    
    return train, validate, test

