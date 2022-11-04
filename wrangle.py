import os
import pandas as pd
from env import get_db_url

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
    
    return df 