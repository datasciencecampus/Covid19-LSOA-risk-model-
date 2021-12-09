# A collection of functions that wrangle and clean the model output into a format that can be presented in Google Data Studio

import pandas as pd

## THINGS TO TEST
## IS THE NEW COLUMN CREATED?
## DOES IT CONTAIN MAX VALUE = 5 AND MIN VALUE = 1

def make_quintiles(df):
    """
    Add a quintile score column for each numerical column in DataFrame 'df'
    
    Parameters
    -------------
    
    :df: A dataframe
    :type: Pandas DataFrame
    """
    
    df_quint = df.copy()
    
    for col in df_quint.select_dtypes(exclude=['object']).columns:
            
            # by default quintiles are numbered 0-4, add one to result in a scale from 1 to 5
            df_quint[col + '_quint'] = pd.qcut(df[col], 5, labels=False).astype('int') + 1
            
    return df_quint

## THINGS TO TEST
## IS THE NEW COLUMN CREATED
## DOES IT CONTAIN INTS

def encode_column(df, colname):
    """
    Create a new column of encodings for a categorical column given as 'colname' in DataFrame 'df'
    
    Parameters
    -------------
    
    :df: A dataframe
    :type: Pandas DataFrame
    
    :colname: Name of a column in DataFrame 'df'
    :type: str
    """
    
    df_encode = df.copy()
    
    df_encode[colname] = df[colname].astype('category')
    
    df_encode[colname + '_encode'] = df_encode[colname].cat.codes
    
    return df_encode

## THINGS TO TEST
## DATAFRAME DIMENSIONS

def pivot_results(df):
    """
    Pivot DataFrame 'df' into long format for easier plotting in Data Studio
    
    Parameters
    -------------
    
    :df: A dataframe
    :type: Pandas DataFrame
    """
    
    # columns to stay the same
    id_vars = ['LSOA11CD', 'travel_cluster','travel_cluster_encode']
    
    # columns to pivot into long format
    value_vars = [col for col in df if col not in id_vars]

    df_piv = pd.melt(df, 
                    value_vars = value_vars,
                    id_vars = id_vars,
                    var_name = 'feature',
                    value_name = 'value')

    return df_piv


def pretty_rename(df, colname, lookup):
    """
    Replace datframe values with pretty names for presentation on a dashboard
    
    The names in column 'colname' that appear as keys in a dictionary called 'lookup' 
    are replaced with the values in the dictionary
    
        
    Parameters
    -------------
    
    :df: A dataframe
    :type: Pandas DataFrame
    
    :colname: Name of a column in DataFrame 'df'
    :type: str
    
    :lookup: A dictionary of key:value pairs that map value names to pretty names
    :type: Pandas DataFrame
    
    """
    
    df[colname].replace(to_replace = lookup, inplace=True)
    
    return df






    
    
    
    
    
    