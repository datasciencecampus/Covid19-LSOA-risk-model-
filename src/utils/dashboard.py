import pandas as pd

def make_quintiles(df):
    """
    Add a quintile score column for each numerical column in DataFrame 'df'
    
    :param df: A dataframe
    :type df: Pandas DataFrame
    
    :return: DataFrame with extra columns containing quintile information
    :rtype: Pandas DataFrame
    """
    
    df_quint = df.copy()
    
    for col in df_quint.select_dtypes(exclude=['object']).columns:
            
            # by default quintiles are numbered 0-4, add one to result in a scale from 1 to 5
            df_quint[col + '_quint'] = pd.qcut(df[col], 5, labels=False).astype('int') + 1
            
    return df_quint

def encode_column(df, colname):
    """
    Create a new column of encodings for a categorical column given as 'colname' in DataFrame 'df'
    
    :param df: A dataframe
    :type df: Pandas DataFrame
    
    :param colname: Name of a column in DataFrame 'df'
    :type colname: string
    
    :return: DataFrame with the specific column encoded
    :rtype: Pandas DataFRame
    """
    
    df_encode = df.copy()
    
    df_encode[colname] = df[colname].astype('category')
    
    df_encode[colname + '_encode'] = df_encode[colname].cat.codes
    
    return df_encode

def pivot_results(df):
    """
    Pivot DataFrame 'df' into long format for easier plotting in Data Studio
    
    :param df: A dataframe
    :type df: Pandas DataFrame
    
    :return: DataFrame pivoted into long format
    :rtype: Pandas DataFrame
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
    
    :param df: A dataframe
    :type df: Pandas DataFrame
    
    :param colname: Name of a column in DataFrame 'df'
    :type colname: string
    
    :param lookup: A dictionary of key:value pairs that map value names to pretty names
    :type lookup: dictionary
    
    :return: A DataFrame with values replaced according to the specified lookup dictionary
    :rtype: Pandas DataFrame
    
    """
    
    df[colname].replace(to_replace = lookup, inplace=True)
    
    return df

