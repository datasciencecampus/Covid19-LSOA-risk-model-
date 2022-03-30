import pandas as pd
from utils import config as cf

def normalise(df, columns, by=None, suffix=''):
    """
    Normalise columns for given df
    
    :param df: A DataFrame with columns to be normalised
    :type df: Pandas DataFrame
    
    :param columns: List of column names to which normalisation should be applied
    :type columns: [str]
    
    :param by: Column name which should be the denominator in the normalisation
    :type by: str
    
    :param suffix: String to be added to the column name when normalisation is applied
    :type suffux: str
    """
    if by is None:
        
        # normalise by the sum of the column
        by = df[columns].sum(axis=1)
        
    for column in columns:
        name = column + suffix
        df[name] = df[[column]].div(by, axis=0)
    
    return df

def combining_and_remap_travel_cluster(df):
    """
    This function combines smaller travel clusters and then renames them so they are ordered.
    
    :param df: A dataframe containing a column named 'travel_cluster'
    :type df: Pandas DataFrame
    
    :return: Dataframe 'df' with the contents of the 'travel_cluster' column transformed
    :rtype: Pandas DataFrame
    """

    # define the clusters to be aggregated
    travl_clst_comb_dict = {'L3. Transition area from L2 to L4': 'L4. >70% suburban dwellers',
       'L5. Transition area from L4 to L6': 'L4. >70% suburban dwellers',
       'L7. Transition area from L6 to L8':'L6. >70% exurban dwellers'}   
    
    # apply the name changes using the specified in the dictionary above
    df['travel_cluster'] = df['travel_cluster'].map(travl_clst_comb_dict).fillna(df['travel_cluster'])

    # rename the clusters to preserve a 1 to 5 order of population density
    travl_renam_dict = {'L1. >70% metropolitan core dwellers':'L1. >70% metropolitan core dwellers',
                 'L2. >70% outer metropolitan dwellers':'L2. >70% outer metropolitan dwellers',
                 'L4. >70% suburban dwellers':'L3. >70% suburban dwellers',
                 'L6. >70% exurban dwellers':'L4. >70% exurban dwellers',
                 'L8. >70% rural dwellers':'L5. >70% rural dwellers'}  

    df['travel_cluster'] = df['travel_cluster'].map(travl_renam_dict).fillna(df['travel_cluster'])
    
    return df


def create_time_slice(df, t1, t2):
    '''
    Return the dataframe 'df' sliced between two given dates, 't1' and 't2'
    
    :param df: A DataFrame to which slicing should be applied
    :type df: Pandas DataFrame
    
    :param t1: The beginning of the time slice
    :type t1: string
    
    :param t2: The end of the time slice
    :type t2: string    
    
    :return: A slice of DataFrame df between t1 and t2
    :rtype: Pandas DataFrame
    '''
    
    return df[(df['Date'] > t1) & (df['Date'] <= t2)]


def get_ethnicities_list(df, subgroups):
    """
    This function returns a list of ethnicities. If subgroups = False then the list of umbrella ethnicities from
    the config file is return. If subgroups = True then the function returns a list of column names from a dataframe
    passed to the 'df' argument. The column names must be from the 2011 census and most not be in the umbrella ethnicities 
    list defined in the config file.
    
    The purpose of identifying the ethnicity subgroup columns is so that they can be dropped from the dataframe
    later in the preprocessing script
    
    :param df: A dataframe for which we want to check for the presence of census ethnicity features
    :type df: Pandas DataFrame
    
    :param subgroups: Boolean for whether to check the column names of the dataframe, or simply extract
    a list of ethnicity feature names from the config file
    :type subgroups: bool
    
    :return: A list of ethnicity feature names
    :rtype: [str]
    """
    
    ethnicities = cf.features_dict['umbrella_ethnicity']['columns']
    
    if subgroups:
        
        ethnicities = [column for column in df.columns.to_list() if 'CENSUS_2011' in column if column not in ethnicities] 

    return ethnicities