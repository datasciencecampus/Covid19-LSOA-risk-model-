import pandas as pd
from utils import config as conf
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression

def normalise(df, columns,by=None, suffix=''):
    """
    Normalise columns for given df
    """
    if by is None:
        by = df[columns].sum(axis=1)
    for column in columns:
        name = column+ suffix
        df[name]=df[[column]].div(by,axis=0)
    
    return df

def combining_and_remap_travel_cluster(df):
    """
    This function combines smaller travel clusters and then renames them so they are ordered.
    
    Params:
    df: pandas dataframe
    """

    travl_clst_comb_dict={'L3. Transition area from L2 to L4': 'L4. >70% suburban dwellers',
       'L5. Transition area from L4 to L6': 'L4. >70% suburban dwellers',
       'L7. Transition area from L6 to L8':'L6. >70% exurban dwellers'}    #these are being aggregated
    
    

    df['travel_cluster']=df['travel_cluster'].map(travl_clst_comb_dict).fillna(df['travel_cluster'])

    #KJ : in renaming, could we remove >70%? this was the case before aggregating for instance: L1. metropolitan core dwellers. This needs to be updated in models, etc.  
    
    travl_renam_dict={'L1. >70% metropolitan core dwellers':'L1. >70% metropolitan core dwellers',
                 'L2. >70% outer metropolitan dwellers':'L2. >70% outer metropolitan dwellers',
                 'L4. >70% suburban dwellers':'L3. >70% suburban dwellers',
                 'L6. >70% exurban dwellers':'L4. >70% exurban dwellers',
                 'L8. >70% rural dwellers':'L5. >70% rural dwellers'}   #being renamed to follow a sequential pattern

    df['travel_cluster']=df['travel_cluster'].map(travl_renam_dict).fillna(df['travel_cluster'])
    
    
    return df

# TA: come back to this...
# KJ : could we add the default here -  also not quite sure what subgroup is doing. Does it mean getting all levels which are not in umbrella_ethnicity? why? 
# KJ: what other ethnic groups are in umbrella ethnicity- is this already aggregation of all other groups? Can see that the normalisation is happening in the notebook; can we move that here? Can we make sure that normalisation is by total ethnic groups as opposed to only ethnic groups selected for analysis? 
# KJ: Also worried about hard codded matters -e.g. focus on columns started with CENSUS_2011; shall we remove subgroups and only focus on whatever we have on config file? Happy to discuss this. 

def get_ethnicities_list(df,subgroups):
    """
    This function gets the ratios of different ethnicities. Option to either get ratios of umbrella ethnicity groups or granular sub-groups.
    
    Params:
    df: pandas dataframe
    subgroups: bool, if true only granular ethnic groups extracted from ratio
    """
    ethnicities = conf.features_dict['umbrella_ethnicity']
    
    #ask question here, is it meant to be this or subgroups if true?
    
    if subgroups:
        ethnicities= [column for column in df.columns.to_list() if 'CENSUS_2011' in column if column not in ethnicities] #granular ethnic groups

    return ethnicities

def divide_comm_vars_by_area(df):
    """
    This function divides selected variables by area
    """
    commnl_ftrs=conf.features_dict['commnl_ftrs']
    shared_dwellings=conf.features_dict['shared_dwellings']
    
    ftw_cols=conf.features_dict['ftw_cols']

    cols_to_divide=commnl_ftrs+shared_dwellings+ftw_cols
    
    df[cols_to_divide]=df[cols_to_divide].div(df['Area'],axis=0)
    
    return df

# KJ: had problem running this; one index missing from data included in config file. This function can also benefit from .normalise function above. can become shorter. 
def aggregating_proportion_health(df):
    """
    This function will aggregate all the good health vars and take the proprtion of good and bad health
    """
    
    good_health_ftrs=conf.features_dict['good_health_ftrs']
    bad_health_ftrs=conf.features_dict['bad_health_ftrs']
    
    Good_health_sum = df[good_health_ftrs].sum(axis=1)#sum up good health
    bad_health_sum = df[bad_health_ftrs].sum(axis=1)#sum up bad health

    
    all_health = good_health_ftrs +bad_health_ftrs #extend the list to also get bad health
    df['GOOD_FAIR_HEALTH']=Good_health_sum.div(df[all_health].sum(axis=1)) #get % of people with good health in each lsoa
    df['BAD_HEALTH']=bad_health_sum.div(df[all_health].sum(axis=1)) #get % of people with bad health in each lsoa
    
    df=df.drop(columns=all_health)

    return df

# KJ: can we remove dictionaries like this below to config for people being able simply to update it when needed. prefer to have no hard code in functions in data.py. 
def create_occupation_groups(df):
    """
    Get occupation groups
    """
    
    occpn_ftrs=conf.features_dict['occpn_ftrs']
    
    occpn_dict={'OCCUPATION_MANAGERS_DIRECTORS_AND_SENIOR_OFFICIALS':'1_prof_other',
            'OCCUPATION_SCEINCE_RESEARCH_ENGINEERING_AND_TECHNOLOGY_PROFESSIONALS':'1_prof_other',
            'OCCUPATION_HEALTH_PROFESSIONALS':'1_prof_healthcare',
            'OCCUPATION_BUSINESS_MEDIA_AND_PUBLIC_PROFESSIONALS':'1_prof_other',
            'OCCUPATION_SCIENCE_ENGINEERING_AND_TECHNOLOGY_ASSOCIATE_PROFESSIONALS':'1_prof_other',
            'OCCUPATION_HEALTH_AND_SOCIAL_CARE_ASSOCIATE_PROFESSIONALS':'1_prof_healthcare',
            'OCCUPATION_HEALTH_ASSOCIATE_PROFESSIONALS':'1_prof_healthcare',
            'OCCUPATION_WELFARE_AND_HOUSING_ASSOCIATE_PROFESSIONALS':'1_prof_other',
            'OCCUPATION_PROTECTIVE_SERVICE_OCCUPATIONS':'1_Group_skilled',
            'OCCUPATION_CULTURE_MEDIA_AND_SPORTS_OCCUPATIONS':'1_Group_skilled',
            'OCCUPATION_BUSINESS_AND_PUBLIC_SERVICE_ASSOCIATE_PROFESSIONALS':'1_prof_other',
            'OCCUPATION_ADMINISTRATIVE_AND_SECRETARIAL_OCCUPATIONS':'1_prof_other',
            'OCCUPATION_SKILLED_AGRICULTURAL_AND_RELATED_TRADES':'1_Group_trade',
            'OCCUPATION_SKILLED_METAL_ELECTRICAL_AND_ELECTRICAL_TRADES':'1_Group_trade',
            'OCCUPATION_SKILLED_CONSTRUCTION_AND_BUILDING_TRADES':'1_Group_trade',
            'OCCUPATION_TEXTILES_AND_GARMENTS_TRADES':'1_Group_trade',
            'OCCUPATION_PRINTING_TRADES':'1_Group_trade',
            'OCCUPATION_FOOD_PREPARATION_AND_HOSPITALITY_TRADES':'1_Group_trade',
            'OCCUPATION_CARING_PERSONAL_SERVICE_OCCUPATIONS':'1_Group_skilled',
            'OCCUPATION_LEISURE_TRAVEL_AND_RELATED_PERSONAL_SERVICE_OCCUPATIONS':'1_Group_skilled',
            'OCCUPATION_SALES_OCCUPATIONS':'1_Group_skilled',
            'OCCUPATION_CUSTOMER_SERVICE_OCCUPATIONS':'1_Group_skilled',
            'OCCUPATION_PROCESS_PLANT_AND_MACHINE_OPERATIVES':'1_Group_skilled',
            'OCCUPATION_TRANSPORT_AND_MOBILE_MACHINE_DRIVERS_AND_OPERATIVES':'1_Group_skilled'}
    
    df_occpn=df[occpn_ftrs].copy()
    
    df_occpn.rename(columns=occpn_dict,inplace=True)
    
    df_occpn=df_occpn.groupby(axis=1, level=0).sum()
    
    df=pd.concat([df,df_occpn], axis=1)
    
    return df


def get_total_vaccinations_first_second(df, normalise):
    """
    Total Vaccinations and choose if to normalise by people
    
    df:pandas df
    normalise:bool, normalise by area if true
    """
    
    df['total_vaccinated_first_dose']=df['dose_first_vaccine_male']+df['dose_first_vaccine_female']

    df['total_vaccinated_second_dose']=df['dose_second_vaccine_male']+df['dose_second_vaccine_female']
    
    if normalise==True:
        df['total_vaccinated_first_dose']=df['total_vaccinated_first_dose'].div(df['ALL_PEOPLE'],axis=0)
        df['total_vaccinated_second_dose']=df['total_vaccinated_second_dose'].div(df['ALL_PEOPLE'],axis=0)
   
    
    return df


def sklearn_vif(exogs, data):
    '''
    This function calculates variance inflation function
    in sklearn way. 
    Output of this function is used to test for 
    multi-collinearity between the variables.
    Most research papers consider a VIF (Variance Inflation Factor) > 10
    as an indicator of multicollinearity, but some choose a
    more conservative threshold of 5 or even 2.5.
    See reference below:
    https://quantifyinghealth.com/vif-threshold/#:~:text=Most%20research%20papers%20consider%20a,of%205%20or%20even%202.5.
    '''
    # initialize dictionaries
    vif_dict, tolerance_dict = {}, {}

    # form input data for each exogenous variable
    for exog in exogs:
        cols=data.columns
        data=StandardScaler().fit_transform(data.values)
        data=pd.DataFrame(data,columns=cols)

        not_exog = [i for i in exogs if i != exog]
        X, y = data[not_exog], data[exog]
        
        # extract r-squared from the fit
        r_squared = LinearRegression().fit(X, y).score(X, y)
         
        # calculate VIF
        vif = 1/(1 - r_squared)
        vif_dict[exog] = vif

        # calculate tolerance
        tolerance = 1 - r_squared
        tolerance_dict[exog] = tolerance

    # return VIF DataFrame
    df_vif = pd.DataFrame({'VIF': vif_dict, 'Tolerance': tolerance_dict})

    return df_vif
