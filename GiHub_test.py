
import pandas as pd
import numpy as np
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

def posibilities(opt_list):
    """
    Generates all possible combinations of a list of options, from zero to the length of the input list.

    Parameters:
    opt_list (list): A list of options.

    Returns:
    list: A list of lists, representing all possible combinations of the input options.
    """
    # Initialize an empty list to hold all possible combinations
    list_posibilities=[]
    
    # Iterate over all possible subset sizes of the input list
    for L in range(0, len(opt_list)+1):
        # Generate all possible combinations of the input list for the current subset size
        for subset in itertools.combinations(opt_list, L):
            # Append the current combination to the list of all possible combinations
            list_posibilities.append(list(subset))
    
    # Filter out any empty lists and return the final list of possible combinations
    return list(filter(None, list_posibilities))

def columns_check(df):
    """
    Performs a quality check on a DataFrame's columns by generating summary statistics for each column.

    Parameters:
    df (pandas.DataFrame): A pandas DataFrame.

    Returns:
    pandas.DataFrame: A new DataFrame with columns representing the original DataFrame's columns and rows representing summary statistics for each column.
    """
    # Initialize empty lists to hold summary statistics for each column
    list_col_name=[]
    list_nrows=[]
    list_nulls=[]
    list_nunique=[]
    list_wdata=[]
    list_sample=[]
    
    # Iterate over each column in the input DataFrame
    for col in df.columns:
        # Generate summary statistics for the current column
        col_name = str(col)
        nrows = len(df[col])
        nulls = df[col].isnull().sum()
        wdata = df[col].count()
        nunique = len(df[col].unique())
        unique_col_list = list(df[col].unique())[:40]
        
        # Append the summary statistics to their respective lists
        list_col_name.append(col_name)
        list_nrows.append(nrows)
        list_wdata.append(wdata)
        list_nulls.append(nulls)
        list_nunique.append(nunique)
        list_sample.append(unique_col_list)
    
    # Create a new DataFrame with columns representing the summary statistics and rows representing each original column
    cols_name=['col_name','nrows','wdata','nulls','nunique','sample']
    data={'col_name':list_col_name,'nrows':list_nrows,'wdata':list_wdata,'nulls':list_nulls,'nunique':list_nunique,'sample':list_sample}
    df_new=pd.DataFrame(data).sort_values(by='nunique', ascending=False)
    
    # Return the new DataFrame with sorted summary statistics
    return df_new


def lr_analysis(df, x_cols, y_col, cols_names_short):
    """
    Perform linear regression analysis for all possible combinations of x variables against a y variable.
    The function returns a pandas DataFrame with regression coefficients, p-values, R-squared, standard error,
    and other metrics for each regression model.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The data frame containing all the variables.
    x_cols : list
        A list of strings containing the names of the independent variables to include in the regression models.
    y_col : string
        The name of the dependent variable.
    cols_names_short : list
        A list of strings containing the short names of all the variables.
    
    Returns:
    --------
    lr_models : pandas DataFrame
        A data frame containing the results of all the regression models.
    """
    
    # Drop rows with missing values
    df.dropna(inplace=True)
    
    # Get all possible combinations of x columns
    combination_list = posibilities(x_cols)
    
    # Create an empty data frame to store the results
    df_lr = pd.DataFrame(columns=x_cols)
    df_lr[y_col] = ""
    df_lr['rsquared_adj'] = ""
    df_lr['pvalues'] = ""
    df_lr['p-significant'] = ""
    df_lr['standard_error'] = ""
    df_lr.insert(0, "Event", "")
    
    # Create another data frame for p-values only
    df_lr_pval = pd.DataFrame(columns=x_cols)
    df_lr_pval[y_col] = ""
    df_lr_pval['rsquared_adj'] = ""
    df_lr_pval['pvalues'] = ""
    df_lr_pval['p-significant'] = ""
    df_lr_pval['standard_error'] = ""
    df_lr_pval.insert(0, "Event", "")
    
    i = 0
    for col in combination_list:
        # Select the x and y variables for this regression model
        X = df[col]
        y = df[y_col]
        
        # Fit a linear regression model
        t = LinearRegression().fit(X, y)
        
        # Fit another model with statsmodels to get more information
        X2 = sm.add_constant(X)
        est = sm.OLS(y, X2)
        est2 = est.fit()
        
        # Store the results in the data frame
        df_lr.loc[i, 'Event'] = ', '.join((str(cols_names_short[n]) for n in col)) + ' vs. ' + y_col
        df_lr.loc[i, col] = t.coef_
        df_lr.loc[i, y_col] = t.intercept_
        df_lr.loc[i, 'rsquared_adj'] = est2.rsquared_adj
        df_lr.loc[i, 'standard_error'] = list(est2.bse)
        df_lr.loc[i, 'pvalues'] = list(est2.pvalues)
        df_lr.loc[i, 'p-significant'] = all(p < 0.05 for p in list(est2.pvalues))

        df_lr_pval.loc[i, 'Event'] = "pvalue - " + ', '.join((str(cols_names_short[n]) for n in col)) + ' vs. ' + y_col
        df_lr_pval.loc[i, col] = list(est2.pvalues[1:])
        df_lr_pval.loc[i, y_col] = est2.pvalues[0]
        df_lr_pval.loc[i, 'rsquared_adj'] = est2.rsquared_adj
        df_lr_pval.loc[i,'standard_error']=list(est2.bse)
        df_lr_pval.loc[i,'pvalues']=list(est2.pvalues)
        df_lr_pval.loc[i,'p-significant']=all(p <0.05 for p in list(est2.pvalues))
        i=i+1
    lr_models=df_lr.append(df_lr_pval)
    return lr_models


def sales_Dummy_Data():
    np.random.seed(622)
   # Create a list of advisor IDs, repeating each ID 5 times for a total of 15 records
    advisor_id = ['A001', 'A002', 'A003', 'A004', 'A005'] * 3

    # Create a list of months, repeating each month 5 times for a total of 15 records
    months = ['Jan','Jan','Jan','Jan','Jan','Feb','Feb','Feb','Feb','Feb', 'Mar', 'Mar', 'Mar', 'Mar', 'Mar']

    # Create a random list of number of sales for each advisor
    number_sales = np.random.randint(10, 100, 15)

    # Create a random list of total revenue for each advisor
    total_revenue = np.random.randint(500, 5000, 15)

    # Create a random list of number of calls for each advisor, ensuring that it's always higher than the number of sales
    number_calls = np.random.randint(number_sales.max() + 1, number_sales.max() + 50, 15)

    # Combine the four lists into a pandas DataFrame
    df = pd.DataFrame({'month':months,'advisorID': advisor_id, 'number_sales': number_sales, 'total_revenue': total_revenue, 'number_calls': number_calls})
    return df

def Survey_Dummy_Data():
    np.random.seed(622)

    # Create a list of survey dates
    survey_dates = pd.date_range(start='2023-03-01', end='2023-03-15', periods=100).strftime('%m/%d/%Y')
 
    # Create a list of customer IDs
    customer_id = np.concatenate([np.arange(1011, 1101),[None]*10])

    # Create a random list of NPS scores
    nps_scores = np.concatenate([np.random.randint(0, 9, 30), np.random.randint(9, 11, 70)])

    # Create a random list of whether the customer provided feedback or not
    feedback = np.random.choice([True, False], size=100, p=[0.7, 0.3])

    # Create a list of advisor IDs based on the customer IDs
    advisor_id = ['A001', 'A002', 'A003','A004', 'A005', 'A006', 'A007', 'A008', 'A009', 'A010']


    # Combine the five lists into a pandas DataFrame
    df = pd.DataFrame({'survey_date': survey_dates,'customer_id': customer_id, 'advisor_id': np.random.choice(advisor_id, size=100), 'nps_score': nps_scores, 'feedback_provided': feedback,'count':1})
    return df


def nps_category(nps_value):
    if nps_value >= 9:
        return ("promoter", 1)
    elif nps_value >= 7:
        return ("neutral", 0)
    else:
        return ("detractor", -1)