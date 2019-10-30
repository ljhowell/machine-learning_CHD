'''This is the preprocessing module created by Lewis Howell for the Exeter NatSci Machine Learning Group. 30/10/19
The module contains functions for feature selection, dealing with missing values, feature scaling 
and splitting dataset into test and training sets.
	- chose_features
	- drop_missing
	- impute_missing
	- scale_data
	- split_data
'''

print("Importing the preprocessing module for the Exeter NatSci Machine Learning Group.....")


def chose_features(dataset, features=[], n_features = -1, v=0, vv =0):
    '''Return reduced dataset with only chosen columns
    - dataset: pandas dataframe of dataset to have columns chosen
    - features (optional, default = all features): list of strings matching features to keep
    - n_features (optional) - if specified, the top n features from the scaled list is chosen: 
    ['glucose', 'age', 'totChol', 'cigsPerDay', 'diaBP', 'prevalentHyp',
        'diabetes', 'BPMeds', 'male', 'BMI', 'prevalentStroke',
        'education', 'heartRate', 'currentSmoker'],
    - v (optional) - Verbose (default 0) int 0 or 1. Print no. of features kept and lost 
    - vv (optional) - Very verbose (default 0) int 0 or 1. Print list of chosen and rejected features
    '''
    features = dataset.columns
    
    if n_features != -1:
        if n_features > len(dataset.columns):
            print('WARNING: chose_features has an error: n_features must be less than the number of columns')
            return(-1)
        else:
            ordered_f = ['TenYearCHD','sysBP','glucose', 'age', 'totChol', 'cigsPerDay', 'diaBP', 'prevalentHyp',
            'diabetes', 'BPMeds', 'male', 'BMI', 'prevalentStroke',
            'education', 'heartRate', 'currentSmoker']
            features = ordered_f[0:n_features+1]

    if v == 1: 
        print('Now selecting chosen features....')
        print('\t * Number of features: ', len(features)-1, '(and "10YearCHD")')
        print('\t * Number of dropped features: ', len(dataset.columns) - len(features))

    if vv == 1:        
        print('Now selecting chosen features....')
        print('\t * Chosen features: ', features)
        print('\t * Dropped features: ',[col for col in dataset.columns if col not in features])
    
    return dataset.copy()[features] #reduced dataset

def drop_missing(dataset,v=0):
    '''Drop rows with any missing values and return dataset with dropped rows. Prints number and percentage of rows dropped
    - Dataset: pandas Dataframe
    - v (optional): verbosity (Int: 0 or 1)
    '''
   
    dataset2 = dataset.copy().dropna().reset_index(drop=True)
    lost = len(dataset) - len(dataset2)
    if v == 1:
        print('Now dropping rows with missing values....')
        print('\t * Dropped {} rows {:.1f}%. {} rows remaining\n'.format(lost,lost/len(dataset)*100,len(dataset2)))

    return dataset2

def impute_missing(dataset, strategy = 'median', v=0, vv=0):
    '''Imputation - alternative to removing missing values.
    Fill all missing with column average (median or mean)
    dataset - Pandas Dataframe to be imputed
    strategy - str (optional) 'median' (default) or 'mean' to fill missing values with
    - v (optional) - Verbose (default 0) int 0 or 1. Print no. of missing and imputed values  
    - vv (optional) - Very verbose (default 0) int 0 or 1. Print list of imputed features with counts and replaced value
    '''
    from sklearn.impute import SimpleImputer
    from pandas import DataFrame
    from numpy import NaN
    my_imputer = SimpleImputer(strategy=strategy)
    dataset2 = DataFrame(my_imputer.fit_transform(dataset),columns=dataset.columns)

    if v == 1: 
        print('Imputing missing values with {}....'.format(strategy))
        print('\t * Number of missing values: ', dataset.isna().sum().sum())
        print('\t * Number of imputed values: ', dataset.isna().sum().sum() - dataset2.isna().sum().sum())
        print('\n')
    if vv == 1:
        subbed = DataFrame(dataset.isna().sum().sort_values(ascending=False),columns=['N_missing'])
        subbed= subbed.assign(Imputed_value=NaN)
        for col in subbed.index:
            if strategy == 'median':
                subbed.loc[col,'Imputed_value'] = dataset[col].median()
            elif strategy == 'mean':
                subbed.loc[[col,'Imputed_value']] = dataset[col].mean()
        print(subbed)
    
    return dataset2


def scale_data(data, method='standard',v=0):
    '''Return dataset scaled by MinMaxScalar or StandardScalar methods from sklearn.preprocessing
    - data: pandas dataframe of data to be scaled
    - method (optional): str of either 'minmax' for MinMaxScalar or 'std' for StandardScaler (default arg)
    - v (optional -default = 0): Verbose
    '''
    from sklearn import preprocessing
    from pandas import DataFrame

    if v == 1:
        print("Scaling data....\n\t * Using {} scaling".format(method))    

    if method == 'minmax':
        scaler_minmax = preprocessing.MinMaxScaler((0,1))
        return DataFrame(scaler_minmax.fit_transform(data.copy()),columns=data.columns) 
    
    elif method == 'standard':
        scaler_std = preprocessing.StandardScaler() #with_std=False
        return DataFrame(scaler_std.fit_transform(data.copy()),columns=data.columns)
    
    else:
        print('\nscale_data encountered a failure!! Check parameters\n')
        return(-1)

def split_data(dataset,dep_var='TenYearCHD', test_size = 0.2, v = 0):
    '''Split the dataset, return X_train, X_test, y_train, y_test as Pandas Dataframes
    - dataset: Pandas Dataframe. Data to split into training and test data
    - dep_var (optional, default = 'TenYearCHD'): string. Name of column to be dependant variable
    - test_size (optional, default = 0.2): float (0.0-1.0). Proportion of total data to make up test set.
    - v (optional -default = 0): Verbose
    Returns 4 datasets in order: X_train, X_test, y_train, y_test
    '''
    from sklearn.model_selection import train_test_split
    y = dataset[dep_var]
    X = dataset.drop([dep_var], axis = 1)
    if v == 1: 
        print('\nSplitting data set into training and test sets....')
        print('\t * {}% data in training set\n\t * {}% data in test set'.format(100*(1-test_size),100*test_size))

    return train_test_split(X, y, test_size = test_size, random_state=0)

print("Successfully imported the preprocessing module")