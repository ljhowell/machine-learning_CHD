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


def chose_features(dataset, features=['TenYearCHD','sysBP', 'glucose', 'age', 'totChol', 'cigsPerDay', 'diaBP', 'prevalentHyp',
            'diabetes', 'BPMeds','male', 'BMI', 'prevalentStroke',
            'education', 'heartRate', 'currentSmoker'], n_features = -1, v=0, vv =0):
    '''Return reduced dataset with only chosen columns
    - dataset: pandas dataframe of dataset to have columns chosen
    - features (optional, default = all features): list of strings matching features to keep
    - n_features (optional) - if specified, the top n features from the scaled list is chosen: 
    ['TenYearCHD','sysBP', 'glucose','age', 'totChol', 'cigsPerDay', 'diaBP', 'prevalentHyp',
            'diabetes', 'BPMeds','male', 'BMI', 'prevalentStroke',
            'education', 'heartRate', 'currentSmoker'],
    - v (optional) - Verbose (default 0) int 0 or 1. Print no. of features kept and lost 
    - vv (optional) - Very verbose (default 0) int 0 or 1. Print list of chosen and rejected features
    '''
    
    
    if n_features != -1:
        if n_features > len(dataset.columns):
            print('WARNING: chose_features has an error: n_features must be less than the number of columns')
            return(-1)
        else:
            ordered_f = ['TenYearCHD','sysBP', 'glucose', 'age', 'totChol', 'cigsPerDay', 'diaBP', 'prevalentHyp',
            'diabetes', 'BPMeds','male', 'BMI', 'prevalentStroke',
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

    return dataset2.reset_index(drop=True)

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
    
    return dataset2.reset_index(drop=True)


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
        cols = data.columns.drop('TenYearCHD',errors='ignore')
        scaled_cols = DataFrame(scaler_minmax.fit_transform(data[cols].copy()),columns=cols) #drop TenyearCHD as we don't want to scale this
        return(scaled_cols.join(data['TenYearCHD'])) #may break if data doesn't contain TenYearCHD column

    elif method == 'standard':
        scaler_std = preprocessing.StandardScaler() #with_std=False
        cols = data.columns.drop('TenYearCHD',errors='ignore')
        scaled_cols = DataFrame(scaler_std.fit_transform(data[cols].copy()),columns=cols)
        return(scaled_cols.join(data['TenYearCHD']))

    else:
        print('\nscale_data encountered a failure!! Check parameters\n')
        return(-1)

def split_data(dataset,dep_var='TenYearCHD', test_size = 0.2, v = 0, r_state = 0):
    '''Split the dataset, return X_train, X_test, y_train, y_test as Pandas Dataframes
    - dataset: Pandas Dataframe. Data to split into training and test data
    - dep_var (optional, default = 'TenYearCHD'): string. Name of column to be dependant variable
    - test_size (optional, default = 0.2): float (0.0-1.0). Proportion of total data to make up test set.
    - v (optional -default = 0): Verbose
    - r_state (optional): Sets the random state for the split
    Returns 4 datasets in order: X_train, X_test, y_train, y_test
    '''
    from sklearn.model_selection import train_test_split
    y = dataset[dep_var]
    X = dataset.drop([dep_var], axis = 1)
    if v == 1: 
        print('\nSplitting data set into training and test sets....')
        print('\t * {}% data in training set\n\t * {}% data in test set'.format(100*(1-test_size),100*test_size))

    return train_test_split(X, y, test_size = test_size, random_state=r_state)

def upsample(dataset,r_state=0,ratio_1_to_0=1.0,v=0):
    '''Resample dataset by upsampling, increasing number of minority samples by imputation
    - dataset: Pandas Dataframe. Data to upsample
    - r_state (optional): int. Random state to use
    - ratio_1_to_0 (optional): float. Ratio to resample to.
    - v (optional): Verbose
    Returns resampled dataset
    '''
    from sklearn.utils import resample
    from pandas import concat
    
    # Separate majority and minority classes
    df_majority = dataset[dataset['TenYearCHD']==0]
    df_minority = dataset[dataset['TenYearCHD']==1]
    
    if int(len(df_majority)*ratio_1_to_0) == 0:
        print('[ERROR] upsample ratio_1_to_0 too low')
        return dataset

    # Upsample minority class
    df_minority_upsampled = resample(df_minority, 
                                     replace=True,     # sample with replacement
                                     n_samples=int(len(df_majority)*ratio_1_to_0),    # to match majority class
                                     random_state=0) # reproducible results

    # Combine majority class with upsampled minority class
    df_upsampled = concat([df_minority_upsampled,df_majority])
    
    
    # Display class counts 
    if v==1: 
        print("\nUpsampling data....")
        print('Number values before resample:\n',dataset['TenYearCHD'].value_counts().sort_index())
        print('Ratio before 1:0 = {}'.format(dataset['TenYearCHD'].value_counts()[1]/dataset['TenYearCHD'].value_counts()[0]))
        print('Number values after resample:\n',df_upsampled['TenYearCHD'].value_counts().sort_index())
        print('Ratio after 1:0 = {}'.format(df_upsampled['TenYearCHD'].value_counts()[1]/df_upsampled['TenYearCHD'].value_counts()[0]))
    return df_upsampled.reset_index(drop=True)
    
def downsample(dataset,r_state=0,v=0,ratio_1_to_0=1):
    '''Resample dataset by downsampling, decrease number of majority samples by dropping 
    - dataset: Pandas Dataframe. Data to downsample
    - r_state (optional): int. Random state to use.
    - ratio_1_to_0 (optional): float. Ratio to resample to.
    - v (optional): Verbose
    Returns downsampled dataset
    '''
    from sklearn.utils import resample
    from pandas import concat
    
    # Separate majority and minority classes
    df_majority = dataset[dataset['TenYearCHD']==0]
    df_minority = dataset[dataset['TenYearCHD']==1]
    
    if ratio_1_to_0 < dataset['TenYearCHD'].value_counts()[1]/dataset['TenYearCHD'].value_counts()[0]:
        print('[ERROR] downsample invalid ratio_1_to_0, cannot downsample below intial ratio of',
              dataset['TenYearCHD'].value_counts()[1]/dataset['TenYearCHD'].value_counts()[0])
        return dataset
    elif int(len(df_minority)/ratio_1_to_0) == 0:
        print('[ERROR] downsample ratio_1_to_0 too high')
        return dataset

    # Downsample majority class
    df_majority_downsampled = resample(df_majority, 
                                     replace=False,    # sample without replacement
                                     n_samples=int(len(df_minority)/ratio_1_to_0),     # to match minority class
                                     random_state=r_state) # reproducible results

    # Combine minority class with downsampled majority class
    df_downsampled = concat([df_majority_downsampled, df_minority])

    # Display new class counts
    if v==1: 
        print("\nDownsampling data....")
        print('Number values before resample:\n',dataset['TenYearCHD'].value_counts().sort_index())
        print('Ratio before 1:0 = {}'.format(dataset['TenYearCHD'].value_counts()[1]/dataset['TenYearCHD'].value_counts()[0]))
        print('Number values after resample:\n',df_downsampled['TenYearCHD'].value_counts().sort_index())
        print('Ratio after 1:0 = {}'.format(df_downsampled['TenYearCHD'].value_counts()[1]/df_downsampled['TenYearCHD'].value_counts()[0]))
        
    return df_downsampled.reset_index(drop=True)

print("Successfully imported the preprocessing module")