from sklearn.model_selection import train_test_split as tts

def find_constant_columns(dataframe):
    constant_columns = []
    for column in dataframe.columns:
        unique_values = dataframe[column].unique()
        if len(unique_values) == 1:
            constant_columns.append(column)
    return constant_columns      

def delete_constant_columns(dataframe, columns_to_delete):
    dataframe = dataframe.drop(columns_to_delete,axis=1)
    return dataframe  

def find_columns_with_few_values(dataframe, threshold):
    few_values_columns = []
    for column in dataframe.columns:
        unique_values_count = len(dataframe[column].unique())
        if unique_values_count < threshold:
            few_values_columns.append(column)
    return few_values_columns

def find_duplicate_rows(dataframe):
    duplicate_rows = dataframe[dataframe.duplicate()]
    return duplicate_rows

def delete_duplicate_rows(dataframe):
    dataframe = dataframe.drop_duplicate(keep="first")
    return dataframe

def drop_and_fill(dataframe):
    cols_to_drop = dataframe.columns[dataframe.isnull().mean() > 0.5]
    dataframe = dataframe.drop(cols_to_drop, axis=1)
    dataframe = dataframe.fillna(dataframe.mean())
    return dataframe

from sklearn.model_selection import train_test_split

def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets.

    Parameters:
    df (DataFrame): The input dataset.
    target_column (str): The name of the target variable.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): Seed for reproducibility.

    Returns:
    X_train, X_test, y_train, y_test: Splitted training and testing sets.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test

