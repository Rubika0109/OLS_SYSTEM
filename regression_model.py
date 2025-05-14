import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

from data_processing import split_data


def correlation_among_numeric_features(df, cols):
    numeric_col  = df[cols]
    corr = numeric_col.corr()

    corr_features = set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) >0.8:
                colname = corr.columns[i]
                corr_features.add(colname)
    return corr_features


def lr_model(x_train, y_train):
    x_train_with_intercept = sm.add_constant(x_train)
    lr = sm.OLS(y_train, x_train_with_intercept).fit()
    return lr

def identify_significant_vars(lr, p_value_threshold=0.05):
    print(lr.pvalues)
    print(lr.rsquared)
    print(lr.rsquared_adj)
    significant_vars = [var for var in lr.pvalues.keys() if lr.pvalues[var] < p_value_threshold]
    return significant_vars

if __name__ == "__main__":
    capped_data = pd.read_csv(r"C:\project\ols-regression-challenge\capped_data.csv")
    print(capped_data.shape)
    # Pass the list of numeric columns explicitly
    numeric_columns = capped_data.select_dtypes(include=['number']).columns.tolist()
    corr_features = correlation_among_numeric_features(capped_data, numeric_columns)
    print(corr_features)

    highly_corr_cols = [
        "povertypercent",
        "median",
        "pctprivatecoveragealone",
        "medianagefemale",
        "pctempprivcoverage",
        "pctblack",
        "popest2015",
        "pctmarriedhouseholds",
        "upper_bound",
        "lower_bound",
        "pctprivatecoverage",
        "medianagemale",       
        "state_District of Columbia",
        "pctpubliccoveragealone",
    ]    

    cols = [col for col in capped_data.columns if col not in highly_corr_cols]
    len(cols)
    X_train, X_test, Y_train, Y_test = split_data(capped_data[cols],"target_deathrate") 
    lr =lr_model(X_train, Y_train)
    summary = lr.summary()
    print(summary)  

    significant_vars = identify_significant_vars(lr)
    print(len(significant_vars))

