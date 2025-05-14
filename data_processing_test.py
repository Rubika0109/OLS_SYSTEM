import pandas as pd
from data_ingest import IngestData
from data_processing import (
    drop_and_fill,
    find_columns_with_few_values,
    find_constant_columns,
)
from feature_engineering import bin_to_num, cat_to_col, one_hot_encoding

ingest_data = IngestData()  # Instantiate the class
df = pd.read_csv(r"C:\ols-regression-challenge\nrippner-ols-regression-challenge\data\cancer_reg.csv")  # Adjust based on actual structure

constant_columns = find_constant_columns(df)
print("columns that contain a single value:" , constant_columns)
columns_with_few_values = find_columns_with_few_values(df, 10)

df["binnedinc"][0]
df = bin_to_num(df)

df = cat_to_col(df)
df = one_hot_encoding(df)
df = drop_and_fill(df)
print(df.shape)
df.to_csv(r"C:\ols-regression-challenge\prprocessed_data.csv", index=False)
