# ==================== 1. Data Wrangling ====================== #

#============ Duplicates Dropping ============#
duplicates = df.duplicated()
print(duplicates.value_counts())
# The .drop_duplicates() function removes duplicate rows
df = df.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False) 

#============ Column Name Standardization ============#
df.columns = (df.columns.astype(str) # get all column names
              .str.strip() # strip leading/trailing spaces
              .str.lower() # convert to lower case
              .str.replace(r"\s+", "_", regex=True))
                # replace all whitespace with underscore
# map() applies the str.lower() function to each of the columns in our dataset to convert the column names to all lowercase
df.columns = map(str.lower, df.columns)

#============ Column Renaming ============#
# axis=1` refers to the columns, `axis=0` would refer to the rows
# In the dictionary the key refers to the original column name and the value refers to the new column name {'oldname1': 'newname1', 'oldname2': 'newname2'}
df = df.rename({'oldname1': 'newname1', 'oldname2': 'newname2'}, axis=1)

#============ Data Type Checking ============#
# Check each column’s data types by appending .dtypes to the dataframe
print(df.dtypes)
# .nunique() counts the number of unique values in each column 
df.nunique()

#============ String Parsing/Conversion ============#
# To modify strings and transform them into metrics/numbers
print(df.columns)
print(df.price.head())
df['price'].replace(to_replace='[\$,]', value='', inplace=True, regex=True)
df.price = pd.to_numeric(df.price)

#============ Number Extraction from String ============#
print(df.head())
split_df = df['column'].str.split('(\d+)', expand=True)
print(split_df)
df.column = pd.to_numeric(split_df[1])

#============ Data Missingness ============#
# Type 1 - Structurally Missing Data: missed for a logical reason (e.g., not applicable)
# One field is not applicable to a certain group; it is dependent on the value of another field
# Type 2 - Missing Completely at Random (MCAR): missed with no logic, no outside force, or unseen behavior
# One field is gone for no reason, probably due to a bug
# Type 3 - Missing at Random (MAR): missed differently for different groups, but equally within a group
# Some fields are missing (not reported) specifically for some groups due to certain concerns, but not for all values
# Type 4 - Missing Not at Random (MNAR): missed for some logical reason for all data
# Some fields are missing following a possible pattern

# Checking small chunks first with df.head()
# Checking the statistics with df.describe(); df.summary() 
# Checking the columns' data types with df.dtypes

# Count the number of missing values in each column 
df.isna().sum() 

# Replace abnormalities/unexpected with necessary NaNs
# .where() method keeps the values specified in its first argument, and replaces all other values with NaN
df['column_name'] = df['column_name'].where(cond_keep, other=nan, inplace=False, axis=None) 

# To understand the missingness in a column by counting the missing values across another feature's values
# crosstab() computes the frequency of two or more variables; to look at the missingness in a column, add isna() to the column to identify if there is an NaN in that column
pd.crosstab(
        df['anoter_feature'],  # tabulates the other feature (the one against) as the index
        df['feature'].isna(), # tabulates the number of missing values in the column as columns
        rownames = ['anoter_feature_name'], # names the rows
        colnames = ['feature_is_na']) # names the columns

# Method 1 - Listwise Deletion: drop all rows with any missing value (only if there's not much missing data)
df.dropna(inplace=True)
# Method 2 - Pairwise Deletion: drop all rows with a missing value within specified columns
df = df.dropna(subset=['na_in_column1', 'na_in_column2'])
# Method 3: fill the missing values with the mean of the column, or with some other aggregate value
df.fillna(value={'column1':df.column1.mean(), 'column2':df.column2.mean()}, inplace=True)
# Method 4 (only applicable for time series) - Single Imputation: fill a missing value with a value from another time
        # Forward Fill - LOCF
df['column_name1'].ffill(axis=0, inplace=True)
        # Backward Fill - NOCB 
df['column_name2'].bfill(axis=0, inplace=True)
        # Baseline Fill - BOCF
baseline = df['column_name3'].mean()  # or df['column_name3'][0], etc.
df['column_name3'].fillna(value=baseline, inplace=True)
        # Worst Fill - WOCF
worst = df['column_name4'].min()  # or df['column_name4'].max(), etc.
df['column_name4'].fillna(value=worst, inplace=True)

# ==================== 2. Data Tidying ====================== #

#============ Affixes Removal ============#
# sometimes there may be values like URLs which having a common section that's not "important"
# .str.lstrip('https://') removes the “https://” from the left side of the string
print(df.head())
df['url'] = df['url'].str.lstrip('https://') 

#============ Columns Combining ============#
# sometimes two columns may be combined under one new column where their original values can be
# combined under another new column, so that the new columns can be manipulated with pandas' methods
# Or sometimes we need each column to store one type of measurement
# pd.melt(frame=df, id_vars=['columns_preserved', ], value_vars=['columns_turn_variable_data', ],
# value_name='variable_column_name_new', var_name='data_column_name_new')
print(df.columns)
print(df.head())
df = pd.melt(frame=df, id_vars="Account", value_vars=["Checking","Savings"], value_name="Amount", var_name="Account Type")

#============ Columns Splitting ============#
# sometimes multiple measurements are recorded in the same column, and we want to separate these out
# so that we can do individual analysis on each variable
print(df.columns)
print(df.head())
df['month'] = df.birthday.str[0:2]
df['day'] = df.birthday.str[2:4]
df['year'] = df.birthday.str[4:]
df = df[['column1', 'column2', 'column3']]
print(df.head())
# OR
string_split = df['birthday'].str.split('-')
df['month'] = string_split.str.get(0)
df['day'] = string_split.str.get(1)
df['year'] = string_split.str.get(2)
df = df[['column1', 'column2', 'column3']]
print(df.head())
