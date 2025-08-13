import numpy as np
import pandas as pd
import ast

import matplotlib.pyplot as plt
import seaborn as sns

import sys

''' ------------Cleaning and training hyperparameters------------ '''

# Outlier handling #
EST_VALUE_IQR_MULT = 3
ASSESSED_VALUE_IQR_MULT = 3
SOLD_PRICE_IQR_MULT = 2
BEDS_RANGE = (0, 6)
BATHS_RANGE = (0, 5)
STORIES_RANGE = (0, 3)
SQFT_IQR_MULT = 1.5
LOT_SQFT_IRQ_MULT = 1.5
TAX_IQR_MULT = 1.5
MONTHLY_APPRECIATION_IRQ_MULT = 10
MONTHLY_APPRECIATION_RATE_IQR_MULT = 15



''' ------------Program configuration------------ '''

# Process CL arguments #
# Find and set any flags
VISUALIZE_DATA = False
GENERATE_DUMMIES = False
VERBOSE = False
for arg in range(1, len(sys.argv)):
    if sys.argv[arg] == '--dummy':
        GENERATE_DUMMIES = True
    if sys.argv[arg] == '--visualize':
        VISUALIZE_DATA = True
    if sys.argv[arg] == '--verbose':
        VERBOSE = True

# If 1 csv file, it's source. If 2, first is source, second is output. Use defaults for unset filenames
SOURCE_DATA = 'processed_data.csv'
DATACLEANING_OUTPUT_CSV = 'cleaned_processed_data.csv'
cla_csv_files = []
for arg in sys.argv[1:]:
    if arg.lower().endswith('.csv'):
        cla_csv_files.append(arg)

if cla_csv_files:
    SOURCE_DATA = cla_csv_files[0]
    if len(cla_csv_files) == 1:
        DATACLEANING_OUTPUT_CSV = 'cleaned_processed_data.csv'
    else:
        DATACLEANING_OUTPUT_CSV = cla_csv_files[1]

# Load dataset
df = pd.DataFrame(pd.read_csv(SOURCE_DATA))


""" ------------Outlier handling------------ """

# View outliers #
if VISUALIZE_DATA:
    fig, axs = plt.subplots(3, 4, figsize = (20, 10))
    estimated_value_plt = sns.boxplot(x=df['estimated_value'], ax=axs[0, 0])
    assessed_value_plt = sns.boxplot(x=df['assessed_value'], ax=axs[0, 1])
    sold_price_plt = sns.boxplot(x=df['sold_price'], ax=axs[0, 2])
    sqft_plt = sns.boxplot(x=df['sqft'], ax=axs[1, 0])
    beds_plt = sns.boxplot(x=df['beds'], ax=axs[1, 1])
    cooler_baths_plt = sns.violinplot(x=df['baths'].dropna(), ax=axs[0,3])
    tax_plt = sns.boxplot(x=df['tax'], ax=axs[2, 0])
    lot_sqft_plt = sns.boxplot(x=df['lot_sqft'], ax=axs[2, 1])
    stories_plt = sns.boxplot(x=df['stories'], ax=axs[2, 2])
    monthly_appreciation_plt = sns.boxplot(x=df['monthly_appreciation'], ax=axs[2, 3])
    monthly_appreciation_rate_plt = sns.boxplot(x=df['monthly_appreciation_rate'], ax=axs[1, 3])

    plt.tight_layout()
    plt.show()

# Treat outliers #
if VERBOSE:
    print("\nTreating outliers...")
# Function to treat outliers using IQR multiplier #
def clean_outliers_using_multiplier(df, column_name, iqr_multiplier):
    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr

    initial_vals = df[column_name].notna().sum() # Count pre-cleaning values for comparison output

    # Remove estimated values less than q1 - iqr * multiplier or greater than q3 + iqr * multiplier 
    def clean_value(x):
        if x < 0:
            return np.nan
        if x < lower_bound or x > upper_bound:
            return np.nan
        return x

    cleaned_series = df[column_name].apply(clean_value)

    # Count and report exclusions
    post_clean_vals = cleaned_series.notna().sum()
    ignored_outliers = initial_vals - post_clean_vals
    if VERBOSE:
        print(f"{column_name}: outliers ignored: {ignored_outliers} ({(ignored_outliers/initial_vals)*100:.2f}%)")


    return cleaned_series

# Clean outliers in columns using IQR multiplier #
df['estimated_value'] = clean_outliers_using_multiplier(df, 'estimated_value', EST_VALUE_IQR_MULT)
df['assessed_value'] = clean_outliers_using_multiplier(df, 'assessed_value', ASSESSED_VALUE_IQR_MULT)
df['sold_price'] = clean_outliers_using_multiplier(df, 'sold_price', SOLD_PRICE_IQR_MULT)
df['sqft'] = clean_outliers_using_multiplier(df, 'sqft', SQFT_IQR_MULT)
df['lot_sqft'] = clean_outliers_using_multiplier(df, 'lot_sqft', LOT_SQFT_IRQ_MULT)
df['tax'] = clean_outliers_using_multiplier(df, 'tax', TAX_IQR_MULT)
df['monthly_appreciation'] = clean_outliers_using_multiplier(df, 'monthly_appreciation', MONTHLY_APPRECIATION_IRQ_MULT)
df['monthly_appreciation_rate'] = clean_outliers_using_multiplier(df, 'monthly_appreciation_rate', MONTHLY_APPRECIATION_RATE_IQR_MULT)

# Function to clean outliers in beds/baths/stories using range #
def clean_outliers_using_range(df, column_name, value_range):
    min_val, max_val = value_range

    initial_vals = df[column_name].notna().sum()  # Count pre-cleaning values for comparison output

     # Remove values less than min or greater than max
    def clean_value(x):
        if x < min_val or x > max_val:
            return np.nan
        return x
    
    cleaned_series = df[column_name].apply(clean_value)

    # Count and report exclusions
    post_clean_vals = cleaned_series.notna().sum()
    ignored_outliers = initial_vals - post_clean_vals
    if VERBOSE:
        print(f"{column_name}: outliers ignored: {ignored_outliers} ({(ignored_outliers/initial_vals)*100:.2f}%)")

    return cleaned_series

# Clean outliers in beds/baths/stories using range #
df['beds'] = clean_outliers_using_range(df, 'beds', BEDS_RANGE)
df['baths'] = clean_outliers_using_range(df, 'baths', BATHS_RANGE)
df['stories'] = clean_outliers_using_range(df, 'stories', STORIES_RANGE)

# View outliers after outlier treatment #
if VISUALIZE_DATA & VERBOSE:
        fig, axs = plt.subplots(3, 4, figsize = (20, 10))
        estimated_value_plt = sns.boxplot(x=df['estimated_value'], ax=axs[0, 0])
        assessed_value_plt = sns.boxplot(x=df['assessed_value'], ax=axs[0, 1])
        sold_price_plt = sns.boxplot(x=df['sold_price'], ax=axs[0, 2])
        sqft_plt = sns.boxplot(x=df['sqft'], ax=axs[1, 0])
        beds_plt = sns.boxplot(x=df['beds'], ax=axs[1, 1])
        cooler_baths_plt = sns.violinplot(x=df['baths'].dropna(), ax=axs[0,3])
        tax_plt = sns.boxplot(x=df['tax'], ax=axs[2, 0])
        lot_sqft_plt = sns.boxplot(x=df['lot_sqft'], ax=axs[2, 1])
        stories_plt = sns.boxplot(x=df['stories'], ax=axs[2, 2])
        monthly_appreciation_plt = sns.boxplot(x=df['monthly_appreciation'], ax=axs[2, 3])
        monthly_appreciation_rate_plt = sns.boxplot(x=df['monthly_appreciation_rate'], ax=axs[1, 3])

        plt.tight_layout()
        plt.suptitle('Data after outlier treatment', fontsize=16)
        plt.show()


""" ------------Dummy data generation------------ """

# Clean/fill data where needed #
# Note: Means used over medians under assumption that outlier removal has already been run. TODO: Consider adding flag/user control
if GENERATE_DUMMIES:
    if VERBOSE:
        print("\nGenerating dummy data...")

    # Used to track number of data filled in for each column
    pre_clean_null_count = 0

    # Set missing assessed values to mirror average assessed value:estimated value ratio #
    if VERBOSE:
        pre_clean_null_count = df['assessed_value'].isnull().sum()
    avg_assessed_to_estimated_ratio = df['assessed_value'].mean() / df['estimated_value'].mean()
    df['assessed_value'] = df.apply(
        lambda row: row['estimated_value'] * avg_assessed_to_estimated_ratio if pd.isnull(row['assessed_value']) else row['assessed_value'],
        axis=1
    )
    if VERBOSE:
        post_clean_null_count = df['assessed_value'].isnull().sum()
        print(f"assessed_value: {pre_clean_null_count - post_clean_null_count} values filled in ({(pre_clean_null_count - post_clean_null_count)/df.shape[0]*100:.2f}%)")

    # Set missing square footage values to mirror average square footage:(beds + baths) ratio #
    if VERBOSE:
        pre_clean_null_count = df['sqft'].isnull().sum()
    avg_sqft_to_beds_baths_ratio = df['sqft'].mean() / (df['beds'] + df['baths']).mean()

    # (If beds or baths are missing, use simple global average)
    avg_beds = df['beds'].mean()
    avg_baths = df['baths'].mean()

    df['sqft'] = df.apply(
        lambda row: (
                ((row['beds'] if pd.notnull(row['beds']) else avg_beds) + (row['baths'] if pd.notnull(row['baths']) else avg_baths)) * avg_sqft_to_beds_baths_ratio
                if pd.isnull(row['sqft']) else row['sqft']
        ),
        axis=1
    )
    if VERBOSE:
        post_clean_null_count = df['sqft'].isnull().sum()
        print(f"sqft: {pre_clean_null_count - post_clean_null_count} values filled in ({(pre_clean_null_count - post_clean_null_count)/df.shape[0]*100:.2f}%)")

    # Set missing beds to mirror average beds:square footage ratio #
    avg_beds_to_sqft_ratio = df['beds'].mean() / df['sqft'].mean()
    if VERBOSE:
        pre_clean_null_count = df['beds'].isnull().sum()
    df['beds'] = df.apply(
        lambda row: row['sqft'] * avg_beds_to_sqft_ratio if pd.isnull(row['beds']) else row['beds'],
        axis=1
    )
    if VERBOSE:
        post_clean_null_count = df['beds'].isnull().sum()
        print(f"beds: {pre_clean_null_count - post_clean_null_count} values filled in ({(pre_clean_null_count - post_clean_null_count)/df.shape[0]*100:.2f}%)")

    # Set missing baths to mirror average baths:square footage ratio #
    avg_baths_to_sqft_ratio = df['baths'].mean() / df['sqft'].mean()
    if VERBOSE:
        pre_clean_null_count = df['baths'].isnull().sum()
    df['baths'] = df.apply(
        lambda row: row['sqft'] * avg_baths_to_sqft_ratio if pd.isnull(row['baths']) else row['baths'],
        axis=1
    )
    if VERBOSE:
        post_clean_null_count = df['baths'].isnull().sum()
        print(f"baths: {pre_clean_null_count - post_clean_null_count} values filled in ({(pre_clean_null_count - post_clean_null_count)/df.shape[0]*100:.2f}%)")

    # Set missing neighborhood values to 'Unknown' #
    if VERBOSE:
        pre_clean_null_count = df['neighborhoods'].isnull().sum()
    df['neighborhoods'] = df['neighborhoods'].fillna('Unknown')
    if VERBOSE:
        post_clean_null_count = df['neighborhoods'].isnull().sum()
        print(f"neighborhoods: {pre_clean_null_count - post_clean_null_count} values filled in (as Unknown) ({(pre_clean_null_count - post_clean_null_count)/df.shape[0]*100:.2f}%)")

    # Set missing list dates to mirror average list date - sold date difference #
    avg_list_sold_diff = (pd.to_datetime(df['last_sold_date']) - pd.to_datetime(df['list_date'])).mean()
    if VERBOSE:
        pre_clean_null_count = df['list_date'].isnull().sum()
    df['list_date'] = df.apply(
        lambda row: pd.to_datetime(row['last_sold_date']) - avg_list_sold_diff if pd.isnull(row['list_date']) else pd.to_datetime(row['list_date']),
        axis=1
    )
    if VERBOSE:
        post_clean_null_count = df['list_date'].isnull().sum()
        print(f"list_date: {pre_clean_null_count - post_clean_null_count} values filled in ({(pre_clean_null_count - post_clean_null_count)/df.shape[0]*100:.2f}%)")

    # Set missing tax to mirror average tax:estimated value ratio, since some assessed values are missing #
    avg_tax_to_estimated_ratio = df['tax'].mean() / df['estimated_value'].mean()
    if VERBOSE:
        pre_clean_null_count = df['tax'].isnull().sum()
    df['tax'] = df.apply(
        lambda row: row['estimated_value'] * avg_tax_to_estimated_ratio if pd.isnull(row['tax']) else row['tax'],
        axis=1
    )
    if VERBOSE:
        post_clean_null_count = df['tax'].isnull().sum()
        print(f"tax: {pre_clean_null_count - post_clean_null_count} values filled in ({(pre_clean_null_count - post_clean_null_count)/df.shape[0]*100:.2f}%)")

    # Set missing lot_sqft (note - high number of missing values) to mirror average lot_sqft:sqft ratio #
    avg_lot_sqft_to_sqft_ratio = df['lot_sqft'].mean() / df['sqft'].mean()
    if VERBOSE:
        pre_clean_null_count = df['lot_sqft'].isnull().sum()
    df['lot_sqft'] = df.apply(
        lambda row: row['sqft'] * avg_lot_sqft_to_sqft_ratio if pd.isnull(row['lot_sqft']) else row['lot_sqft'],
        axis=1
    )
    if VERBOSE:
        post_clean_null_count = df['lot_sqft'].isnull().sum()
        print(f"lot_sqft: {pre_clean_null_count - post_clean_null_count} values filled in ({(pre_clean_null_count - post_clean_null_count)/df.shape[0]*100:.2f}%)")

    # Set missing new_construction values to False (seemingly never needed) #
    if VERBOSE:
        pre_clean_null_count = df['new_construction'].isnull().sum()
    df['new_construction'] = df['new_construction'].fillna(False)
    if VERBOSE:
        post_clean_null_count = df['new_construction'].isnull().sum()
        print(f"new_construction: {pre_clean_null_count - post_clean_null_count} values filled in (as False) ({(pre_clean_null_count - post_clean_null_count)/df.shape[0]*100:.2f}%)")

    # Set missing stories values to average (also high number of missing values) #
    if VERBOSE:
        pre_clean_null_count = df['stories'].isnull().sum()
    avg_stories = df['stories'].mean()
    df['stories'] = df['stories'].fillna(avg_stories)
    if VERBOSE:
        post_clean_null_count = df['stories'].isnull().sum()
        print(f"stories: {pre_clean_null_count - post_clean_null_count} values filled in (as {avg_stories}) ({(pre_clean_null_count - post_clean_null_count)/df.shape[0]*100:.2f}%)")

    # Confirm all data is cleaned
    if VERBOSE:
        print("\nData after dummy data generation:")
        print(df.info())
        print(df.isnull().sum()*100/df.shape[0])

    # View outliers after dummy data generation #
    if VISUALIZE_DATA & VERBOSE:
        fig, axs = plt.subplots(3, 4, figsize = (20, 10))
        estimated_value_plt = sns.boxplot(x=df['estimated_value'], ax=axs[0, 0])
        assessed_value_plt = sns.boxplot(x=df['assessed_value'], ax=axs[0, 1])
        sold_price_plt = sns.boxplot(x=df['sold_price'], ax=axs[0, 2])
        sqft_plt = sns.boxplot(x=df['sqft'], ax=axs[1, 0])
        beds_plt = sns.boxplot(x=df['beds'], ax=axs[1, 1])
        cooler_baths_plt = sns.violinplot(x=df['baths'].dropna(), ax=axs[0,3])
        tax_plt = sns.boxplot(x=df['tax'], ax=axs[2, 0])
        lot_sqft_plt = sns.boxplot(x=df['lot_sqft'], ax=axs[2, 1])
        stories_plt = sns.boxplot(x=df['stories'], ax=axs[2, 2])
        monthly_appreciation_plt = sns.boxplot(x=df['monthly_appreciation'], ax=axs[2, 3])
        monthly_appreciation_rate_plt = sns.boxplot(x=df['monthly_appreciation_rate'], ax=axs[1, 3])

        plt.tight_layout()
        plt.suptitle('Data after cleaning', fontsize=16)
        plt.show()
    

    # Save cleaned/generated data
    df = df.round(1)
    df.to_csv(DATACLEANING_OUTPUT_CSV, index=False)
    if VERBOSE:
        print(f"Data cleaned and saved to {DATACLEANING_OUTPUT_CSV}")


""" ------------Identify variable correlations------------ """

# Pairplot of all variables #
if VISUALIZE_DATA:
    heatmap_columns = [
        'estimated_value',
        'assessed_value',
        'sold_price',
        'sqft',
        'lot_sqft',
        'tax',
        'monthly_appreciation',
        'monthly_appreciation_rate',
        'beds',
        'baths',
        'stories'
    ]

    # Compute correlation matrix
    corr = df[heatmap_columns].corr()

    # Set up the matplotlib figure
    plt.figure(figsize=(12, 10))
    sns.set(font_scale=1.0)

    # Draw the heatmap
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)

    plt.title("Pairwise Correlation Heatmap")
    plt.tight_layout()
    plt.show()
