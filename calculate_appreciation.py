import pandas as pd
from datetime import datetime
import ast

def is_valid_price_vs_tax(row):
    try:
        tax_data = ast.literal_eval(row['tax_history'])
        sold_year = row['last_sold_date'].year
        assessment = next((entry['assessment']['total'] for entry in tax_data if entry['year'] == sold_year), None)
        
        if assessment is None or assessment == 0:
            return False
        
        sold_price = row['sold_price']
        lower_bound = assessment * 0.7
        upper_bound = assessment * 1.3

        return lower_bound <= sold_price <= upper_bound

    except Exception:
        return False

INPUT_CSV = "HomeHarvest_20250719_000354.csv" 
OUTPUT_CSV = "processed_data.csv"

df = pd.read_csv(INPUT_CSV, parse_dates=['last_sold_date'], low_memory=False)

df['last_sold_date'] = pd.to_datetime(df['last_sold_date'], errors='coerce')

df = df[(df['sold_price'] > 0) & (df['estimated_value'] > 0)]
#df = df[df.apply(is_valid_price_vs_tax, axis=1)]


today = pd.Timestamp(datetime.today().date())
df['months_since_sale'] = (today - df['last_sold_date']).dt.days / 30.44  # avg days in month

df = df[df['months_since_sale'] > 0]

df['baths'] = df['full_baths'] + df['half_baths'] * 0.5

df['monthly_appreciation'] = (df['estimated_value'] - df['sold_price']) / df['months_since_sale']
df['yearly_appreciation'] = df['monthly_appreciation'] * 12

df['monthly_appreciation'] = df['monthly_appreciation'].round(0).astype(int)
df['yearly_appreciation'] = df['yearly_appreciation'].round(0).astype(int)

df['monthly_appreciation_rate'] = (df['monthly_appreciation'] / df['sold_price']) * 100
df['yearly_appreciation_rate'] = (df['yearly_appreciation'] / df['sold_price']) * 100

output_df = df[[
    'property_url',
    'full_street_line',
    'unit',
    'estimated_value',
    'assessed_value',
    'sold_price',
    'last_sold_date',
    'months_since_sale',
    'sqft',
    'beds',
    'baths',
    'neighborhoods',
    'list_date',
    'tax',
    'lot_sqft',
    'new_construction',
    'stories',
    'monthly_appreciation',
    'monthly_appreciation_rate'
]]

output_df = output_df.sort_values(by='monthly_appreciation_rate', ascending=False)
output_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved to {OUTPUT_CSV}")
