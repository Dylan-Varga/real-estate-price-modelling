import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime

CSV_PATH = "HomeHarvest_20250612_175425.csv"

df = pd.read_csv(CSV_PATH, parse_dates=['last_sold_date'], low_memory=False)

df = df.dropna(subset=['sold_price', 'last_sold_date', 'neighborhoods'])

df = df[df['sold_price'] > 0]

df['year'] = df['last_sold_date'].dt.year

df['neighborhood'] = df['neighborhoods'].astype(str).str.split(',').str[0].str.strip()

grouped = df.groupby(['neighborhood', 'year'])['sold_price'].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=grouped, x='year', y='sold_price', hue='neighborhood', marker='o')

plt.title('Average Home Sale Price by Neighborhood (Since 2021)')
plt.xlabel('Year')
plt.xticks([2021, 2022, 2023, 2024, 2025])
plt.ylabel('Average Sold Price ($)')

ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.grid(True)
plt.legend(title='Neighborhood', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()