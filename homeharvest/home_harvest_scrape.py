from homeharvest import scrape_property
from datetime import datetime

""" 
  Credit to ZacharyHampton - HomeHarvest for scraping code
  https://github.com/ZacharyHampton/HomeHarvest
"""

# Generate filename based on current timestamp
current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"HomeHarvest_{current_timestamp}.csv"

properties = scrape_property(
  location="Salem, OR",
  listing_type="sold",  # or (for_sale, for_rent, pending)
  date_from="2020-01-01",  # sold in last 30 days - listed in last 30 days if (for_sale, for_rent)
  date_to="2025-01-01"
  # property_type=['single_family','multi_family'],
  # date_from="2023-05-01", # alternative to past_days
  # date_to="2023-05-28",
  # foreclosure=True
  # mls_only=True,  # only fetch MLS listings
)
print(f"Number of properties: {len(properties)}")

# Export to csv
properties.to_csv(filename, index=False)
print(properties.head())