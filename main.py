import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import gzip

file_path = 'zameen-property-data.csv'

df = pd.read_csv(file_path)

df.drop_duplicates(inplace=True)

df.dropna(subset=['price', 'location', 'city', 'province_name', 'latitude', 'longitude', 'area', 'purpose', 'bedrooms'], inplace=True)
df.fillna({'baths': df['baths'].median(), 'agency': 'Unknown', 'agent': 'Unknown'}, inplace=True)

df['price'] = df['price'].astype(np.float32)
df['latitude'] = df['latitude'].astype(np.float32)
df['longitude'] = df['longitude'].astype(np.float32)
df['baths'] = df['baths'].astype(np.int8)
df['bedrooms'] = df['bedrooms'].astype(np.int8)
df['date_added'] = pd.to_datetime(df['date_added'])

df['area'] = df['area'].str.extract('(\d+\.?\d*)').astype(np.float32)

unnecessary_columns = ['province_name','purpose', 'agency', 'agent','page_url','location', 'city']  # Updated list of unnecessary columns
df.drop(columns=unnecessary_columns, inplace=True)

df = pd.get_dummies(df, columns=['property_type'])

scaler = StandardScaler()
df[['price', 'latitude', 'longitude', 'area', 'baths', 'bedrooms']] = scaler.fit_transform(df[['price', 'latitude', 'longitude', 'area', 'baths', 'bedrooms']])

output_path_csv = 'transformed_property_data.csv.gz'
df.to_csv(output_path_csv, index=False, compression='gzip')

print(f"Transformed data saved to {output_path_csv}")

plt.figure(figsize=(10, 6))
plt.scatter(df['area'], df['price'], alpha=0.5)
plt.title('Price vs. Area')
plt.xlabel('Area')
plt.ylabel('Price')
plt.grid(True)
plt.show()
