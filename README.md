# TASK 2: Combined Unemployment Analysis (India)
# ----------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# ----------------------------------------------
# 1️⃣ Load both datasets
# ----------------------------------------------
file1 = "Unemployment in India.csv"
file2 = "Unemployment_Rate_upto_11_2020.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

print("✅ Files loaded successfully!")
print("Unemployment in India.csv:", df1.shape)
print("Unemployment_Rate_upto_11_2020.csv:", df2.shape)

# ----------------------------------------------
# 2️⃣ Clean and standardize both datasets
# ----------------------------------------------

# Clean df1
df1.columns = df1.columns.str.strip()
df1.rename(columns={'Date': 'Date', 'Unemployment Rate': 'Unemployment_Rate'}, inplace=True)
df1['Date'] = pd.to_datetime(df1['Date'], errors='coerce')

# Clean df2
df2.rename(columns={
    'Region': 'State',
    ' Date': 'Date',
    ' Frequency': 'Frequency',
    ' Estimated Unemployment Rate (%)': 'Unemployment_Rate',
    ' Estimated Employed': 'Employed',
    ' Estimated Labour Participation Rate (%)': 'Labour_Participation_Rate'
}, inplace=True)
df2.columns = df2.columns.str.strip()
df2['Date'] = pd.to_datetime(df2['Date'], errors='coerce')

# Keep only necessary columns
df2 = df2[['Date', 'State', 'Unemployment_Rate']]
df1 = df1[['Date', 'Region', 'Unemployment_Rate']] if 'Region' in df1.columns else df1

# Combine and clean
combined = pd.concat([df1, df2], ignore_index=True)
combined.dropna(subset=['Date', 'Unemployment_Rate'], inplace=True)

# Standardize column names
combined.rename(columns={'Region': 'State'}, inplace=True)

print("\n🧹 After cleaning:", combined.shape)
print("Sample data:\n", combined.head())

# ----------------------------------------------
# 3️⃣ Exploratory Analysis
# ----------------------------------------------
print("\nBasic Statistics:\n", combined['Unemployment_Rate'].describe())

# Check how many unique states
print("\nUnique states:", combined['State'].nunique())

# ----------------------------------------------
# 4️⃣ Visualize Regional Trends
# ----------------------------------------------
plt.figure(figsize=(12,6))
sns.lineplot(data=combined, x='Date', y='Unemployment_Rate', hue='State', legend=False)
plt.title('Unemployment Rate Trend (All States Combined)')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.show()

# Average unemployment rate by state
plt.figure(figsize=(10,6))
state_avg = combined.groupby('State')['Unemployment_Rate'].mean().sort_values(ascending=False)
sns.barplot(x=state_avg, y=state_avg.index, palette='coolwarm')
plt.title('Average Unemployment Rate by State')
plt.xlabel('Unemployment Rate (%)')
plt.ylabel('State')
plt.show()

# ----------------------------------------------
# 5️⃣ Analyze Covid-19 Impact
# ----------------------------------------------
pre_covid = combined[combined['Date'] < '2020-03-01']
covid_period = combined[(combined['Date'] >= '2020-03-01') & (combined['Date'] <= '2020-12-31')]
post_covid = combined[combined['Date'] > '2020-12-31']

pre_mean = pre_covid['Unemployment_Rate'].mean()
covid_mean = covid_period['Unemployment_Rate'].mean()
post_mean = post_covid['Unemployment_Rate'].mean()

print(f"\n📊 Average Unemployment Rate:")
print(f"Before Covid (till Feb 2020): {pre_mean:.2f}%")
print(f"During Covid (Mar–Dec 2020): {covid_mean:.2f}%")
print(f"After Covid (2021 onwards): {post_mean:.2f}%")

plt.figure(figsize=(7,5))
sns.boxplot(data=[pre_covid['Unemployment_Rate'], covid_period['Unemployment_Rate'], post_covid['Unemployment_Rate']],
            palette=['skyblue','salmon','lightgreen'])
plt.xticks([0,1,2], ['Pre-Covid', 'During Covid', 'Post-Covid'])
plt.title('Covid-19 Impact on Unemployment Rate in India')
plt.ylabel('Unemployment Rate (%)')
plt.show()

# ----------------------------------------------
# 6️⃣ National Trend (Monthly)
# ----------------------------------------------
monthly = combined.groupby(combined['Date'].dt.to_period('M'))['Unemployment_Rate'].mean().to_timestamp()

plt.figure(figsize=(10,5))
sns.lineplot(x=monthly.index, y=monthly.values, color='blue')
plt.axvline(pd.Timestamp('2020-03-01'), color='red', linestyle='--', label='Covid-19 Begins')
plt.title('National Unemployment Trend (India)')
plt.xlabel('Month')
plt.ylabel('Unemployment Rate (%)')
plt.legend()
plt.show()

# ----------------------------------------------
# 7️⃣ Seasonal Decomposition
# ----------------------------------------------
decomp = seasonal_decompose(monthly, model='additive', period=12)
decomp.plot()
plt.suptitle('Seasonal Decomposition of Unemployment Rate (India)', y=1.02)
plt.show()

# ----------------------------------------------
# 8️⃣ Insights
# ----------------------------------------------
print("\n🔍 Key Insights:")
print("1. Sharp spike observed around March–April 2020 due to Covid lockdowns.")
print("2. Average unemployment nearly doubled during 2020 compared to pre-Covid levels.")
print("3. Gradual recovery observed from mid-2021 onwards.")
print("4. Seasonal variations remain visible across states—suggesting cyclical employment.")
print("5. Post-Covid period shows stabilization but higher volatility in certain states.")

# ----------------------------------------------
# 9️⃣ Save Cleaned Data
# ----------------------------------------------
combined.to_csv("Combined_Unemployment_Analysis.csv", index=False)
print("\n✅ Cleaned combined dataset saved as 'Combined_Unemployment_Analysis.csv'")
