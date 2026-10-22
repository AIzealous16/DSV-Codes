# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Kaggle dataset
df = pd.read_csv("netflix_customer_churn.csv")

# Display first 5 records
print("First 5 Records:")
print(df.head())

# Display dataset information
print("\nDataset Information:")
print(df.info())

# Display basic statistics
print("\nStatistical Summary:")
print(df.describe())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

 # Handle missing numerical values using mean
numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].mean())

# Handle missing categorical values using mode
categorical_columns = df.select_dtypes(include=["object", "str"]).columns

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

# Remove duplicate records
print("\nNumber of Duplicate Records:", df.duplicated().sum())

df = df.drop_duplicates()

# Check missing values after cleaning
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Display cleaned dataset
print("\nCleaned Dataset:")
print(df.head())


# -------------------------------
# VISUALIZATION 1: Churn Count
# -------------------------------

plt.figure(figsize=(6, 4))
sns.countplot(x="churned", data=df)
plt.title("Customer Churn Count")
plt.xlabel("Churned")
plt.ylabel("Number of Customers")
plt.show()


# ----------------------------------
# VISUALIZATION 2: Churn by Gender
# ----------------------------------

plt.figure(figsize=(7, 4))
sns.countplot(x="gender", hue="churned", data=df)
plt.title("Customer Churn by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")
plt.show()


# -----------------------------------------
# VISUALIZATION 3: Churn by Subscription
# -----------------------------------------

plt.figure(figsize=(7, 4))
sns.countplot(x="subscription_type", hue="churned", data=df)
plt.title("Churn by Subscription Type")
plt.xlabel("Subscription Type")
plt.ylabel("Number of Customers")
plt.show()


# --------------------------------
# VISUALIZATION 4: Watch Hours
# --------------------------------

plt.figure(figsize=(7, 4))
sns.histplot(df["watch_hours"], bins=20, kde=True)
plt.title("Distribution of Watch Hours")
plt.xlabel("Watch Hours")
plt.ylabel("Frequency")
plt.show()


# -----------------------------------------
# VISUALIZATION 5: Watch Hours vs Churn
# -----------------------------------------

plt.figure(figsize=(7, 4))
sns.boxplot(x="churned", y="watch_hours", data=df)
plt.title("Watch Hours vs Customer Churn")
plt.xlabel("Churned")
plt.ylabel("Watch Hours")
plt.show()


# --------------------------------
# VISUALIZATION 6: Churn by Region
# --------------------------------

plt.figure(figsize=(8, 4))
sns.countplot(x="region", hue="churned", data=df)
plt.title("Customer Churn by Region")
plt.xlabel("Region")
plt.ylabel("Number of Customers")
plt.xticks(rotation=45)
plt.show()


# Save cleaned dataset
df.to_csv("cleaned_netflix_customer_churn.csv", index=False)

print("\nCleaned dataset saved successfully!")