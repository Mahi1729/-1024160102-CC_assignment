import pandas as pd


# ============================================================
# Assignment-3
# UCS420 "Cognitive Computing"
# Topic: Pandas
# ============================================================


# ============================================================
# Q1. Create the dataset
# ============================================================

data = {
    "Tid": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Refund": ["Yes", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No"],
    "Marital Status": [
        "Single", "Married", "Single", "Married", "Divorced",
        "Married", "Divorced", "Single", "Married", "Single"
    ],
    "Taxable Income": [
        "125K", "100K", "70K", "120K", "95K",
        "60K", "220K", "85K", "75K", "90K"
    ],
    "Cheat": ["No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "Yes"]
}

df = pd.DataFrame(data)

print("=" * 60)
print("Q1. DATASET")
print("=" * 60)
print(df)


# ============================================================
# Q2. Locate row 0, 4, 7 and 8
# ============================================================

print("\n" + "=" * 60)
print("Q2. ROWS 0, 4, 7 AND 8")
print("=" * 60)

print(df.iloc[[0, 4, 7, 8]])


# ============================================================
# Q3. Navigate the DataFrame
# ============================================================

# Q3.1 Select rows from index 3 to 7

print("\n" + "=" * 60)
print("Q3.1. ROWS FROM INDEX 3 TO 7")
print("=" * 60)

print(df.iloc[3:8])


# Q3.2 Select rows from index 4 to 8
# and columns 2 to 4

print("\n" + "=" * 60)
print("Q3.2. ROWS 4 TO 8 AND COLUMNS 2 TO 4")
print("=" * 60)

print(df.iloc[4:9, 2:5])


# Q3.3 Select all rows with column index 1 to 3

print("\n" + "=" * 60)
print("Q3.3. ALL ROWS, COLUMNS 1 TO 3")
print("=" * 60)

print(df.iloc[:, 1:4])


# ============================================================
# Q4. Read a CSV file and display first five rows
# ============================================================

print("\n" + "=" * 60)
print("Q4. IRIS DATASET - FIRST FIVE ROWS")
print("=" * 60)

# Make sure Iris.csv is present in the same folder as this file.
try:
    iris = pd.read_csv("Iris.csv")
    print(iris.head(5))
except FileNotFoundError:
    print("Iris.csv was not found.")
    print("Download Iris.csv from Kaggle and place it in the same folder.")


# ============================================================
# Q5. Delete row 4 and column 3 from Iris dataset
# ============================================================

print("\n" + "=" * 60)
print("Q5. AFTER DELETING ROW 4 AND COLUMN 3")
print("=" * 60)

try:
    iris = pd.read_csv("Iris.csv")

    # Delete row 4
    iris = iris.drop(index=4)

    # Delete column with index 3
    iris = iris.drop(iris.columns[3], axis=1)

    print(iris)

except FileNotFoundError:
    print("Iris.csv was not found. Q5 could not be performed.")


# ============================================================
# Q6. Employee Dataset
# ============================================================

employee_data = {
    "Employee_ID": [101, 102, 103, 104, 105],
    "Name": ["Alice", "Bob", "Charlie", "Diana", "Edward"],
    "Department": ["HR", "IT", "IT", "Marketing", "Sales"],
    "Age": [29, 34, 41, 28, 38],
    "Salary": [50000, 70000, 65000, 55000, 60000],
    "Years_of_Experience": [4, 8, 10, 3, 12],
    "Joining_Date": [
        "2020-03-15", "2017-07-19", "2013-06-01",
        "2021-02-10", "2010-11-25"
    ],
    "Gender": ["Female", "Male", "Male", "Female", "Male"],
    "Bonus": [5000, 7000, 6000, 4500, 5000],
    "Rating": [4.5, 4.0, 3.8, 4.7, 3.5]
}

employees = pd.DataFrame(employee_data)

# Save the original employee dataset as employees.csv
employees.to_csv("employees.csv", index=False)


# ============================================================
# Q6(a). Shape of DataFrame
# ============================================================

print("\n" + "=" * 60)
print("Q6(a). SHAPE OF DATAFRAME")
print("=" * 60)

print(employees.shape)


# ============================================================
# Q6(b). Summary of DataFrame
# ============================================================

print("\n" + "=" * 60)
print("Q6(b). SUMMARY / INFORMATION")
print("=" * 60)

employees.info()


# ============================================================
# Q6(c). Descriptive Statistics
# ============================================================

print("\n" + "=" * 60)
print("Q6(c). DESCRIPTIVE STATISTICS")
print("=" * 60)

print(employees.describe())


# ============================================================
# Q6(d). First 5 rows and last 3 rows
# ============================================================

print("\n" + "=" * 60)
print("Q6(d). FIRST 5 ROWS")
print("=" * 60)

print(employees.head(5))

print("\n" + "=" * 60)
print("Q6(d). LAST 3 ROWS")
print("=" * 60)

print(employees.tail(3))


# ============================================================
# Q6(e). Statistics
# ============================================================

print("\n" + "=" * 60)
print("Q6(e). STATISTICS")
print("=" * 60)

# i. Average salary
average_salary = employees["Salary"].mean()
print("i. Average Salary:", average_salary)

# ii. Total bonus
total_bonus = employees["Bonus"].sum()
print("ii. Total Bonus:", total_bonus)

# iii. Youngest employee's age
youngest_age = employees["Age"].min()
print("iii. Youngest Age:", youngest_age)

# iv. Highest performance rating
highest_rating = employees["Rating"].max()
print("iv. Highest Rating:", highest_rating)


# ============================================================
# Q6(f). Sort DataFrame by Salary in descending order
# ============================================================

print("\n" + "=" * 60)
print("Q6(f). SORTED BY SALARY - DESCENDING")
print("=" * 60)

sorted_employees = employees.sort_values(
    by="Salary",
    ascending=False
)

print(sorted_employees)


# ============================================================
# Q6(g). Add Performance Category
# ============================================================

def performance(rating):
    if rating >= 4.5:
        return "Excellent"
    elif rating >= 4.0:
        return "Good"
    else:
        return "Average"


employees["Performance"] = employees["Rating"].apply(performance)

print("\n" + "=" * 60)
print("Q6(g). PERFORMANCE CATEGORY")
print("=" * 60)

print(employees[["Name", "Rating", "Performance"]])


# ============================================================
# Q6(h). Identify missing values
# ============================================================

print("\n" + "=" * 60)
print("Q6(h). MISSING VALUES")
print("=" * 60)

print(employees.isnull().sum())


# ============================================================
# Q6(i). Rename Employee_ID column to ID
# ============================================================

employees = employees.rename(columns={"Employee_ID": "ID"})

print("\n" + "=" * 60)
print("Q6(i). AFTER RENAMING Employee_ID TO ID")
print("=" * 60)

print(employees)


# ============================================================
# Q6(j). Find employees
# ============================================================

# i. More than 5 years of experience

print("\n" + "=" * 60)
print("Q6(j)(i). MORE THAN 5 YEARS OF EXPERIENCE")
print("=" * 60)

experience_result = employees[
    employees["Years_of_Experience"] > 5
]

print(experience_result)


# ii. Belong to the IT department

print("\n" + "=" * 60)
print("Q6(j)(ii). EMPLOYEES IN IT DEPARTMENT")
print("=" * 60)

it_result = employees[
    employees["Department"] == "IT"
]

print(it_result)


# ============================================================
# Q6(k). Add Tax column
# ============================================================

employees["Tax"] = employees["Salary"] * 0.10

print("\n" + "=" * 60)
print("Q6(k). AFTER ADDING TAX COLUMN")
print("=" * 60)

print(employees)


# ============================================================
# Q6(l). Save modified DataFrame to a new CSV file
# ============================================================

employees.to_csv("modified_employees.csv", index=False)

print("\n" + "=" * 60)
print("Q6(l). SAVING MODIFIED DATASET")
print("=" * 60)

print("Modified dataset saved as: modified_employees.csv")
