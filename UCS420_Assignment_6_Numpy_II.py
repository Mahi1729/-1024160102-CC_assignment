import numpy as np

# UCS420: Cognitive Computing - Assignment 6 (NumPy-II)

print("=" * 70)
print("Q1. SENSOR READINGS")
print("=" * 70)

temperature = np.array([25, 28, 31, 35, 38, 27, 33, 40])

# a. Add +2°C using vectorization.
corrected_temperature = temperature + 2
print("\na) Temperature after adding +2°C:")
print(corrected_temperature)

# b. Convert Celsius to Fahrenheit.
fahrenheit = (9 / 5) * corrected_temperature + 32
print("\nb) Temperatures in Fahrenheit:")
print(fahrenheit)

# c. Boolean indexing selects only readings greater than 32°C.
greater_than_32 = corrected_temperature[corrected_temperature > 32]
print("\nc) Readings greater than 32°C:")
print(greater_than_32)

# d. Count readings exceeding 32°C.
count_greater_than_32 = np.sum(corrected_temperature > 32)
print("\nd) Number of readings exceeding 32°C:")
print(count_greater_than_32)

# e. Vectorization avoids explicit loops and performs operations on
# the whole array efficiently. Boolean indexing uses a condition mask
# to directly select the required values.
print("\ne) Explanation:")
print("Vectorization performs operations on the entire NumPy array at once,")
print("which is generally faster and more efficient than explicitly iterating")
print("through every element. Boolean indexing creates a Boolean mask and")
print("directly selects the values satisfying a condition.")


print("\n" + "=" * 70)
print("Q2. DAILY STEPS")
print("=" * 70)

steps = np.array([
    [5000, 6200, 7100],
    [8000, 7500, 9000],
    [4500, 5100, 4800],
    [9000, 8500, 9500]
])

# a. Total steps recorded.
print("\na) Total steps:")
print(np.sum(steps))

# b. Mean number of steps.
print("\nb) Mean number of steps:")
print(np.mean(steps))

# c. Maximum and minimum values.
print("\nc) Maximum steps:", np.max(steps))
print("   Minimum steps:", np.min(steps))

# d. axis=0 combines rows, giving totals for each day/column.
print("\nd) Total steps for each day (axis=0):")
print(np.sum(steps, axis=0))

# e. axis=1 combines columns, giving totals for each user/row.
print("\ne) Total steps for each user (axis=1):")
print(np.sum(steps, axis=1))

# f. argmax gives the flattened index; unravel_index converts it
# back to the corresponding (user, day) position.
max_position = np.unravel_index(np.argmax(steps), steps.shape)
print("\nf) Position of maximum steps (user, day):")
print(max_position)
print("   Maximum value:", steps[max_position])


print("\n" + "=" * 70)
print("Q3. NUMPY SLICING, COPYING AND RESHAPING")
print("=" * 70)

# a. Create original array.
original = np.array([1, 2, 3, 4, 5, 6])
print("\na) Original array:")
print(original)

# b. Slice index 1 to 4 -> indices 1, 2, 3.
subset = original[1:4]
print("\nb) subset = original[1:4]:")
print(subset)

# c. A normal NumPy slice is a view, so changing it also changes original.
subset[0] = 999
print("\nc) After modifying subset[0] = 999:")
print("Original:", original)
print("Subset: ", subset)

# d. copy() creates independent data, so modifying it does not affect original.
copied_array = original[1:4].copy()
copied_array[0] = 500
print("\nd) After modifying copied_array[0] = 500:")
print("Original:     ", original)
print("Copied array: ", copied_array)

# e. Create numbers 1 to 12 and reshape them into 3 x 4.
matrix = np.arange(1, 13).reshape(3, 4)
print("\ne) 3 x 4 matrix:")
print(matrix)

# f. Indexing and slicing.
first_row = matrix[0, :]
last_row = matrix[-1, :]
second_column = matrix[:, 1]
rows_1_to_2_cols_2_to_3 = matrix[1:3, 2:4]

print("\nf) Required indexing/slicing:")
print("First row:")
print(first_row)
print("Last row:")
print(last_row)
print("Second column:")
print(second_column)
print("Elements from rows 1-2 and columns 2-3:")
print(rows_1_to_2_cols_2_to_3)

# g. flatten() creates a copy; ravel() usually creates a view.
flattened = matrix.flatten()
raveled = matrix.ravel()

print("\ng) Using flatten():")
print(flattened)
print("Using ravel():")
print(raveled)

# h. ravel() usually shares memory with original, so the original changes.
raveled[0] = 1000
print("\nh) After modifying raveled[0] = 1000:")
print("Original matrix:")
print(matrix)
print("Raveled array:")
print(raveled)

# i. flatten() creates an independent copy, so original does not change.
flattened[1] = 2000
print("\ni) After modifying flattened[1] = 2000:")
print("Original matrix:")
print(matrix)
print("Flattened array:")
print(flattened)

# j. Display matrix properties.
print("\nj) Properties of the 3 x 4 matrix:")
print("Shape:", matrix.shape)
print("ndim:", matrix.ndim)
print("Size:", matrix.size)
print("dtype:", matrix.dtype)


print("\n" + "=" * 70)
print("Q4. COGNITIVE-ASSISTIVE SYSTEM - OLS")
print("=" * 70)

y = np.array([40, 65, 30, 85])

X = np.array([
    [6, 70, 3],
    [5, 50, 6],
    [8, 80, 2],
    [4, 30, 8]
])

# a. Shape and dimensions.
print("\na) Shape of X:", X.shape)
print("   Dimensions of X:", X.ndim)

# b. Transpose changes rows into columns and columns into rows.
XT = X.T
print("\nb) X.T:")
print(XT)
print("Transpose represents each feature as a separate row.")

# c. Matrix product X.T @ X.
XTX = X.T @ X
print("\nc) X.T @ X:")
print(XTX)

# d. X.T @ X is invertible for this dataset.
XTX_inverse = np.linalg.inv(XTX)
print("\nd) Inverse of X.T @ X:")
print(XTX_inverse)

# e. Correct Ordinary Least Squares equation:
# beta = (X.T X)^(-1) X.T y
beta = XTX_inverse @ X.T @ y
print("\ne) OLS coefficients (beta):")
print(beta)

# f. Coefficients represent the effect of each feature on predicted score,
# while keeping the other features constant.
print("\nf) Meaning of coefficients:")
print("First coefficient  -> effect of one additional hour of sleep.")
print("Second coefficient -> effect of one additional unit of activity.")
print("Third coefficient  -> effect of one additional unit of stress.")
print("These effects are interpreted while keeping the other features constant.")

# g. Predict assistance score for a new user.
new_user = np.array([5, 40, 7])
predicted_score = new_user @ beta

print("\ng) New user:", new_user)
print("   Predicted assistance score:", predicted_score)

print("\n" + "=" * 70)
print("ASSIGNMENT COMPLETED")
print("=" * 70)
