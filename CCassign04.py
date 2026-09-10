import numpy as np

# ============================================================
# Assignment-4 UCS420: Cognitive Computing
# NumPy - Introduction-1
# ============================================================


# ============================================================
# Q1. Basic NumPy Array Operations
# ============================================================

# Create a 1-D NumPy array with 5 elements.
arr = np.array([10, 20, 30, 40, 50])

print("=" * 60)
print("Q1. BASIC NUMPY ARRAY OPERATIONS")
print("=" * 60)

# Vectorization applies the operation to every element without a loop.
print("\nOriginal array:")
print(arr)

# a) Addition of 2 to all elements.
print("\na) Add 2 to every element:")
print(arr + 2)

# b) Multiply every element by 3.
print("\nb) Multiply every element by 3:")
print(arr * 3)

# c) Divide every element by 2.
print("\nc) Divide every element by 2:")
print(arr / 2)


# ============================================================
# Q2. Basic NumPy Array Questions
# ============================================================

print("\n" + "=" * 60)
print("Q2. BASIC NUMPY ARRAY QUESTIONS")
print("=" * 60)

# a) Reverse the NumPy array.
arr = np.array([1, 2, 3, 6, 4, 5])

# [::-1] traverses the array from the last element to the first.
print("\na) Original array:")
print(arr)

print("Reversed array:")
print(arr[::-1])


# b-i) Find the most frequent value and its indices.
x = np.array([1, 2, 3, 4, 5, 1, 2, 1, 1, 1])

# np.unique(..., return_counts=True) gives each distinct value and its frequency.
values, counts = np.unique(x, return_counts=True)
most_frequent = values[np.argmax(counts)]
indices = np.where(x == most_frequent)[0]

print("\nb-i) Array x:")
print(x)
print("Most frequent value:", most_frequent)
print("Frequency:", np.max(counts))
print("Indices:", indices)


# b-ii) Find the most frequent value and its indices.
y = np.array([1, 1, 1, 2, 3, 4, 2, 4, 3, 3])

values, counts = np.unique(y, return_counts=True)
most_frequent = values[np.argmax(counts)]
indices = np.where(y == most_frequent)[0]

print("\nb-ii) Array y:")
print(y)
print("Most frequent value:", most_frequent)
print("Frequency:", np.max(counts))
print("Indices:", indices)


# ============================================================
# Q3. Accessing Elements of a 2-D Array
# ============================================================

print("\n" + "=" * 60)
print("Q3. ACCESSING ELEMENTS OF A 2-D ARRAY")
print("=" * 60)

# A 2-D NumPy array is created using nested lists.
arr = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

print("\n2-D array:")
print(arr)

# NumPy uses zero-based indexing: row 1, column 2 -> [0, 1].
print("\na) 1st row, 2nd column:")
print(arr[0, 1])

# 3rd row, 1st column -> [2, 0].
print("\nb) 3rd row, 1st column:")
print(arr[2, 0])


# ============================================================
# Q4. linspace(), Array Properties and Transpose
# ============================================================

print("\n" + "=" * 60)
print("Q4. LINSPACE, ARRAY PROPERTIES AND TRANSPOSE")
print("=" * 60)


Mahi = np.linspace(10, 100, 25)

print("\nArray named Mahi:")
print(Mahi)

# a) Number of dimensions.
print("\nDimensions (ndim):")
print(Mahi.ndim)

# b) Shape of the array.
print("\nShape:")
print(Mahi.shape)

# c) Total number of elements.
print("\nTotal elements:")
print(Mahi.size)

# d) Data type of each element.
print("\nData type:")
print(Mahi.dtype)

# e) Total number of bytes occupied by the array.
print("\nTotal bytes consumed:")
print(Mahi.nbytes)

# For a 1-D array, reshape(1, -1) converts it into a row vector.
# This makes the transpose visible as a column vector.
Mahi_transpose = Mahi.reshape(1, 25).T

print("\nTranspose using reshape():")
print(Mahi_transpose)

# Yes, T can also transpose a 2-D row-vector representation.
Mahi_T = Mahi.reshape(1, 25).T

print("\nTranspose using T attribute:")
print(Mahi_T)

print("\nCan we do the same with T attribute?")
print("Yes. After reshaping the 1-D array into a 2-D row vector,")
print("the T attribute can be used to transpose it into a column vector.")


# ============================================================
# Q5. 2-D Array Operations, Reshape and Resize
# ============================================================

print("\n" + "=" * 60)
print("Q5. 2-D ARRAY OPERATIONS")
print("=" * 60)

# Create a 3 x 4 array with the given 12 values.
ucs420_Mahi = np.array([
    [10, 20, 30, 40],
    [50, 60, 70, 80],
    [90, 15, 20, 35]
])

print("\nOriginal array - ucs420_Mahi:")
print(ucs420_Mahi)

# Calculate statistical properties over all elements.
print("\nMean:")
print(np.mean(ucs420_Mahi))

print("\nMedian:")
print(np.median(ucs420_Mahi))

print("\nMaximum:")
print(np.max(ucs420_Mahi))

print("\nMinimum:")
print(np.min(ucs420_Mahi))

# np.unique returns each value only once, removing duplicates.
print("\nUnique elements:")
print(np.unique(ucs420_Mahi))

# Reshape changes the dimensions while keeping all 12 elements.
reshaped_ucs420_Mahi = ucs420_Mahi.reshape(4, 3)

print("\nReshaped array - reshaped_ucs420_Mahi (4 x 3):")
print(reshaped_ucs420_Mahi)

# resize() changes the total number of elements.
# Since 2 x 3 needs only 6 elements, the first 6 values are retained.
resized_ucs420_Mahi = np.resize(ucs420_Mahi, (2, 3))

print("\nResized array - resized_ucs420_Mahi (2 x 3):")
print(resized_ucs420_Mahi)


print("\n" + "=" * 60)
print("ASSIGNMENT-4 COMPLETED")
print("=" * 60)
