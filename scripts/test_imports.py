# test_imports.py

# Import necessary libraries and print a success message if each import is successful
try:
    import pandas as pd
    print("pandas imported successfully")
except ImportError as e:
    print(f"Error importing pandas: {e}")

try:
    import numpy as np
    print("numpy imported successfully")
except ImportError as e:
    print(f"Error importing numpy: {e}")

try:
    import sklearn
    print("sklearn imported successfully")
except ImportError as e:
    print(f"Error importing sklearn: {e}")

try:
    import tensorflow as tf
    print("tensorflow imported successfully")
except ImportError as e:
    print(f"Error importing tensorflow: {e}")

try:
    import nltk
    print("nltk imported successfully")
except ImportError as e:
    print(f"Error importing nltk: {e}")

try:
    import spacy
    print("spacy imported successfully")
except ImportError as e:
    print(f"Error importing spacy: {e}")

try:
    import xgboost as xgb
    print("xgboost imported successfully")
except ImportError as e:
    print(f"Error importing xgboost: {e}")

try:
    import requests
    print("requests imported successfully")
except ImportError as e:
    print(f"Error importing requests: {e}")

# Final message
print("All import tests completed.")
