import pandas as pd

# List of datasets to load, including emails1.csv
datasets = [
    r'C:\Users\novem\phishfortress\data\emails1.csv',
    r'C:\Users\novem\phishfortress\data\emails2.csv',
    r'C:\Users\novem\phishfortress\data\emails3.csv',
    r'C:\Users\novem\phishfortress\data\TREC_07.csv',
    r'C:\Users\novem\phishfortress\data\TREC_05.csv',
    r'C:\Users\novem\phishfortress\data\Enron (1).csv',
    r'C:\Users\novem\phishfortress\data\Nazario.csv',
    r'C:\Users\novem\phishfortress\data\Ling.csv',
    r'C:\Users\novem\phishfortress\data\TREC_06.csv',
    r'C:\Users\novem\phishfortress\data\SpamAssasin.csv',
    r'C:\Users\novem\phishfortress\data\Nazario_5.csv',
    r'C:\Users\novem\phishfortress\data\Nigerian_Fraud.csv'
]

# Iterate over each file and display the columns
for filepath in datasets:
    try:
        # Load just the first row for efficiency
        df = pd.read_csv(filepath, nrows=1)
        print(f"Columns for {filepath}: {list(df.columns)}")
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
