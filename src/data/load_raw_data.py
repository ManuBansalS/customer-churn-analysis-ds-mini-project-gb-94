import pandas as pd

class LoadData:
    def __init__(self):
        # Load the raw data from the Raw data folder
        self.retention = pd.read_csv("../../data/01_raw/Retention.csv")
        self.bob = pd.read_csv("../../data/01_raw/BoB.csv")
        print("Raw data loaded successfully!")