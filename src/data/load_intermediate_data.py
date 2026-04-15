import pandas as pd

class LoadIntermediateData:
    def __init__(self):
        # Load the intermediate data from the intermediate data folder
        self.cleaned_retention = pd.read_csv("../../data/02_intermediate/cleaned_retention.csv")
        self.cleaned_bob = pd.read_csv("../../data/02_intermediate/cleaned_bob.csv")
        print("Intermediate data loaded successfully!")