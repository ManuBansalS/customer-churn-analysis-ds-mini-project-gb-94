class StoreData:
    def __init__(self, retention, bob):
        self.retention = retention
        self.bob = bob
        
    def save_data(self):
        self.retention.to_csv("../../data/02_intermediate/cleaned_retention.csv", index=False)
        self.bob.to_csv("../../data/02_intermediate/cleaned_bob.csv", index=False)
        print("Data saved successfully to the intermediate data folder!")