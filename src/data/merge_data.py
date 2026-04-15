import pandas as pd


class MergeData:
    def __init__(self, retention_cleaned, bob_cleaned):
        self.retention_cleaned = retention_cleaned.copy()
        self.bob_cleaned = bob_cleaned.copy()
        self.merged_df = None

    def merge_data(self):
        """
        Merge BoB and Retention data using LEFT JOIN from BoB.

        LEFT join ensures ALL BoB customers are retained in the dataset.
        Customers without retention records (no risk/cancellation cases)
        are naturally classified as no-churn in downstream analysis.

        Returns
        -------
        pd.DataFrame
            Merged DataFrame with all BoB records + matching Retention records.
        """
        print("Merging retention and BoB data (LEFT JOIN)...")

        # Ensure same datatype
        self.bob_cleaned['account_number'] = self.bob_cleaned['account_number'].astype(str).str.strip()
        self.retention_cleaned['customer_account_number'] = (
            self.retention_cleaned['customer_account_number'].astype(str).str.strip()
        )

        # LEFT JOIN: keep all BoB customers
        merged = pd.merge(
            self.bob_cleaned,
            self.retention_cleaned,
            left_on='account_number',
            right_on='customer_account_number',
            how='left'
        )

        # Remove duplicate column
        if 'customer_account_number' in merged.columns:
            merged.drop(columns=['customer_account_number'], inplace=True)

        # Fill NaN for unmatched retention columns
        if 'case_type' in merged.columns:
            merged['case_type'] = merged['case_type'].fillna('No Case')
        if 'resolution_status' in merged.columns:
            merged['resolution_status'] = merged['resolution_status'].fillna('No Case')
        if 'current_status' in merged.columns:
            merged['current_status'] = merged['current_status'].fillna('No Case')

        self.merged_df = merged

        print(f"Merged dataset shape: {self.merged_df.shape}")
        print(f"Columns: {self.merged_df.columns.tolist()}")
        print(f"Unique customers: {self.merged_df['account_number'].nunique()}")

        return self.merged_df