"""
Silver Layer - Churn Classification Module
============================================
Classifies customers into 3 churn categories based on the use case:

  - resolution_status == 'Customer Lost' → churn for that agreement
  - Customer-level classification by aggregating retention outcomes:
    > no_churn     : No 'Customer Lost' in any retention case (or no retention case at all)
    > partial_churn: Some retention cases are 'Customer Lost', but not all
    > full_churn   : ALL retention cases resulted in 'Customer Lost'

Works with the LEFT-joined merged data so ALL BoB customers are included.
"""

import pandas as pd


class ChurnClassification:
    def __init__(self, merged_df):
        self.merged_df = merged_df.copy()
        self.churn_df = None

    def classify_churn(self):
        """
        Classify each customer into no_churn, partial_churn, or full_churn
        using aggregated retention outcomes per customer.

        Logic:
            1. Customers with NO retention records  → no_churn
            2. Customers IN retention but 0 'Customer Lost' → no_churn
            3. Customers IN retention with ALL 'Customer Lost' → full_churn
            4. Customers IN retention with SOME 'Customer Lost' → partial_churn

        Returns
        -------
        pd.DataFrame
            Customer-level DataFrame with churn classification.
        """
        print("Performing churn classification...")
        df = self.merged_df

        # ── Per-customer: count total retention cases ──
        # Only count actual retention cases, not 'No Case' (from LEFT join NaN fill)
        has_case = df['resolution_status'] != 'No Case'

        total_cases = df[has_case].groupby('account_number').size().reset_index(name='total_retention_cases')

        # ── Per-customer: count 'Customer Lost' cases ──
        lost_mask = df['resolution_status'] == 'Customer Lost'
        lost_cases = df[lost_mask].groupby('account_number').size().reset_index(name='customer_lost_cases')

        # ── Per-customer: count 'Customer Saved' cases ──
        saved_mask = df['resolution_status'] == 'Customer Saved'
        saved_cases = df[saved_mask].groupby('account_number').size().reset_index(name='customer_saved_cases')

        # ── All unique customers from BoB ──
        all_customers = pd.DataFrame({'account_number': df['account_number'].unique()})

        # ── Merge counts ──
        result = all_customers.merge(total_cases, on='account_number', how='left')
        result = result.merge(lost_cases, on='account_number', how='left')
        result = result.merge(saved_cases, on='account_number', how='left')

        result[['total_retention_cases', 'customer_lost_cases', 'customer_saved_cases']] = (
            result[['total_retention_cases', 'customer_lost_cases', 'customer_saved_cases']].fillna(0).astype(int)
        )

        # ── Also get agreement counts from BoB ──
        agreements = df.groupby('account_number')['agreement_number'].nunique().reset_index()
        agreements.rename(columns={'agreement_number': 'num_agreements'}, inplace=True)
        result = result.merge(agreements, on='account_number', how='left')

        # ── Churn classification ──
        def classify(row):
            if row['total_retention_cases'] == 0:
                # No retention case at all → no churn
                return 'no_churn'
            elif row['customer_lost_cases'] == 0:
                # Had retention cases but none resulted in loss → no churn
                return 'no_churn'
            elif row['customer_lost_cases'] >= row['total_retention_cases']:
                # ALL retention cases ended in Customer Lost → full churn
                return 'full_churn'
            else:
                # SOME retention cases ended in Customer Lost → partial churn
                return 'partial_churn'

        result['churn_category'] = result.apply(classify, axis=1)

        self.churn_df = result

        # ── Print summary ──
        print(f"\nChurn classification complete.")
        print(f"Total customers: {len(result)}")
        print(f"\nChurn Distribution:")
        dist = result['churn_category'].value_counts()
        for cat, count in dist.items():
            pct = count / len(result) * 100
            print(f"  {cat:>15s}: {count:>6d} ({pct:.1f}%)")

        return self.churn_df