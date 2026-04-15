"""
Silver Layer - Feature Engineering Module
==========================================
Aggregates row-level merged data into customer-level features
for EDA, hypothesis testing, and downstream modeling.

Features are designed to be PREDICTIVE of churn without directly
encoding the churn outcome (avoiding data leakage).
"""

import pandas as pd
import numpy as np


class FeatureEngineering:
    def __init__(self, merged_df):
        self.merged_df = merged_df.copy()

    def engineer_features(self):
        """
        Engineer customer-level predictive features from the merged dataset.

        Feature Groups:
            - Financial: revenue, fees, product value, unit amounts
            - Agreement: count, types, branches, duration
            - Behavioral: BoB ratio, retention case counts, case mix

        Returns
        -------
        pd.DataFrame
            Customer-level feature DataFrame (one row per customer).
        """
        print("Starting customer-level feature engineering...")
        df = self.merged_df

        # ── Parse dates ──
        df['agreement_start_date'] = pd.to_datetime(df['agreement_start_date'], errors='coerce')
        df['agreement_end_date'] = pd.to_datetime(df['agreement_end_date'], errors='coerce')
        df['agreement_duration_days'] = (df['agreement_end_date'] - df['agreement_start_date']).dt.days

        # ── is_bob flag ──
        df['is_bob_flag'] = (df['is_bob'].str.strip().str.lower() == 'yes').astype(int)

        # ── Fill missing numeric ──
        df['unit_amount'] = df['unit_amount'].fillna(df['unit_amount'].median())

        # ── Retention case flag (actual cases, not 'No Case') ──
        df['has_retention_case'] = (df['resolution_status'] != 'No Case').astype(int)

        # ============================================================
        #  Aggregate to Customer Level
        # ============================================================

        # --- Financial features ---
        financial = df.groupby('account_number').agg(
            total_revenue=('total_bob', 'sum'),
            avg_revenue=('total_bob', 'mean'),
            total_product_value=('product_bob', 'sum'),
            total_fees=('fee_bob', 'sum'),
            avg_fees=('fee_bob', 'mean'),
            avg_unit_amount=('unit_amount', 'mean'),
            max_unit_amount=('unit_amount', 'max'),
        ).reset_index()

        # --- Agreement diversity features ---
        diversity = df.groupby('account_number').agg(
            num_agreements=('agreement_number', 'nunique'),
            num_branches=('branch', 'nunique'),
            num_agreement_types=('agreement_type', 'nunique'),
            num_products=('product_name', 'nunique'),
        ).reset_index()

        # --- Duration features ---
        duration = df.groupby('account_number').agg(
            avg_agreement_duration=('agreement_duration_days', 'mean'),
            max_agreement_duration=('agreement_duration_days', 'max'),
        ).reset_index()

        # --- BoB ratio ---
        bob_ratio = df.groupby('account_number')['is_bob_flag'].mean().reset_index()
        bob_ratio.rename(columns={'is_bob_flag': 'bob_ratio'}, inplace=True)

        # --- Retention interaction count ---
        retention_count = df.groupby('account_number')['has_retention_case'].sum().reset_index()
        retention_count.rename(columns={'has_retention_case': 'num_retention_cases'}, inplace=True)

        # --- Dominant agreement type ---
        dom_agr_type = (df.groupby('account_number')['agreement_type']
                        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown')
                        .reset_index())
        dom_agr_type.rename(columns={'agreement_type': 'dominant_agreement_type'}, inplace=True)

        # ── Merge everything ──
        customer_df = financial
        for feat_df in [diversity, duration, bob_ratio, retention_count, dom_agr_type]:
            customer_df = customer_df.merge(feat_df, on='account_number', how='left')

        # ── Derived features ──
        customer_df['revenue_per_agreement'] = (
            customer_df['total_revenue'] / customer_df['num_agreements'].replace(0, np.nan)
        )
        customer_df['fee_to_revenue_ratio'] = (
            customer_df['total_fees'] / customer_df['total_revenue'].replace(0, np.nan)
        )

        # ── Clean up ──
        customer_df = customer_df.fillna(0)
        customer_df.replace([np.inf, -np.inf], 0, inplace=True)

        print(f"Feature engineering complete. Customer-level shape: {customer_df.shape}")
        print(f"Features: {customer_df.columns.tolist()}")

        return customer_df
