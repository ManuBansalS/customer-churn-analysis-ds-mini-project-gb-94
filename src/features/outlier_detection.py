import pandas as pd
import numpy as np


class OutlierDetection:
    """
    IQR-based outlier detection for continuous features.
    Provides summary statistics and identification of outlier rows.
    """

    def __init__(self, data):
        self.data = data.copy()

    def detect_outliers_iqr(self, features=None, factor=1.5):
        """
        Detect outliers using the Inter-Quartile Range (IQR) method.

        A value is an outlier if:
            value < Q1 - factor * IQR   or   value > Q3 + factor * IQR

        Args:
            features: list of column names (defaults to all numeric columns)
            factor: IQR multiplier (default 1.5 for mild outliers, 3.0 for extreme)

        Returns:
            DataFrame with outlier summary per feature
        """
        if features is None:
            features = self.data.select_dtypes(include=[np.number]).columns.tolist()

        results = []
        for feature in features:
            if feature not in self.data.columns:
                continue

            values = self.data[feature].dropna()
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR

            outlier_mask = (values < lower_bound) | (values > upper_bound)
            n_outliers = outlier_mask.sum()
            pct_outliers = round((n_outliers / len(values)) * 100, 2)

            results.append({
                'Feature': feature,
                'Q1': round(Q1, 4),
                'Q3': round(Q3, 4),
                'IQR': round(IQR, 4),
                'Lower_Bound': round(lower_bound, 4),
                'Upper_Bound': round(upper_bound, 4),
                'Num_Outliers': n_outliers,
                'Pct_Outliers': pct_outliers,
                'Total_Rows': len(values)
            })

        summary_df = pd.DataFrame(results)
        print("Outlier Detection Summary (IQR Method):")
        print(f"IQR Factor: {factor}")
        print(summary_df.to_string(index=False))
        return summary_df
