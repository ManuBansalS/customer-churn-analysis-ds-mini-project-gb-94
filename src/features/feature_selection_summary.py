import pandas as pd

class FeatureSelectionSummary:
    def __init__(self, hypothesis_results, data):
        self.results = hypothesis_results
        self.data = data.copy()

    def generate_summary(self):
        """
        Generate comprehensive feature selection summary based on hypothesis testing results
        """
        print("Generating Feature Selection Summary...")

        # Extract significant features from different tests
        significant_features = set()

        # From Chi-Square (Hypothesis 1)
        if 'hypothesis_1' in self.results:
            for result in self.results['hypothesis_1']:
                if result['Significant']:
                    significant_features.add(result['Feature'])

        # From Kruskal-Wallis (Hypothesis 3) - more robust for non-normal data
        if 'hypothesis_3' in self.results:
            for result in self.results['hypothesis_3']:
                if result['Significant']:
                    significant_features.add(result['Feature'])

        # From Correlation (Hypothesis 6)
        if 'hypothesis_6' in self.results:
            for result in self.results['hypothesis_6']:
                if abs(result['Spearman_Corr']) > 0.3:  # Moderate correlation threshold
                    significant_features.add(result['Feature'])

        significant_features = list(significant_features)

        print(f"\nSIGNIFICANT FEATURES IDENTIFIED ({len(significant_features)} features):")
        for idx, feature in enumerate(significant_features, 1):
            print(f"  {idx}. {feature}")

        # Create feature importance visualization data
        feature_data = {
            'Feature': significant_features,
            'Type': ['Categorical' if 'tier' in f or 'type' in f or 'status' in f else 'Continuous' for f in significant_features],
        }
        feature_summary_df = pd.DataFrame(feature_data)

        print("\nFEATURE SUMMARY TABLE")
        print(feature_summary_df.to_string(index=False))

        # Dataset summary
        print(f"\nDATA READY FOR MODELING STAGE")
        print(f"Final dataset shape: {self.data.shape}")
        print(f"Target variable (Churn Category) distribution:")
        print(self.data['churn_category'].value_counts())
        print(f"\nChurn categories and key metrics:")
        churn_summary = self.data.groupby('churn_category')[['total_agreements', 'active_agreements', 'lost_agreements']].agg(['mean', 'std', 'min', 'max'])
        print(churn_summary)

        return {
            'significant_features': significant_features,
            'feature_summary': feature_summary_df,
            'dataset_info': {
                'shape': self.data.shape,
                'churn_distribution': self.data['churn_category'].value_counts().to_dict(),
                'churn_summary': churn_summary
            }
        }