import matplotlib.pyplot as plt
import seaborn as sns

class HypothesisPlotting:
    @staticmethod
    def plot_feature_histogram(df, feature, target_col='churn_category'):
        """
        Plots a stacked density histogram for a given feature grouped by target.
        """
        plt.figure(figsize=(8, 4))
        sns.histplot(data=df, x=feature, hue=target_col, multiple="stack", kde=True)
        plt.title(f'Histogram of {feature} by Churn Category')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_correlation_heatmap(df, features):
        """
        Plots a correlation heatmap for the specified continuous features.
        """
        plt.figure(figsize=(10, 8))
        corr_matrix = df[features].corr(method='spearman')
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Spearman Correlation Heatmap of Continuous Features')
        plt.tight_layout()
        plt.show()
        return corr_matrix
