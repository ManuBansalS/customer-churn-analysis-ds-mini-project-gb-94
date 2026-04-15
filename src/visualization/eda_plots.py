import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class EDAPlots:
    """
    Reusable EDA visualization functions for the Silver Layer.
    Provides interactive Plotly and static matplotlib visualizations:
    - Distribution plots, box plots, density plots, correlation heatmaps
    - Categorical bar charts grouped by target variable
    - Both Plotly (interactive/dynamic) and Matplotlib versions available
    """

    def __init__(self, data, target_column='churn_category'):
        self.data = data.copy()
        self.target = target_column

    # ------------------------------------------------------------------ #
    #  Distribution Plots (Histogram + KDE overlay by churn category)
    # ------------------------------------------------------------------ #
    def plot_distributions(self, features=None):
        """
        Histogram + KDE density overlay for continuous features,
        split by churn category. Limits x-axis to 95th percentile context.
        """
        if features is None:
            features = self.data.select_dtypes(include=[np.number]).columns.tolist()
            features = [f for f in features if f != self.target]

        n = len(features)
        fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n))
        if n == 1:
            axes = [axes]

        categories = self.data[self.target].dropna().unique()
        palette = sns.color_palette("Set2", len(categories))

        for i, feature in enumerate(features):
            for j, cat in enumerate(categories):
                subset = self.data[self.data[self.target] == cat][feature].dropna()
                axes[i].hist(subset, bins=50, alpha=0.5, density=True,
                             label=cat, color=palette[j], edgecolor='white')
            
            axes[i].set_title(f'Distribution of {feature} by Churn Category',
                              fontsize=12, fontweight='bold')
            # Limit the x-axis to the 95th percentile to make the distributions visible
            upper_limit = self.data[feature].quantile(0.95)
            if upper_limit > self.data[feature].min():
                axes[i].set_xlim(self.data[feature].min(), upper_limit)
                
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Density')
            axes[i].legend()

            # Dynamic conclusions based on simple averages
            overall_mean = self.data[feature].mean()
            churners = self.data[self.data[self.target] == 'Full Churn'][feature]
            churn_mean = churners.mean() if not churners.empty else 0
            direction = "higher" if churn_mean > overall_mean else "lower"
            print(f"Conclusion for {feature}: On average, Full Churn customers have {direction} values "
                  f"({churn_mean:.2f}) compared to the overall average ({overall_mean:.2f}). "
                  f"[Plot clipped at 95th percentile: {upper_limit:.2f} for visibility]")

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------ #
    #  Box Plots
    # ------------------------------------------------------------------ #
    def plot_boxplots(self, features=None):
        """
        Box plots for continuous features grouped by churn category.
        Displays clearly by removing extreme outliers from the visual geometry.
        """
        if features is None:
            features = self.data.select_dtypes(include=[np.number]).columns.tolist()
            features = [f for f in features if f != self.target]

        n = len(features)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
        if n == 1:
            axes = [axes]

        for i, feature in enumerate(features):
            # showfliers=False resolves the squashed boxplot visualization issue
            sns.boxplot(data=self.data, x=self.target, y=feature,
                        ax=axes[i], palette='Set2',
                        order=['No Churn', 'Partial Churn', 'Full Churn'],
                        showfliers=False)
            axes[i].set_title(f'{feature}', fontsize=11, fontweight='bold')
            axes[i].set_xlabel('Churn Category')
            axes[i].tick_params(axis='x', rotation=15)
            print(f"Boxplot Conclusion for {feature}: The plot excludes outliers to reveal the core "
                  f"interquartile range and median differences across the churn categories clearly.")

        plt.suptitle('Box Plots of Continuous Features by Churn Category (Outliers Hidden)',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------ #
    #  Density Plots (KDE)
    # ------------------------------------------------------------------ #
    def plot_density(self, features=None):
        """
        KDE density plots for continuous features by churn category.
        Avoids drawing outside natural boundaries by using seaborn cleanly.
        """
        if features is None:
            features = self.data.select_dtypes(include=[np.number]).columns.tolist()
            features = [f for f in features if f != self.target]

        n = len(features)
        fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n))
        if n == 1:
            axes = [axes]

        for i, feature in enumerate(features):
            valid_data = self.data.dropna(subset=[feature, self.target])
            if not valid_data.empty:
                # Use Seaborn's kdeplot for better appealing graphs, fill visual, and boundaries
                sns.kdeplot(data=valid_data, x=feature, hue=self.target,
                            ax=axes[i], fill=True, common_norm=False, cut=0,
                            palette='Set2', linewidth=2, warn_singular=False)
            
            axes[i].set_title(f'Density Plot: {feature}', fontsize=12, fontweight='bold')
            
            upper_limit = self.data[feature].quantile(0.95)
            if upper_limit > self.data[feature].min():
                axes[i].set_xlim(self.data[feature].min(), upper_limit)
                
            axes[i].set_xlabel(feature)
            print(f"Density Conclusion for {feature}: KDE curve shows the probability density up to the 95th percentile, highlighting the dominant distribution shapes per category.")

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------ #
    #  Correlation Heatmap
    # ------------------------------------------------------------------ #
    def plot_correlation_heatmap(self, features=None):
        """
        Correlation heatmap of numeric features showing the full grid.
        Also explicitly extracts and prints highest parameter correlations.
        """
        if features is None:
            features = self.data.select_dtypes(include=[np.number]).columns.tolist()

        corr_matrix = self.data[features].corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Remove masking. The user explicitly requested to view the full diagonal and top triangle.
        sns.heatmap(corr_matrix, annot=True, fmt='.3f',
                    cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                    square=True, linewidths=0.5, ax=ax,
                    cbar_kws={"shrink": 0.8})
        ax.set_title('Correlation Heatmap (Numeric Features)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Present the correlation metrics specifically
        print("\n--- Correlation Metrics ---")
        corr_pairs = corr_matrix.unstack().dropna()
        # Remove self-correlation of 1.0 (diagonal)
        corr_pairs = corr_pairs[corr_pairs != 1.0]
        # Sort by absolute correlation
        sorted_pairs = corr_pairs.reindex(corr_pairs.abs().sort_values(ascending=False).index).drop_duplicates()
        print("Top 10 most correlated feature pairs:")
        print(sorted_pairs.head(10))
        print("\nCorrelation Conclusion: Look for highly correlated feature pairs (|corr| > 0.70) as they might indicate multicollinearity which can negatively impact some models.")

        return corr_matrix

    # ------------------------------------------------------------------ #
    #  Categorical Bar Charts
    # ------------------------------------------------------------------ #
    def plot_categorical_bars(self, features=None):
        """
        Stacked/grouped bar charts for categorical features by churn category.
        """
        if features is None:
            features = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
            features = [f for f in features if f != self.target and f != 'account_number']

        n = len(features)
        if n == 0:
            print("No categorical features to plot.")
            return

        fig, axes = plt.subplots(n, 1, figsize=(12, 5 * n))
        if n == 1:
            axes = [axes]

        for i, feature in enumerate(features):
            ct = pd.crosstab(self.data[feature], self.data[self.target], normalize='index') * 100
            ct = ct.reindex(columns=['No Churn', 'Partial Churn', 'Full Churn'], fill_value=0)
            ct.plot(kind='barh', stacked=True, ax=axes[i], colormap='Set2',
                    edgecolor='white', linewidth=0.5)
            axes[i].set_title(f'{feature} vs Churn Category (%)',
                              fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Percentage (%)')
            axes[i].legend(title='Churn Category', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Additional conclusion piece
            print(f"Categorical Conclusion for {feature}: Proportions show how the composition of churn states varies across different categories of {feature}.")

        plt.tight_layout()
        plt.show()

    # ================================================================== #
    #  PLOTLY INTERACTIVE VISUALIZATIONS
    # ================================================================== #

    def plot_distributions_plotly(self, features=None):
        """
        Interactive Plotly histograms with density overlay for continuous features,
        split by churn category. Limits x-axis to 95th percentile context.
        """
        if features is None:
            features = self.data.select_dtypes(include=[np.number]).columns.tolist()
            features = [f for f in features if f != self.target]

        for feature in features:
            upper_limit = self.data[feature].quantile(0.95)
            filtered_data = self.data[self.data[feature] <= upper_limit]
            
            fig = px.histogram(filtered_data, x=feature, color=self.target,
                              nbins=50, barmode='overlay',
                              title=f'Distribution of {feature} by Churn Category',
                              labels={feature: feature, self.target: 'Churn Category'},
                              opacity=0.7, height=500)
            
            fig.update_layout(
                xaxis_title=feature,
                yaxis_title='Count',
                hovermode='x unified',
                template='plotly_white',
                font=dict(size=11)
            )
            fig.show()

            # Dynamic conclusions
            overall_mean = self.data[feature].mean()
            churners = self.data[self.data[self.target] == 'full_churn'][feature]
            churn_mean = churners.mean() if not churners.empty else 0
            direction = "higher" if churn_mean > overall_mean else "lower"
            print(f"Conclusion for {feature}: On average, Full Churn customers have {direction} values "
                  f"({churn_mean:.2f}) compared to the overall average ({overall_mean:.2f}). "
                  f"[Plot clipped at 95th percentile: {upper_limit:.2f} for visibility]\n")

    def plot_boxplots_plotly(self, features=None):
        """
        Interactive Plotly box plots for continuous features grouped by churn category.
        """
        if features is None:
            features = self.data.select_dtypes(include=[np.number]).columns.tolist()
            features = [f for f in features if f != self.target]

        for feature in features:
            fig = px.box(self.data, x=self.target, y=feature,
                        title=f'Box Plot: {feature} by Churn Category',
                        color=self.target,
                        height=500)
            
            fig.update_layout(
                xaxis_title='Churn Category',
                yaxis_title=feature,
                template='plotly_white',
                font=dict(size=11),
                hovermode='closest'
            )
            fig.update_traces(boxmean='sd')
            fig.show()

            print(f"Boxplot Conclusion for {feature}: The plot shows the core interquartile range, "
                  f"median, and outliers across churn categories.\n")

    def plot_density_plotly(self, features=None):
        """
        Interactive Plotly density plots (violin plots) for continuous features by churn category.
        """
        if features is None:
            features = self.data.select_dtypes(include=[np.number]).columns.tolist()
            features = [f for f in features if f != self.target]

        for feature in features:
            upper_limit = self.data[feature].quantile(0.95)
            filtered_data = self.data[self.data[feature] <= upper_limit]
            
            fig = px.violin(filtered_data, x=self.target, y=feature,
                           title=f'Density Distribution: {feature} by Churn Category',
                           color=self.target,
                           height=500, points='outliers')
            
            fig.update_layout(
                xaxis_title='Churn Category',
                yaxis_title=feature,
                template='plotly_white',
                font=dict(size=11),
                hovermode='closest'
            )
            fig.show()

            print(f"Density Conclusion for {feature}: KDE curve shows the probability density, "
                  f"highlighting distribution shapes per category (95th percentile view).\n")

    def plot_correlation_heatmap_plotly(self, features=None):
        """
        Interactive Plotly correlation heatmap of numeric features showing the full grid.
        """
        if features is None:
            features = self.data.select_dtypes(include=[np.number]).columns.tolist()

        corr_matrix = self.data[features].corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix.values, 3),
            texttemplate='%{text}',
            textfont={"size": 9},
            colorbar=dict(title="Correlation"),
            hoverongaps=False
        ))

        fig.update_layout(
            title='Correlation Heatmap (Numeric Features)',
            width=900,
            height=800,
            template='plotly_white',
            font=dict(size=10)
        )
        fig.show()

        # Present the correlation metrics
        print("\n--- Correlation Metrics ---")
        corr_pairs = corr_matrix.unstack().dropna()
        corr_pairs = corr_pairs[corr_pairs != 1.0]
        sorted_pairs = corr_pairs.reindex(corr_pairs.abs().sort_values(ascending=False).index).drop_duplicates()
        print("Top 10 most correlated feature pairs:")
        print(sorted_pairs.head(10))
        print("\nCorrelation Conclusion: Look for highly correlated feature pairs (|corr| > 0.70) "
              "as they might indicate multicollinearity.\n")

        return corr_matrix

    def plot_categorical_bars_plotly(self, features=None):
        """
        Interactive Plotly stacked bar charts for categorical features by churn category.
        """
        if features is None:
            features = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
            features = [f for f in features if f != self.target and f != 'account_number']

        if len(features) == 0:
            print("No categorical features to plot.")
            return

        for feature in features:
            ct = pd.crosstab(self.data[feature], self.data[self.target], normalize='index') * 100
            ct = ct.reindex(columns=['no_churn', 'partial_churn', 'full_churn'], fill_value=0)
            ct_reset = ct.reset_index()
            
            fig = go.Figure()
            
            churn_categories = ['no_churn', 'partial_churn', 'full_churn']
            colors = ['#1f77b4', '#ff7f0e', '#d62728']
            
            for i, churn_cat in enumerate(churn_categories):
                if churn_cat in ct.columns:
                    fig.add_trace(go.Bar(
                        x=ct_reset[feature],
                        y=ct_reset[churn_cat],
                        name=churn_cat.replace('_', ' ').title(),
                        marker_color=colors[i],
                        hovertemplate='<b>%{x}</b><br>' + churn_cat + ': %{y:.2f}%<extra></extra>'
                    ))
            
            fig.update_layout(
                title=f'{feature} vs Churn Category (%)',
                xaxis_title=feature,
                yaxis_title='Percentage (%)',
                barmode='stack',
                height=500,
                template='plotly_white',
                hovermode='x unified',
                font=dict(size=11)
            )
            fig.show()

            print(f"Categorical Conclusion for {feature}: Proportions show how the composition "
                  f"of churn states varies across different categories of {feature}.\n")
