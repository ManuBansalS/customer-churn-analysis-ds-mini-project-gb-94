import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd


class NormalityPlots:
    """
    Reusable normality visualization functions.
    Generates Q-Q plots and histograms with normal curve overlay
    for checking the normality assumption of continuous features.
    """

    def __init__(self, data):
        self.data = data.copy()

    def plot_qq_and_histogram(self, features):
        """
        Generate side-by-side Q-Q plots and histograms for the given features.

        Args:
            features: list of column names to check for normality
        """
        valid_features = [f for f in features if f in self.data.columns]
        n = len(valid_features)
        if n == 0:
            print("No valid features found to plot.")
            return

        fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
        if n == 1:
            axes = axes.reshape(1, -1)

        for i, feature in enumerate(valid_features):
            values = self.data[feature].dropna()

            # Handle outliers for visualization by filtering top 1% 
            # so the graph is drawn with appropriate visualization as requested
            q99 = values.quantile(0.99)
            capped_values = values[values <= q99]
            
            # --- Histogram with normal curve overlay ---
            axes[i, 0].hist(capped_values, bins=50, density=True, alpha=0.7,
                            color='steelblue', edgecolor='white')
            mu, sigma = capped_values.mean(), capped_values.std()
            x_range = np.linspace(capped_values.min(), capped_values.max(), 200)
            axes[i, 0].plot(x_range, stats.norm.pdf(x_range, mu, sigma),
                            'r-', linewidth=2, label='Normal Fit')
            axes[i, 0].set_title(f'Histogram: {feature} (Capped 99th)',
                                 fontsize=11, fontweight='bold')
            axes[i, 0].set_xlabel(feature)
            axes[i, 0].set_ylabel('Density')
            axes[i, 0].legend()
            
            # --- Q-Q Plot ---
            # Q-Q plot works better when the bulk values are visualised without getting squashed by massive outliers.
            stats.probplot(capped_values, dist="norm", plot=axes[i, 1])
            axes[i, 1].set_title(f'Q-Q Plot: {feature} (Capped 99th)',
                                 fontsize=11, fontweight='bold')
            axes[i, 1].get_lines()[0].set(markerfacecolor='steelblue',
                                           markeredgecolor='steelblue',
                                           markersize=3)
            axes[i, 1].get_lines()[1].set(color='red', linewidth=2)

        plt.tight_layout()
        plt.show()

    def plot_qq_and_histogram_plotly(self, features):
        """
        Generate interactive Plotly Q-Q plots and histograms for the given features.
        Uses subplots to show Q-Q plot and histogram side by side.

        Args:
            features: list of column names to check for normality
        """
        valid_features = [f for f in features if f in self.data.columns]
        n = len(valid_features)
        
        if n == 0:
            print("No valid features found to plot.")
            return

        for i, feature in enumerate(valid_features):
            values = self.data[feature].dropna()

            # Handle outliers by capping at 99th percentile
            q99 = values.quantile(0.99)
            capped_values = values[values <= q99]
            
            # Create subplots
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=(f'Histogram: {feature} (Capped 99th)', 
                               f'Q-Q Plot: {feature} (Capped 99th)'),
                specs=[[{'type': 'histogram'}, {'type': 'scatter'}]]
            )

            # --- Histogram with normal curve overlay ---
            fig.add_trace(
                go.Histogram(
                    x=capped_values,
                    nbinsx=50,
                    name='Data',
                    opacity=0.7,
                    marker_color='steelblue',
                    showlegend=False
                ),
                row=1, col=1
            )

            # Add normal distribution overlay
            mu, sigma = capped_values.mean(), capped_values.std()
            x_range = np.linspace(capped_values.min(), capped_values.max(), 200)
            y_range = stats.norm.pdf(x_range, mu, sigma) * len(capped_values) * (capped_values.max() - capped_values.min()) / 50
            
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_range,
                    mode='lines',
                    name='Normal Fit',
                    line=dict(color='red', width=2),
                    showlegend=True
                ),
                row=1, col=1
            )

            # --- Q-Q Plot ---
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(capped_values)))
            sample_quantiles = np.sort(capped_values)
            
            fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=sample_quantiles,
                    mode='markers',
                    name='Sample Data',
                    marker=dict(color='steelblue', size=4),
                    showlegend=False
                ),
                row=1, col=2
            )

            # Add Q-Q reference line
            min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
            max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
            ref_line = np.linspace(min_val, max_val, 100)
            
            fig.add_trace(
                go.Scatter(
                    x=ref_line,
                    y=ref_line,
                    mode='lines',
                    name='Reference',
                    line=dict(color='red', width=2),
                    showlegend=False
                ),
                row=1, col=2
            )

            fig.update_xaxes(title_text='Value', row=1, col=1)
            fig.update_yaxes(title_text='Density', row=1, col=1)
            fig.update_xaxes(title_text='Theoretical Quantiles', row=1, col=2)
            fig.update_yaxes(title_text='Sample Quantiles', row=1, col=2)

            fig.update_layout(
                title=f'Normality Assessment: {feature}',
                height=500,
                width=1000,
                template='plotly_white',
                font=dict(size=10),
                hovermode='closest'
            )

            fig.show()

            print(f"Normality Assessment for {feature}: "
                  f"Histogram shows distribution shape (capped at 99th percentile). "
                  f"Q-Q plot shows how closely data follows normal distribution. "
                  f"Points near the red line indicate normality.\n")
