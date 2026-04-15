"""
Silver Layer - Hypothesis Testing Module
==========================================
Consolidated module with 6 business-driven hypotheses for churn analysis.
Each hypothesis uses appropriate statistical tests on customer-level data.

Hypotheses:
    H1: Agreement type is independent of churn category       → Chi-Square
    H2: Revenue does not differ across churn categories        → Kruskal-Wallis
    H3: Number of agreements does not differ across categories → Kruskal-Wallis
    H4: Fee structure does not differ across churn categories  → Kruskal-Wallis
    H5: Agreement duration has no relationship with churn      → Kruskal-Wallis
    H6: Financial metrics have no correlation with churn        → Spearman
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, kruskal, spearmanr
from scipy import stats


class HypothesisTesting:
    """Run 6 hypothesis tests on customer-level churn data."""

    def __init__(self, data, target_column='churn_category'):
        self.data = data.copy()
        self.target = target_column

    # ────────────────────────────────────────────────────────────────
    #  H1: Chi-Square Test – Agreement Type vs Churn Category
    # ────────────────────────────────────────────────────────────────
    def hypothesis_1_chi_square(self):
        """
        H1: Dominant agreement type is independent of churn category.
        Test: Chi-Square Test of Independence
        """
        print("=" * 65)
        print("  HYPOTHESIS 1: Agreement Type vs Churn (Chi-Square)")
        print("=" * 65)
        print("H0: Agreement type is independent of churn category")
        print("H1: Agreement type is associated with churn category\n")

        feature = 'dominant_agreement_type'
        if feature not in self.data.columns:
            print(f"  Column '{feature}' not found. Skipping.")
            return []

        test_data = self.data[[feature, self.target]].dropna()
        contingency = pd.crosstab(test_data[feature], test_data[self.target])

        chi2, p_value, dof, expected = chi2_contingency(contingency)
        alpha = 0.05
        significant = p_value < alpha
        decision = "Reject H0" if significant else "Fail to Reject H0"

        print(f"  Chi-Square Statistic : {chi2:.4f}")
        print(f"  Degrees of Freedom   : {dof}")
        print(f"  P-Value              : {p_value:.6f}")
        print(f"  Decision (α=0.05)    : {decision}")
        print(f"  → {'Agreement type IS associated with churn.' if significant else 'No significant association found.'}\n")

        # Visualise
        fig, ax = plt.subplots(figsize=(10, 5))
        ct_pct = pd.crosstab(test_data[feature], test_data[self.target], normalize='index') * 100
        ct_pct.plot(kind='barh', stacked=True, ax=ax, colormap='Set2', edgecolor='white')
        ax.set_title('H1: Agreement Type vs Churn Category (%)', fontsize=13, fontweight='bold')
        ax.set_xlabel('Percentage (%)')
        ax.legend(title='Churn Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

        return [{'Hypothesis': 'H1', 'Feature': feature, 'Test': 'Chi-Square',
                 'Statistic': round(chi2, 4), 'P-Value': round(p_value, 6),
                 'Decision': decision}]

    # ────────────────────────────────────────────────────────────────
    #  H2: Kruskal-Wallis – Revenue across Churn Categories
    # ────────────────────────────────────────────────────────────────
    def hypothesis_2_revenue(self):
        """
        H2: Revenue does not differ significantly across churn categories.
        Test: Kruskal-Wallis H-test (non-parametric, no normality required)
        """
        print("=" * 65)
        print("  HYPOTHESIS 2: Revenue vs Churn (Kruskal-Wallis)")
        print("=" * 65)
        print("H0: Revenue distribution is the same across churn categories")
        print("H1: At least one churn category has a different revenue distribution\n")

        feature = 'total_revenue'
        return self._run_kruskal(feature, 'H2')

    # ────────────────────────────────────────────────────────────────
    #  H3: Kruskal-Wallis – Number of Agreements across Churn
    # ────────────────────────────────────────────────────────────────
    def hypothesis_3_agreements(self):
        """
        H3: Number of agreements does not differ across churn categories.
        Test: Kruskal-Wallis H-test
        """
        print("=" * 65)
        print("  HYPOTHESIS 3: Number of Agreements vs Churn (Kruskal-Wallis)")
        print("=" * 65)
        print("H0: Number of agreements is the same across churn categories")
        print("H1: At least one category has a different number of agreements\n")

        feature = 'num_agreements'
        return self._run_kruskal(feature, 'H3')

    # ────────────────────────────────────────────────────────────────
    #  H4: Kruskal-Wallis – Fee Structure across Churn
    # ────────────────────────────────────────────────────────────────
    def hypothesis_4_fees(self):
        """
        H4: Fee structure does not differ across churn categories.
        Test: Kruskal-Wallis H-test
        """
        print("=" * 65)
        print("  HYPOTHESIS 4: Fees vs Churn (Kruskal-Wallis)")
        print("=" * 65)
        print("H0: Total fees distribution is the same across churn categories")
        print("H1: At least one churn category has different fee distribution\n")

        feature = 'total_fees'
        return self._run_kruskal(feature, 'H4')

    # ────────────────────────────────────────────────────────────────
    #  H5: Kruskal-Wallis – Agreement Duration across Churn
    # ────────────────────────────────────────────────────────────────
    def hypothesis_5_duration(self):
        """
        H5: Agreement duration has no relationship with churn.
        Test: Kruskal-Wallis H-test
        """
        print("=" * 65)
        print("  HYPOTHESIS 5: Agreement Duration vs Churn (Kruskal-Wallis)")
        print("=" * 65)
        print("H0: Agreement duration is the same across churn categories")
        print("H1: At least one churn category has different agreement duration\n")

        feature = 'avg_agreement_duration'
        return self._run_kruskal(feature, 'H5')

    # ────────────────────────────────────────────────────────────────
    #  H6: Spearman Correlation – Financial Metrics vs Churn
    # ────────────────────────────────────────────────────────────────
    def hypothesis_6_correlation(self):
        """
        H6: Financial metrics have no monotonic correlation with churn severity.
        Test: Spearman Rank Correlation
        """
        print("=" * 65)
        print("  HYPOTHESIS 6: Financial Correlation with Churn (Spearman)")
        print("=" * 65)
        print("H0: No monotonic correlation between features and churn severity")
        print("H1: Significant correlation exists\n")

        churn_numeric = self.data[self.target].map({
            'no_churn': 0, 'partial_churn': 1, 'full_churn': 2
        })

        features = ['total_revenue', 'avg_revenue', 'total_fees', 'avg_unit_amount',
                     'num_agreements', 'avg_agreement_duration', 'num_retention_cases',
                     'bob_ratio']

        results = []
        for feature in features:
            if feature not in self.data.columns:
                continue
            test_data = pd.DataFrame({'feature': self.data[feature], 'churn': churn_numeric}).dropna()
            if len(test_data) > 0:
                corr, p_val = spearmanr(test_data['feature'], test_data['churn'])
                significant = p_val < 0.05
                decision = 'Reject H0' if significant else 'Fail to Reject H0'
                results.append({
                    'Hypothesis': 'H6', 'Feature': feature,
                    'Test': 'Spearman', 'Statistic': round(corr, 4),
                    'P-Value': round(p_val, 6), 'Decision': decision
                })
                print(f"  {feature:>25s}:  ρ = {corr:+.4f},  p = {p_val:.6f}  → {decision}")

        # Visualise
        if results:
            corr_df = pd.DataFrame(results)
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ['#e74c3c' if r['Decision'] == 'Reject H0' else '#95a5a6' for r in results]
            ax.barh(corr_df['Feature'], corr_df['Statistic'], color=colors, edgecolor='black', linewidth=0.5)
            ax.axvline(x=0, color='black', linewidth=0.8)
            ax.set_xlabel('Spearman Correlation (ρ)', fontsize=12)
            ax.set_title('H6: Feature Correlations with Churn Severity', fontsize=13, fontweight='bold')
            plt.tight_layout()
            plt.show()

        print()
        return results

    # ────────────────────────────────────────────────────────────────
    #  Run All Hypotheses
    # ────────────────────────────────────────────────────────────────
    def run_all(self):
        """Run all 6 hypothesis tests and return consolidated results."""
        all_results = []
        all_results.extend(self.hypothesis_1_chi_square())
        all_results.extend(self.hypothesis_2_revenue())
        all_results.extend(self.hypothesis_3_agreements())
        all_results.extend(self.hypothesis_4_fees())
        all_results.extend(self.hypothesis_5_duration())
        all_results.extend(self.hypothesis_6_correlation())

        print("\n" + "=" * 65)
        print("  HYPOTHESIS TESTING SUMMARY")
        print("=" * 65)
        summary_df = pd.DataFrame(all_results)
        print(summary_df.to_string(index=False))
        return summary_df

    # ────────────────────────────────────────────────────────────────
    #  Helper: Kruskal-Wallis test with box plot
    # ────────────────────────────────────────────────────────────────
    def _run_kruskal(self, feature, hypothesis_id):
        """Run Kruskal-Wallis test for a numeric feature and create a box plot."""
        if feature not in self.data.columns:
            print(f"  Column '{feature}' not found. Skipping.")
            return []

        test_data = self.data[[feature, self.target]].dropna()
        groups = [grp[feature].values for _, grp in test_data.groupby(self.target)]

        if len(groups) < 2:
            print("  Not enough groups for test. Skipping.")
            return []

        h_stat, p_value = kruskal(*groups)
        alpha = 0.05
        significant = p_value < alpha
        decision = "Reject H0" if significant else "Fail to Reject H0"

        print(f"  Feature             : {feature}")
        print(f"  H-Statistic         : {h_stat:.4f}")
        print(f"  P-Value             : {p_value:.6f}")
        print(f"  Decision (α=0.05)   : {decision}")

        # Group means for interpretation
        group_means = test_data.groupby(self.target)[feature].mean()
        for cat, mean_val in group_means.items():
            print(f"    Mean ({cat:>15s}): {mean_val:.2f}")
        print()

        # Box plot
        fig, ax = plt.subplots(figsize=(8, 5))
        order = ['no_churn', 'partial_churn', 'full_churn']
        order = [o for o in order if o in test_data[self.target].unique()]
        sns.boxplot(data=test_data, x=self.target, y=feature, ax=ax,
                    palette='Set2', order=order, showfliers=False)
        ax.set_title(f'{hypothesis_id}: {feature} by Churn Category', fontsize=13, fontweight='bold')
        ax.set_xlabel('Churn Category')
        ax.set_ylabel(feature)
        plt.tight_layout()
        plt.show()

        return [{'Hypothesis': hypothesis_id, 'Feature': feature,
                 'Test': 'Kruskal-Wallis', 'Statistic': round(h_stat, 4),
                 'P-Value': round(p_value, 6), 'Decision': decision}]
