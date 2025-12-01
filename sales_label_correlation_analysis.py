"""
Comprehensive Statistical Analysis: Sales Label Correlations
Analysis of correlations between PnL forward points and driver variables
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class SalesLabelCorrelationAnalysis:
    """Comprehensive correlation analysis for sales labels and PnL"""

    def __init__(self, csv_path):
        """Initialize with data file path"""
        self.csv_path = csv_path
        self.df = None
        self.pnl_columns = []
        self.driver_columns = []
        self.results = {}

    def load_data(self):
        """Load and prepare dataset"""
        print("=" * 80)
        print("LOADING DATASET")
        print("=" * 80)

        self.df = pd.read_csv(self.csv_path)
        print(f"\n✓ Dataset loaded successfully")
        print(f"  Shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")

        # Identify PnL columns (PnL_fwd_pts_*_mediana)
        self.pnl_columns = [col for col in self.df.columns if 'PnL_fwd_pts_' in col and '_mediana' in col]
        # Sort by the window number (01, 05, 25, 50, 90)
        def extract_window(col):
            parts = col.split('_')
            for i, part in enumerate(parts):
                if part == 'pts' and i + 1 < len(parts):
                    return int(parts[i + 1])
            return 0
        self.pnl_columns = sorted(self.pnl_columns, key=extract_window)

        # Driver columns
        self.driver_columns = ['LABEL_GENERAL_SCORE', 'BQI_ABS', 'FF_ATM',
                               'delta_total', 'theta_total', 'FF_BAT']

        # Filter to available columns
        self.driver_columns = [col for col in self.driver_columns if col in self.df.columns]

        print(f"\n  PnL Variables ({len(self.pnl_columns)}):")
        for col in self.pnl_columns:
            print(f"    - {col}")

        print(f"\n  Driver Variables ({len(self.driver_columns)}):")
        for col in self.driver_columns:
            print(f"    - {col}")

        # Remove rows with all NaN in drivers or PnL
        initial_rows = len(self.df)
        self.df = self.df.dropna(subset=self.driver_columns + self.pnl_columns, how='all')
        final_rows = len(self.df)

        print(f"\n  Rows after cleaning: {final_rows:,} (removed {initial_rows - final_rows:,} empty rows)")

        return self

    def descriptive_statistics(self):
        """Generate descriptive statistics for all variables"""
        print("\n" + "=" * 80)
        print("1. DESCRIPTIVE STATISTICS")
        print("=" * 80)

        # Drivers statistics
        print("\n>>> DRIVER VARIABLES <<<\n")
        driver_stats = self.df[self.driver_columns].describe().T
        driver_stats['count'] = driver_stats['count'].astype(int)
        print(driver_stats.to_string())

        # PnL statistics
        print("\n>>> PNL FORWARD POINTS VARIABLES <<<\n")
        pnl_stats = self.df[self.pnl_columns].describe().T
        pnl_stats['count'] = pnl_stats['count'].astype(int)
        print(pnl_stats.to_string())

        self.results['descriptive_stats'] = {
            'drivers': driver_stats,
            'pnl': pnl_stats
        }

        return self

    def correlation_analysis(self):
        """Calculate Pearson and Spearman correlations with p-values"""
        print("\n" + "=" * 80)
        print("2. CORRELATION ANALYSIS WITH PNL")
        print("=" * 80)

        # Pearson correlation
        pearson_corr = pd.DataFrame(index=self.driver_columns, columns=self.pnl_columns)
        pearson_pval = pd.DataFrame(index=self.driver_columns, columns=self.pnl_columns)

        # Spearman correlation
        spearman_corr = pd.DataFrame(index=self.driver_columns, columns=self.pnl_columns)
        spearman_pval = pd.DataFrame(index=self.driver_columns, columns=self.pnl_columns)

        for driver in self.driver_columns:
            for pnl in self.pnl_columns:
                # Remove NaN pairs
                mask = self.df[[driver, pnl]].notna().all(axis=1)
                x = self.df.loc[mask, driver]
                y = self.df.loc[mask, pnl]

                if len(x) > 2:
                    # Pearson
                    p_corr, p_pval = stats.pearsonr(x, y)
                    pearson_corr.loc[driver, pnl] = p_corr
                    pearson_pval.loc[driver, pnl] = p_pval

                    # Spearman
                    s_corr, s_pval = stats.spearmanr(x, y)
                    spearman_corr.loc[driver, pnl] = s_corr
                    spearman_pval.loc[driver, pnl] = s_pval

        # Convert to numeric
        pearson_corr = pearson_corr.astype(float)
        pearson_pval = pearson_pval.astype(float)
        spearman_corr = spearman_corr.astype(float)
        spearman_pval = spearman_pval.astype(float)

        print("\n>>> PEARSON CORRELATION (Linear) <<<\n")
        print(pearson_corr.to_string())

        print("\n>>> PEARSON P-VALUES <<<\n")
        print(pearson_pval.to_string())

        print("\n>>> SPEARMAN CORRELATION (Monotonic) <<<\n")
        print(spearman_corr.to_string())

        print("\n>>> SPEARMAN P-VALUES <<<\n")
        print(spearman_pval.to_string())

        self.results['correlations'] = {
            'pearson_corr': pearson_corr,
            'pearson_pval': pearson_pval,
            'spearman_corr': spearman_corr,
            'spearman_pval': spearman_pval
        }

        return self

    def driver_ranking(self):
        """Rank drivers by average absolute correlation"""
        print("\n" + "=" * 80)
        print("3. DRIVER RANKING BY PREDICTIVE POWER")
        print("=" * 80)

        pearson_corr = self.results['correlations']['pearson_corr']
        spearman_corr = self.results['correlations']['spearman_corr']

        ranking_data = []

        for driver in self.driver_columns:
            pearson_abs_mean = pearson_corr.loc[driver].abs().mean()
            spearman_abs_mean = spearman_corr.loc[driver].abs().mean()
            avg_abs_corr = (pearson_abs_mean + spearman_abs_mean) / 2
            n_windows = len(self.pnl_columns)

            ranking_data.append({
                'Driver': driver,
                'Avg_Abs_Correlation': avg_abs_corr,
                'Pearson_Avg': pearson_abs_mean,
                'Spearman_Avg': spearman_abs_mean,
                'N_Windows': n_windows
            })

        ranking_df = pd.DataFrame(ranking_data)
        ranking_df = ranking_df.sort_values('Avg_Abs_Correlation', ascending=False)
        ranking_df['Rank'] = range(1, len(ranking_df) + 1)
        ranking_df = ranking_df[['Rank', 'Driver', 'Avg_Abs_Correlation', 'Pearson_Avg', 'Spearman_Avg', 'N_Windows']]

        print("\n>>> DRIVER RANKING <<<\n")
        print(ranking_df.to_string(index=False))

        self.results['ranking'] = ranking_df
        self.best_driver = ranking_df.iloc[0]['Driver']

        print(f"\n🏆 BEST DRIVER: {self.best_driver}")
        print(f"   Average Absolute Correlation: {ranking_df.iloc[0]['Avg_Abs_Correlation']:.4f}")

        return self

    def range_analysis_best_driver(self):
        """Analyze PnL by percentile ranges for best driver"""
        print("\n" + "=" * 80)
        print(f"4. RANGE ANALYSIS: {self.best_driver}")
        print("=" * 80)

        driver = self.best_driver
        percentiles = [25, 50, 75, 90]

        range_results = []

        for percentile in percentiles:
            threshold = self.df[driver].quantile(percentile / 100)

            print(f"\n>>> PERCENTILE {percentile} (Threshold: {threshold:.4f}) <<<\n")

            above_mask = self.df[driver] >= threshold
            below_mask = self.df[driver] < threshold

            n_above = above_mask.sum()
            n_below = below_mask.sum()

            print(f"  Above threshold: {n_above:,} trades")
            print(f"  Below threshold: {n_below:,} trades")

            for pnl in self.pnl_columns:
                pnl_above = self.df.loc[above_mask, pnl].mean()
                pnl_below = self.df.loc[below_mask, pnl].mean()
                differential = pnl_above - pnl_below
                winner = "ABOVE" if differential > 0 else "BELOW"

                range_results.append({
                    'Percentile': percentile,
                    'PnL_Variable': pnl,
                    'N_Above': n_above,
                    'N_Below': n_below,
                    'Mean_PnL_Above': pnl_above,
                    'Mean_PnL_Below': pnl_below,
                    'Differential': differential,
                    'Winner': winner
                })

                print(f"  {pnl}:")
                print(f"    Above: {pnl_above:>10.4f} | Below: {pnl_below:>10.4f} | Diff: {differential:>10.4f} | Winner: {winner}")

        range_df = pd.DataFrame(range_results)
        self.results['range_analysis'] = range_df

        return self

    def quartile_analysis(self):
        """Analyze PnL by quartiles for all drivers"""
        print("\n" + "=" * 80)
        print("5. QUARTILE ANALYSIS (ALL DRIVERS)")
        print("=" * 80)

        quartile_results = []

        for driver in self.driver_columns:
            print(f"\n>>> DRIVER: {driver} <<<\n")

            # Calculate quartiles
            q1 = self.df[driver].quantile(0.25)
            q2 = self.df[driver].quantile(0.50)
            q3 = self.df[driver].quantile(0.75)

            # Assign quartiles
            quartile_labels = pd.cut(self.df[driver],
                                    bins=[-np.inf, q1, q2, q3, np.inf],
                                    labels=['Q1 (Bottom 25%)', 'Q2', 'Q3', 'Q4 (Top 25%)'])

            for pnl in self.pnl_columns:
                for q_label in ['Q1 (Bottom 25%)', 'Q2', 'Q3', 'Q4 (Top 25%)']:
                    mask = quartile_labels == q_label
                    n_trades = mask.sum()
                    mean_pnl = self.df.loc[mask, pnl].mean()
                    std_pnl = self.df.loc[mask, pnl].std()

                    quartile_results.append({
                        'Driver': driver,
                        'PnL_Variable': pnl,
                        'Quartile': q_label,
                        'N_Trades': n_trades,
                        'Mean_PnL': mean_pnl,
                        'Std_PnL': std_pnl
                    })

            # Print summary for this driver
            driver_summary = pd.DataFrame(quartile_results)
            driver_summary = driver_summary[driver_summary['Driver'] == driver]
            pivot = driver_summary.pivot_table(
                index='Quartile',
                columns='PnL_Variable',
                values='Mean_PnL'
            )
            pivot = pivot.reindex(['Q1 (Bottom 25%)', 'Q2', 'Q3', 'Q4 (Top 25%)'])
            print(pivot.to_string())

        quartile_df = pd.DataFrame(quartile_results)
        self.results['quartile_analysis'] = quartile_df

        return self

    def top_bottom_analysis(self):
        """Analyze Top 10% vs Bottom 10% for each driver"""
        print("\n" + "=" * 80)
        print("6. TOP 10% vs BOTTOM 10% ANALYSIS")
        print("=" * 80)

        top_bottom_results = []

        for driver in self.driver_columns:
            print(f"\n>>> DRIVER: {driver} <<<\n")

            p10 = self.df[driver].quantile(0.10)
            p90 = self.df[driver].quantile(0.90)

            bottom_mask = self.df[driver] <= p10
            top_mask = self.df[driver] >= p90

            n_bottom = bottom_mask.sum()
            n_top = top_mask.sum()

            print(f"  Bottom 10% (≤ {p10:.4f}): {n_bottom:,} trades")
            print(f"  Top 10% (≥ {p90:.4f}): {n_top:,} trades")
            print()

            for pnl in self.pnl_columns:
                mean_bottom = self.df.loc[bottom_mask, pnl].mean()
                mean_top = self.df.loc[top_mask, pnl].mean()
                spread = mean_top - mean_bottom
                direction = "POSITIVE (Expected)" if spread > 0 else "NEGATIVE (Inverse)"

                top_bottom_results.append({
                    'Driver': driver,
                    'PnL_Variable': pnl,
                    'N_Bottom': n_bottom,
                    'N_Top': n_top,
                    'Mean_PnL_Bottom': mean_bottom,
                    'Mean_PnL_Top': mean_top,
                    'Spread': spread,
                    'Direction': direction
                })

                print(f"  {pnl}:")
                print(f"    Bottom 10%: {mean_bottom:>10.4f} | Top 10%: {mean_top:>10.4f}")
                print(f"    SPREAD: {spread:>10.4f} ({direction})")

        top_bottom_df = pd.DataFrame(top_bottom_results)
        self.results['top_bottom_analysis'] = top_bottom_df

        return self

    def extreme_scenarios(self):
        """Analyze extreme scenarios for best driver"""
        print("\n" + "=" * 80)
        print(f"7. EXTREME SCENARIOS: {self.best_driver}")
        print("=" * 80)

        driver = self.best_driver
        percentiles = [75, 85, 95]

        extreme_results = []

        for percentile in percentiles:
            threshold = self.df[driver].quantile(percentile / 100)
            mask = self.df[driver] >= threshold
            n_trades = mask.sum()
            retention_pct = (n_trades / len(self.df)) * 100

            print(f"\n>>> PERCENTILE {percentile} (Threshold: {threshold:.4f}) <<<")
            print(f"  Trades retained: {n_trades:,} ({retention_pct:.2f}%)")
            print()

            for pnl in self.pnl_columns:
                mean_pnl = self.df.loc[mask, pnl].mean()
                std_pnl = self.df.loc[mask, pnl].std()

                extreme_results.append({
                    'Percentile': percentile,
                    'Threshold': threshold,
                    'N_Trades': n_trades,
                    'Retention_%': retention_pct,
                    'PnL_Variable': pnl,
                    'Mean_PnL': mean_pnl,
                    'Std_PnL': std_pnl,
                    'Lower_Bound': mean_pnl - std_pnl,
                    'Upper_Bound': mean_pnl + std_pnl
                })

                print(f"  {pnl}:")
                print(f"    Mean: {mean_pnl:>10.4f} ± {std_pnl:.4f}")
                print(f"    Range: [{mean_pnl - std_pnl:>10.4f}, {mean_pnl + std_pnl:>10.4f}]")

        extreme_df = pd.DataFrame(extreme_results)
        self.results['extreme_scenarios'] = extreme_df

        return self

    def filter_recommendations(self):
        """Generate filter recommendations"""
        print("\n" + "=" * 80)
        print(f"8. FILTER RECOMMENDATIONS: {self.best_driver}")
        print("=" * 80)

        driver = self.best_driver
        filter_configs = [
            ('Conservative', 75),
            ('Balanced', 90),
            ('Aggressive', 95)
        ]

        filter_results = []

        for filter_name, percentile in filter_configs:
            threshold = self.df[driver].quantile(percentile / 100)
            mask = self.df[driver] >= threshold
            n_trades = mask.sum()
            retention_pct = (n_trades / len(self.df)) * 100

            print(f"\n>>> {filter_name.upper()} FILTER (P{percentile}) <<<")
            print(f"  Threshold: {driver} ≥ {threshold:.4f}")
            print(f"  Retention: {n_trades:,} trades ({retention_pct:.2f}%)")
            print(f"\n  Expected PnL:")

            for pnl in self.pnl_columns:
                mean_pnl = self.df.loc[mask, pnl].mean()

                filter_results.append({
                    'Filter_Type': filter_name,
                    'Percentile': percentile,
                    'Threshold': threshold,
                    'N_Trades': n_trades,
                    'Retention_%': retention_pct,
                    'PnL_Variable': pnl,
                    'Expected_PnL': mean_pnl
                })

                print(f"    {pnl}: {mean_pnl:>10.4f}")

        # Anti-filters (low performance zones)
        print("\n>>> ANTI-FILTERS (ZONES TO AVOID) <<<")

        p25 = self.df[driver].quantile(0.25)
        mask_low = self.df[driver] <= p25
        n_low = mask_low.sum()

        print(f"\n  🚫 LOW ZONE: {driver} ≤ {p25:.4f}")
        print(f"     Affects: {n_low:,} trades ({(n_low/len(self.df)*100):.2f}%)")
        print(f"     Expected PnL (typically lower):")

        for pnl in self.pnl_columns:
            mean_pnl_low = self.df.loc[mask_low, pnl].mean()
            print(f"       {pnl}: {mean_pnl_low:>10.4f}")

        filter_df = pd.DataFrame(filter_results)
        self.results['filter_recommendations'] = filter_df

        return self

    def generate_visualizations(self):
        """Generate all visualizations"""
        print("\n" + "=" * 80)
        print("9. GENERATING VISUALIZATIONS")
        print("=" * 80)

        output_dir = Path('analysis_output')
        output_dir.mkdir(exist_ok=True)

        # 1. Correlation Heatmap (Pearson)
        print("\n  [1/6] Correlation Heatmap (Pearson)...")
        fig, ax = plt.subplots(figsize=(14, 8))
        pearson_corr = self.results['correlations']['pearson_corr']

        # Simplify column names for visualization
        simplified_cols = [col.replace('PNL_FWD_PTS_', '').replace('_mediana', '') for col in self.pnl_columns]
        pearson_display = pearson_corr.copy()
        pearson_display.columns = simplified_cols

        sns.heatmap(pearson_display.astype(float), annot=True, fmt='.3f',
                   cmap='RdYlGn', center=0, vmin=-1, vmax=1,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Pearson Correlation: Drivers vs PnL Forward Points',
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('PnL Windows', fontsize=12, fontweight='bold')
        plt.ylabel('Driver Variables', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / '01_correlation_heatmap_pearson.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Correlation Heatmap (Spearman)
        print("  [2/6] Correlation Heatmap (Spearman)...")
        fig, ax = plt.subplots(figsize=(14, 8))
        spearman_corr = self.results['correlations']['spearman_corr']
        spearman_display = spearman_corr.copy()
        spearman_display.columns = simplified_cols

        sns.heatmap(spearman_display.astype(float), annot=True, fmt='.3f',
                   cmap='RdYlGn', center=0, vmin=-1, vmax=1,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Spearman Correlation: Drivers vs PnL Forward Points',
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('PnL Windows', fontsize=12, fontweight='bold')
        plt.ylabel('Driver Variables', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / '02_correlation_heatmap_spearman.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Driver Ranking Bar Chart
        print("  [3/6] Driver Ranking...")
        fig, ax = plt.subplots(figsize=(12, 8))
        ranking = self.results['ranking'].copy()

        bars = ax.barh(ranking['Driver'], ranking['Avg_Abs_Correlation'],
                      color=sns.color_palette("husl", len(ranking)))
        ax.set_xlabel('Average Absolute Correlation', fontsize=12, fontweight='bold')
        ax.set_ylabel('Driver Variable', fontsize=12, fontweight='bold')
        ax.set_title('Driver Ranking by Predictive Power', fontsize=16, fontweight='bold', pad=20)
        ax.invert_yaxis()

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, ranking['Avg_Abs_Correlation'])):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{val:.4f}', va='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_dir / '03_driver_ranking.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Quartile Analysis for Best Driver
        print("  [4/6] Quartile Analysis (Best Driver)...")
        fig, ax = plt.subplots(figsize=(14, 8))

        quartile_data = self.results['quartile_analysis']
        best_driver_data = quartile_data[quartile_data['Driver'] == self.best_driver]

        pivot = best_driver_data.pivot(index='Quartile', columns='PnL_Variable', values='Mean_PnL')
        pivot = pivot.reindex(['Q1 (Bottom 25%)', 'Q2', 'Q3', 'Q4 (Top 25%)'])
        pivot.columns = simplified_cols

        x = np.arange(len(pivot.index))
        width = 0.15
        multiplier = 0

        for i, pnl in enumerate(pivot.columns):
            offset = width * multiplier
            bars = ax.bar(x + offset, pivot[pnl], width, label=pnl)
            multiplier += 1

        ax.set_xlabel('Quartile', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean PnL', fontsize=12, fontweight='bold')
        ax.set_title(f'Quartile Analysis: {self.best_driver}', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x + width * (len(pivot.columns) - 1) / 2)
        ax.set_xticklabels(pivot.index, rotation=0)
        ax.legend(loc='best', fontsize=9)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / '04_quartile_analysis_best_driver.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 5. Top 10% vs Bottom 10% Spread
        print("  [5/6] Top vs Bottom 10% Analysis...")
        fig, ax = plt.subplots(figsize=(14, 10))

        top_bottom = self.results['top_bottom_analysis']

        n_drivers = len(self.driver_columns)
        n_pnl = len(self.pnl_columns)

        x = np.arange(n_pnl)
        width = 0.12

        for i, driver in enumerate(self.driver_columns):
            driver_data = top_bottom[top_bottom['Driver'] == driver]
            spreads = driver_data['Spread'].values
            offset = width * (i - n_drivers/2 + 0.5)
            ax.bar(x + offset, spreads, width, label=driver)

        ax.set_xlabel('PnL Window', fontsize=12, fontweight='bold')
        ax.set_ylabel('Spread (Top 10% - Bottom 10%)', fontsize=12, fontweight='bold')
        ax.set_title('Top 10% vs Bottom 10% Spread by Driver', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(simplified_cols, rotation=45, ha='right')
        ax.legend(loc='best', fontsize=9)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / '05_top_bottom_spread.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 6. Filter Recommendations Performance
        print("  [6/6] Filter Recommendations...")
        fig, ax = plt.subplots(figsize=(14, 8))

        filter_data = self.results['filter_recommendations']

        filter_types = filter_data['Filter_Type'].unique()
        x = np.arange(len(simplified_cols))
        width = 0.25

        for i, filter_type in enumerate(filter_types):
            filter_subset = filter_data[filter_data['Filter_Type'] == filter_type]
            expected_pnls = filter_subset['Expected_PnL'].values
            offset = width * (i - 1)
            bars = ax.bar(x + offset, expected_pnls, width, label=filter_type)

        ax.set_xlabel('PnL Window', fontsize=12, fontweight='bold')
        ax.set_ylabel('Expected PnL', fontsize=12, fontweight='bold')
        ax.set_title(f'Filter Recommendations: {self.best_driver}',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(simplified_cols, rotation=45, ha='right')
        ax.legend(loc='best', fontsize=10)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / '06_filter_recommendations.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n✓ All visualizations saved to: {output_dir}/")

        return self

    def generate_summary_report(self):
        """Generate executive summary and conclusions"""
        print("\n" + "=" * 80)
        print("10. EXECUTIVE SUMMARY AND CONCLUSIONS")
        print("=" * 80)

        output_dir = Path('analysis_output')
        report_file = output_dir / 'EXECUTIVE_SUMMARY.txt'

        with open(report_file, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("EXECUTIVE SUMMARY: SALES LABEL CORRELATION ANALYSIS\n")
            f.write("=" * 80 + "\n\n")

            # Dataset overview
            f.write("1. DATASET OVERVIEW\n")
            f.write("-" * 80 + "\n")
            f.write(f"   Total Observations: {len(self.df):,}\n")
            f.write(f"   PnL Variables: {len(self.pnl_columns)}\n")
            f.write(f"   Driver Variables: {len(self.driver_columns)}\n\n")

            # Top 3 drivers
            f.write("2. TOP 3 DRIVERS BY PREDICTIVE POWER\n")
            f.write("-" * 80 + "\n")
            ranking = self.results['ranking'].head(3)
            for idx, row in ranking.iterrows():
                f.write(f"   #{int(row['Rank'])}. {row['Driver']}\n")
                f.write(f"       Average Absolute Correlation: {row['Avg_Abs_Correlation']:.4f}\n")
                f.write(f"       Pearson Avg: {row['Pearson_Avg']:.4f} | Spearman Avg: {row['Spearman_Avg']:.4f}\n\n")

            # Key findings
            f.write("3. KEY FINDINGS\n")
            f.write("-" * 80 + "\n")

            best_driver = self.results['ranking'].iloc[0]['Driver']
            best_corr = self.results['ranking'].iloc[0]['Avg_Abs_Correlation']

            f.write(f"   ✓ BEST DRIVER: {best_driver}\n")
            f.write(f"     Shows strongest correlation with PnL (avg: {best_corr:.4f})\n\n")

            # Analyze top vs bottom performance
            top_bottom = self.results['top_bottom_analysis']
            best_driver_tb = top_bottom[top_bottom['Driver'] == best_driver]
            avg_spread = best_driver_tb['Spread'].mean()

            f.write(f"   ✓ TOP 10% vs BOTTOM 10% PERFORMANCE:\n")
            f.write(f"     Average Spread: {avg_spread:.4f}\n")
            if avg_spread > 0:
                f.write(f"     Direction: POSITIVE (Higher {best_driver} → Higher PnL) ✓\n\n")
            else:
                f.write(f"     Direction: NEGATIVE (Inverse relationship) ⚠\n\n")

            # Paradoxes and anomalies
            f.write("   ✓ PARADOXES & ANOMALIES:\n")

            # Check for negative correlations in top drivers
            pearson_corr = self.results['correlations']['pearson_corr']
            for driver in self.driver_columns[:3]:  # Top 3
                avg_corr = pearson_corr.loc[driver].mean()
                if avg_corr < 0:
                    f.write(f"     ⚠ {driver} shows NEGATIVE average correlation ({avg_corr:.4f})\n")

            # Check for inconsistent performance
            for driver in self.driver_columns:
                driver_corrs = pearson_corr.loc[driver].values
                if (driver_corrs > 0).any() and (driver_corrs < 0).any():
                    f.write(f"     ⚠ {driver} shows MIXED correlations across windows\n")

            f.write("\n")

            # Filter recommendations
            f.write("4. FILTER RECOMMENDATIONS\n")
            f.write("-" * 80 + "\n\n")

            filter_recs = self.results['filter_recommendations']

            for filter_type in ['Conservative', 'Balanced', 'Aggressive']:
                filter_subset = filter_recs[filter_recs['Filter_Type'] == filter_type]
                if len(filter_subset) > 0:
                    row = filter_subset.iloc[0]
                    avg_pnl = filter_subset['Expected_PnL'].mean()

                    f.write(f"   {filter_type.upper()} FILTER (P{int(row['Percentile'])})\n")
                    f.write(f"   Threshold: {best_driver} ≥ {row['Threshold']:.4f}\n")
                    f.write(f"   Retention: {row['N_Trades']:,.0f} trades ({row['Retention_%']:.2f}%)\n")
                    f.write(f"   Expected Avg PnL: {avg_pnl:.4f}\n\n")

            # Final recommendations
            f.write("5. FINAL RECOMMENDATIONS\n")
            f.write("-" * 80 + "\n\n")

            f.write(f"   1. PRIMARY FILTER: Use {best_driver} as main selection criterion\n")
            f.write(f"      → Recommend BALANCED filter (P90) for optimal risk/return\n\n")

            f.write(f"   2. SECONDARY FILTERS: Consider combining with:\n")
            ranking = self.results['ranking']
            if len(ranking) > 1:
                second_best = ranking.iloc[1]['Driver']
                f.write(f"      → {second_best} (Rank #2)\n")
            if len(ranking) > 2:
                third_best = ranking.iloc[2]['Driver']
                f.write(f"      → {third_best} (Rank #3)\n")
            f.write("\n")

            f.write(f"   3. ANTI-FILTERS: Avoid trades where:\n")
            p25 = self.df[best_driver].quantile(0.25)
            f.write(f"      → {best_driver} < {p25:.4f} (Bottom 25%)\n\n")

            f.write(f"   4. MONITORING: Track correlation stability over time\n")
            f.write(f"      → Relationships may evolve with market conditions\n\n")

            # Footer
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        print(f"\n✓ Executive summary saved to: {report_file}")

        # Print summary to console
        with open(report_file, 'r', encoding='utf-8') as f:
            print("\n" + f.read())

        return self

    def export_results(self):
        """Export all results to CSV files"""
        print("\n" + "=" * 80)
        print("11. EXPORTING RESULTS TO CSV")
        print("=" * 80)

        output_dir = Path('analysis_output')

        # Export each result dataframe
        exports = [
            ('descriptive_stats_drivers.csv', self.results['descriptive_stats']['drivers']),
            ('descriptive_stats_pnl.csv', self.results['descriptive_stats']['pnl']),
            ('correlations_pearson.csv', self.results['correlations']['pearson_corr']),
            ('correlations_pearson_pvalues.csv', self.results['correlations']['pearson_pval']),
            ('correlations_spearman.csv', self.results['correlations']['spearman_corr']),
            ('correlations_spearman_pvalues.csv', self.results['correlations']['spearman_pval']),
            ('driver_ranking.csv', self.results['ranking']),
            ('range_analysis.csv', self.results['range_analysis']),
            ('quartile_analysis.csv', self.results['quartile_analysis']),
            ('top_bottom_analysis.csv', self.results['top_bottom_analysis']),
            ('extreme_scenarios.csv', self.results['extreme_scenarios']),
            ('filter_recommendations.csv', self.results['filter_recommendations'])
        ]

        for filename, df in exports:
            filepath = output_dir / filename
            df.to_csv(filepath, index=True)
            print(f"  ✓ {filename}")

        print(f"\n✓ All results exported to: {output_dir}/")

        return self

    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("\n" + "🔬" * 40)
        print("COMPREHENSIVE STATISTICAL ANALYSIS")
        print("Sales Label Correlations with PnL Forward Points")
        print("🔬" * 40 + "\n")

        try:
            (self.load_data()
             .descriptive_statistics()
             .correlation_analysis()
             .driver_ranking()
             .range_analysis_best_driver()
             .quartile_analysis()
             .top_bottom_analysis()
             .extreme_scenarios()
             .filter_recommendations()
             .generate_visualizations()
             .generate_summary_report()
             .export_results())

            print("\n" + "=" * 80)
            print("✓ ANALYSIS COMPLETE!")
            print("=" * 80)
            print(f"\nAll results saved to: analysis_output/")
            print("  - 6 visualizations (PNG, 300 DPI)")
            print("  - 12 CSV data files")
            print("  - 1 executive summary report")

        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # Run analysis
    analysis = SalesLabelCorrelationAnalysis('combined_mediana_labeled.csv')
    analysis.run_complete_analysis()
