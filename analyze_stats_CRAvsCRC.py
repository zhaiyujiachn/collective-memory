import json
import numpy as np
from scipy import stats

# --- Configuration ---
CRA_FILE = 'final_code_data/DBLP/cra_list.json'
CRC_FILE = 'final_code_data/DBLP/crc_list.json'
ALPHA = 0.05 # Significance level

# --- Load Data ---
try:
    with open(CRA_FILE, 'r') as f:
        cra_list_from_json = json.load(f)
    with open(CRC_FILE, 'r') as f:
        crc_list_from_json = json.load(f)
except FileNotFoundError:
    print(f"Error: Make sure '{CRA_FILE}' and '{CRC_FILE}' are in the correct directory.")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from one or both files. Check file format.")
    exit()

# Convert lists to NumPy arrays for efficient calculation
# Shape will be (num_papers, num_quantiles), e.g., (100, 10)
cra_data = np.array(cra_list_from_json)
crc_data = np.array(crc_list_from_json)

# --- Data Validation (Basic Checks) ---
if cra_data.shape != crc_data.shape:
    print("Error: CRA and CRC data arrays have different shapes!")
    print(f"CRA shape: {cra_data.shape}, CRC shape: {crc_data.shape}")
    exit()
if len(cra_data.shape) != 2 or cra_data.shape[1] != 10:
     print(f"Warning: Expected data shape (num_papers, 10), but got {cra_data.shape}. "
           f"Proceeding, but ensure this is correct.")

num_papers = cra_data.shape[0]
num_quantiles = cra_data.shape[1]
time_quantiles_indices = np.arange(1, num_quantiles + 1) # For trend analysis [1, 2, ..., 10]

print(f"Data loaded successfully: {num_papers} papers, {num_quantiles} time quantiles.\n")


# --- 1. Hypothesis Test: Compare OVERALL CRA vs CRC ---
# We compare the average CRA per paper vs the average CRC per paper
print("--- 1. Comparing Overall CRA vs CRC (Average per Paper) ---")

# Calculate the mean for each paper across the quantiles
cra_mean_per_paper = np.mean(cra_data, axis=1)
crc_mean_per_paper = np.mean(crc_data, axis=1)

# Calculate the difference for normality test
difference_mean = crc_mean_per_paper - cra_mean_per_paper

# Check normality of the differences
shapiro_test_stat, shapiro_p_value = stats.shapiro(difference_mean)
print(f"Shapiro-Wilk test on mean differences (normality check): Statistic={shapiro_test_stat:.4f}, P-value={shapiro_p_value:.4g}")

if shapiro_p_value > ALPHA:
    print("Differences appear normally distributed. Using Paired Samples t-test.")
    # Paired t-test
    t_statistic, p_value_ttest = stats.ttest_rel(crc_mean_per_paper, cra_mean_per_paper)
    # Effect size (Cohen's d for paired samples)
    mean_diff = np.mean(difference_mean)
    std_diff = np.std(difference_mean, ddof=1) # Sample std dev
    cohen_d = mean_diff / std_diff if std_diff != 0 else 0
    # Confidence Interval (CI) for the mean difference
    dof = len(difference_mean) - 1
    ci_lower, ci_upper = stats.t.interval(1 - ALPHA, dof, loc=mean_diff, scale=stats.sem(difference_mean))

    print(f"Paired Samples t-test: t({dof}) = {t_statistic:.4f}, P-value = {p_value_ttest:.4g}")
    print(f"Mean Difference (CRC - CRA): {mean_diff:.4f}")
    print(f"{1-ALPHA:.0%} Confidence Interval for Mean Difference: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"Effect Size (Cohen's d): {cohen_d:.4f}")
    if p_value_ttest < ALPHA:
        comparison = "higher" if mean_diff > 0 else "lower"
        print(f"Conclusion: Overall CRC is statistically significantly {comparison} than CRA (p < {ALPHA}).")
    else:
        print(f"Conclusion: No statistically significant difference between overall CRA and CRC (p >= {ALPHA}).")

else:
    print("Differences do not appear normally distributed. Using Wilcoxon Signed-Rank test.")
    # Wilcoxon Signed-Rank test (non-parametric paired test)
    try:
        # Use correction for zero differences automatically if needed with 'auto'
        wilcoxon_statistic, p_value_wilcoxon = stats.wilcoxon(crc_mean_per_paper, cra_mean_per_paper, alternative='two-sided', method='auto')
        # Calculate median difference for reporting
        median_diff = np.median(difference_mean)
        print(f"Wilcoxon Signed-Rank Test: Statistic = {wilcoxon_statistic:.4f}, P-value = {p_value_wilcoxon:.4g}")
        print(f"Median Difference (CRC - CRA): {median_diff:.4f}")
        if p_value_wilcoxon < ALPHA:
             comparison = "higher" if median_diff > 0 else "lower" # Based on median
             print(f"Conclusion: Overall CRC is statistically significantly different from (likely {comparison} than) CRA (p < {ALPHA}).")
        else:
             print(f"Conclusion: No statistically significant difference between overall CRA and CRC (p >= {ALPHA}).")
    except ValueError as e:
      print(f"Wilcoxon test could not be performed: {e}") # e.g., if all differences are zero

print("\n" + "="*50 + "\n")


# --- 2. Hypothesis Test: Compare CRA vs CRC at EACH Time Quantile ---
print(f"--- 2. Comparing CRA vs CRC at Each of the {num_quantiles} Time Quantiles ---")

for j in range(num_quantiles):
    quantile_num = j + 1
    print(f"\n--- Quantile {quantile_num} ---")
    cra_at_quantile = cra_data[:, j]
    crc_at_quantile = crc_data[:, j]
    difference_at_quantile = crc_at_quantile - cra_at_quantile

    # Optional: Check normality for each quantile's difference (can be verbose)
    # shapiro_stat_q, shapiro_p_q = stats.shapiro(difference_at_quantile)
    # print(f"  Shapiro-Wilk (Quantile {quantile_num}): p={shapiro_p_q:.4g}")
    # For simplicity here, we might apply the same test type as determined overall,
    # or consistently use Wilcoxon if unsure / expect non-normality often.
    # Let's re-run the logic for each quantile:

    shapiro_stat_q, shapiro_p_q = stats.shapiro(difference_at_quantile)
    if shapiro_p_q > ALPHA:
        print(f"  Differences normal at Q{quantile_num}. Using Paired t-test.")
        t_stat_q, p_val_q_ttest = stats.ttest_rel(crc_at_quantile, cra_at_quantile)
        mean_diff_q = np.mean(difference_at_quantile)
        std_diff_q = np.std(difference_at_quantile, ddof=1)
        cohen_d_q = mean_diff_q / std_diff_q if std_diff_q != 0 else 0
        dof_q = len(difference_at_quantile) - 1
        ci_lower_q, ci_upper_q = stats.t.interval(1 - ALPHA, dof_q, loc=mean_diff_q, scale=stats.sem(difference_at_quantile))

        print(f"  Paired t-test (Q{quantile_num}): t({dof_q}) = {t_stat_q:.4f}, P-value = {p_val_q_ttest:.4g}")
        print(f"  Mean Difference (CRC-CRA) at Q{quantile_num}: {mean_diff_q:.4f}, 95% CI: [{ci_lower_q:.4f}, {ci_upper_q:.4f}], Cohen's d: {cohen_d_q:.4f}")
        if p_val_q_ttest < ALPHA: print("  Conclusion: Significant difference found.")
        else: print("  Conclusion: No significant difference found.")
    else:
        print(f"  Differences non-normal at Q{quantile_num}. Using Wilcoxon test.")
        try:
            wilcoxon_stat_q, p_val_q_wilcoxon = stats.wilcoxon(crc_at_quantile, cra_at_quantile, alternative='two-sided', method='auto')
            median_diff_q = np.median(difference_at_quantile)
            print(f"  Wilcoxon Test (Q{quantile_num}): Statistic = {wilcoxon_stat_q:.4f}, P-value = {p_val_q_wilcoxon:.4g}")
            print(f"  Median Difference (CRC-CRA) at Q{quantile_num}: {median_diff_q:.4f}")
            if p_val_q_wilcoxon < ALPHA: print("  Conclusion: Significant difference found.")
            else: print("  Conclusion: No significant difference found.")
        except ValueError as e:
            print(f"  Wilcoxon test failed for Q{quantile_num}: {e}")

print("\n" + "="*50 + "\n")


# --- 3. Hypothesis Test: Time Trend Analysis (Average across Papers) ---
print("--- 3. Analyzing Time Trends for Average CRA and CRC ---")

# Calculate the average CRA and CRC across all papers for each time quantile
mean_cra_over_time = np.mean(cra_data, axis=0)
mean_crc_over_time = np.mean(crc_data, axis=0)

# --- CRA Trend ---
print("\n--- CRA Time Trend ---")
# Spearman rank correlation (robust to non-linear monotonic trends)
corr_cra, p_value_cra_trend = stats.spearmanr(time_quantiles_indices, mean_cra_over_time)
print(f"Spearman Correlation (Average CRA vs. Time Quantile):")
print(f"  Correlation Coefficient (rho): {corr_cra:.4f}")
print(f"  P-value: {p_value_cra_trend:.4g}")

if p_value_cra_trend < ALPHA:
    trend_direction = "increasing" if corr_cra > 0 else "decreasing"
    print(f"  Conclusion: There is a statistically significant monotonic {trend_direction} trend in average CRA over time (p < {ALPHA}).")
else:
    print(f"  Conclusion: No statistically significant monotonic trend detected in average CRA over time (p >= {ALPHA}).")

# --- CRC Trend ---
print("\n--- CRC Time Trend ---")
corr_crc, p_value_crc_trend = stats.spearmanr(time_quantiles_indices, mean_crc_over_time)
print(f"Spearman Correlation (Average CRC vs. Time Quantile):")
print(f"  Correlation Coefficient (rho): {corr_crc:.4f}")
print(f"  P-value: {p_value_crc_trend:.4g}")

if p_value_crc_trend < ALPHA:
    trend_direction = "increasing" if corr_crc > 0 else "decreasing"
    print(f"  Conclusion: There is a statistically significant monotonic {trend_direction} trend in average CRC over time (p < {ALPHA}).")
else:
    print(f"  Conclusion: No statistically significant monotonic trend detected in average CRC over time (p >= {ALPHA}).")

print("\n" + "="*50 + "\n")
print("Analysis complete.")

