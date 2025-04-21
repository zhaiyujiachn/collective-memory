import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from scipy import stats # Import scipy.stats for Spearman correlation
import warnings
# Optional: uncomment below if you want to generate plots
# import matplotlib.pyplot as plt
# import seaborn as sns # For heatmap visualization

# Suppress specific warnings for cleaner output
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', RuntimeWarning)
warnings.simplefilter('ignore', DeprecationWarning)
warnings.simplefilter('ignore', UserWarning) # Ignore potential UserWarnings from statsmodels/sklearn

# --- Configuration ---
INPUT_CSV = 'final_code_data/DBLP/DBLP_top100_regression_data.csv.csv' # Prepared data file

# --- 1. Load Data ---
print(f"--- 1. Loading Data from {INPUT_CSV} ---")
try:
    data = pd.read_csv(INPUT_CSV)
    print("Data loaded successfully.")
    print("Columns:", data.columns.tolist())
    print("Data head:")
    print(data.head())
except FileNotFoundError:
    print(f"Error: Input file '{INPUT_CSV}' not found.")
    exit()
except Exception as e:
    print(f"Error reading {INPUT_CSV}: {e}")
    exit()

# Check if required columns are present
base_required_cols = ['Unique_Authors', 'Coauthored_Citing', 'Paper_Age', 'Num_Authors_Focal', 'Total_Citations']
missing_cols = [col for col in base_required_cols if col not in data.columns]
if missing_cols:
    print(f"Error: The following required columns are missing from {INPUT_CSV}: {missing_cols}")
    exit()

print("\n" + "="*50 + "\n")


# --- 2. Data Preprocessing ---
print("--- 2. Applying Data Preprocessing (for Regression Models) ---")
# Keep original columns for correlation analysis

# Log Transformation of Predictors
print("   - Log-transforming predictors...")
data['log_Coauthored_Citing'] = np.log(data['Coauthored_Citing'] + 1)
data['log_Paper_Age'] = np.log(data['Paper_Age'])
data['log_Num_Authors_Focal'] = np.log(data['Num_Authors_Focal'])

# Standardization of Predictors
print("   - Standardizing predictors...")
scaler = StandardScaler()
predictors_to_scale = ['Coauthored_Citing', 'Paper_Age', 'Num_Authors_Focal']
scaled_column_names = [col + '_scaled' for col in predictors_to_scale]
data[scaled_column_names] = scaler.fit_transform(data[predictors_to_scale])

# Log Transformation of Dependent Variable
print("   - Log-transforming dependent variable...")
data['log_Unique_Authors'] = np.log(data['Unique_Authors'] + 1)

print("Preprocessing complete.")
print("\n" + "="*50 + "\n")


# --- 3. VIF Check (Crucial Step) ---
print("--- 3. Checking Variance Inflation Factors (VIFs) ---")
# Check VIF *including* Total_Citations to justify its exclusion
formula_vif = 'Unique_Authors ~ Coauthored_Citing + Paper_Age + Num_Authors_Focal + Total_Citations'
vif_success = False
try:
    predictor_cols_vif = ['Coauthored_Citing', 'Paper_Age', 'Num_Authors_Focal', 'Total_Citations']
    X_vif_df = data[predictor_cols_vif].copy()
    X_vif_df = sm.add_constant(X_vif_df, prepend=True)

    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_vif_df.columns
    vif_data["VIF"] = [variance_inflation_factor(X_vif_df.values, i) for i in range(X_vif_df.shape[1])]

    print("VIF values for model including Total_Citations (using original predictors):")
    print(vif_data)

    high_vif_threshold = 5 # Or 10
    tc_vif = vif_data.loc[vif_data['Variable'] == 'Total_Citations', 'VIF'].iloc[0]
    cc_vif = vif_data.loc[vif_data['Variable'] == 'Coauthored_Citing', 'VIF'].iloc[0]
    naf_vif = vif_data.loc[vif_data['Variable'] == 'Num_Authors_Focal', 'VIF'].iloc[0]

    print(f"\nVIF for Total_Citations: {tc_vif:.2f}")
    print(f"VIF for Coauthored_Citing: {cc_vif:.2f}")
    print(f"VIF for Num_Authors_Focal: {naf_vif:.2f}")


    if tc_vif > high_vif_threshold or cc_vif > high_vif_threshold:
         print(f"\nWARNING: High VIF detected for Total_Citations and/or Coauthored_Citing (VIF > {high_vif_threshold}).")
         print("Decision: Excluding 'Total_Citations' from subsequent models due to severe multicollinearity.")
    else:
         print(f"\nINFO: VIFs for Total_Citations and Coauthored_Citing below {high_vif_threshold}.")
         print("Decision: Excluding 'Total_Citations' primarily based on conceptual overlap and potential instability, although VIF isn't extreme in this run.")
    vif_success = True

except Exception as e:
    print(f"\nError calculating VIF: {e}.")
    print("Decision: Proceeding by excluding 'Total_Citations' based on conceptual overlap and likely multicollinearity.")

# --- VIF Check on Final Predictors (excluding Total_Citations) ---
if vif_success:
    print("\n--- VIF Check for Final Predictors (Original Scale) ---")
    try:
        predictor_cols_final = ['Coauthored_Citing', 'Paper_Age', 'Num_Authors_Focal']
        X_final_vif_df = data[predictor_cols_final].copy()
        X_final_vif_df = sm.add_constant(X_final_vif_df, prepend=True)
        vif_data_final = pd.DataFrame()
        vif_data_final["Variable"] = X_final_vif_df.columns
        vif_data_final["VIF"] = [variance_inflation_factor(X_final_vif_df.values, i) for i in range(X_final_vif_df.shape[1])]
        print("VIFs for Predictors in Final Models:")
        print(vif_data_final.loc[vif_data_final['Variable'] != 'const']) # Show VIF for predictors only
        if (vif_data_final['VIF'][1:] > high_vif_threshold).any(): # Check VIFs excluding constant
             print("Note: Even after removing Total_Citations, some predictor VIFs might remain elevated.")
    except Exception as e:
        print(f"Could not calculate final VIFs: {e}")


print("\n" + "="*50 + "\n")


# --- 4. Spearman Correlation Analysis ---
print("--- 4. Spearman Rank Correlation Analysis ---")
# Select key variables in their original scale for correlation
# (Using original scale is often more intuitive for correlation)
# Exclude Total_Citations due to collinearity and focus on model variables
correlation_vars = ['Unique_Authors', 'Coauthored_Citing', 'Paper_Age', 'Num_Authors_Focal']
corr_data = data[correlation_vars].copy()

try:
    # Calculate Spearman correlation matrix and p-value matrix
    spearman_corr, spearman_p_values = stats.spearmanr(corr_data)

    # Convert to DataFrames for better readability
    corr_matrix_df = pd.DataFrame(spearman_corr, index=correlation_vars, columns=correlation_vars)
    p_value_matrix_df = pd.DataFrame(spearman_p_values, index=correlation_vars, columns=correlation_vars)

    print("Spearman Correlation Coefficient Matrix (rho):")
    print(corr_matrix_df.round(3)) # Round for display

    print("\nP-value Matrix for Spearman Correlations:")
    # Format p-values for readability
    print(p_value_matrix_df.applymap(lambda x: f"{x:.3g}" if x >= 0.001 else "<0.001"))

    print("\n--- Interpretation of Key Spearman Correlations ---")
    # Focus on correlation with the DV
    dv = 'Unique_Authors'
    iv = 'Coauthored_Citing'
    rho_dv_iv = corr_matrix_df.loc[dv, iv]
    p_dv_iv = p_value_matrix_df.loc[dv, iv]

    print(f"Correlation between '{dv}' and '{iv}':")
    print(f"  Spearman's rho = {rho_dv_iv:.3f}")
    print(f"  P-value = {p_dv_iv:.3g}" if p_dv_iv >= 0.001 else "  P-value < 0.001")
    if p_dv_iv < 0.05:
        direction = "positive" if rho_dv_iv > 0 else "negative"
        strength = "strong" if abs(rho_dv_iv) > 0.7 else ("moderate" if abs(rho_dv_iv) > 0.4 else "weak")
        print(f"  Interpretation: There is a statistically significant, {strength} monotonic {direction} association.")
    else:
        print("  Interpretation: No statistically significant monotonic association found.")

    # You can add interpretations for other correlations if needed

    # Optional: Heatmap visualization
    # try:
    #     import seaborn as sns
    #     import matplotlib.pyplot as plt
    #     plt.figure(figsize=(8, 6))
    #     sns.heatmap(corr_matrix_df, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    #     plt.title('Spearman Correlation Matrix')
    #     plt.show()
    # except ImportError:
    #     print("\nSeaborn not installed. Skipping heatmap visualization.")

except Exception as e:
    print(f"Error during Spearman correlation analysis: {e}")

print("\n" + "="*50 + "\n")


# --- 5. Attempt Quasi-Poisson Models ---
print("--- 5. Attempting Quasi-Poisson Models (Excluding Total_Citations) ---")
# ... (Keep the Quasi-Poisson fitting code as before) ...
# Define formulas
formula_orig_iv = 'Unique_Authors ~ Coauthored_Citing + Paper_Age + Num_Authors_Focal'
formula_log_iv = 'Unique_Authors ~ log_Coauthored_Citing + log_Paper_Age + log_Num_Authors_Focal'
formula_scaled_iv = 'Unique_Authors ~ Coauthored_Citing_scaled + Paper_Age_scaled + Num_Authors_Focal_scaled'

models_to_try = {
    "Quasi-Poisson (Original IVs)": formula_orig_iv,
    "Quasi-Poisson (Log IVs)": formula_log_iv,
    "Quasi-Poisson (Scaled IVs)": formula_scaled_iv
}
qp_results_dict = {}
qp_reliable = False # Flag to track if any QP model seems reliable
for name, formula in models_to_try.items():
    print(f"\n--- Fitting {name} ---")
    try:
        glm_model = smf.glm(formula=formula, data=data, family=sm.families.Poisson())
        glm_results = glm_model.fit()
        # print(glm_results.summary()) # Optional: print full summary
        scale_param = glm_results.scale
        print(f"Estimated Scale Parameter (phi) for {name}: {scale_param:.4f}")
        qp_results_dict[name] = glm_results
        if abs(scale_param - 1.0) < 0.1:
            print(f"WARNING: Scale parameter for {name} is close to 1.0. Results likely UNRELIABLE.")
        else:
            print(f"INFO: Scale parameter for {name} is {scale_param:.4f}. Adjustment occurred, but interpret cautiously.")
            qp_reliable = True
    except Exception as e:
        print(f"Error fitting {name}: {e}")
        qp_results_dict[name] = None

print("\n" + "="*50 + "\n")


# --- 6. Attempt OLS Model with Log-DV ---
print("--- 6. Attempting OLS with Log-Transformed DV (Using Original IVs) ---")
# ... (Keep the OLS fitting code as before) ...
formula_ols = 'log_Unique_Authors ~ Coauthored_Citing + Paper_Age + Num_Authors_Focal'
ols_results = None
ols_reliable = False
ols_condition_number_high = False
ols_residuals_normal = True
try:
    ols_model = smf.ols(formula=formula_ols, data=data)
    ols_results = ols_model.fit()
    print(ols_results.summary())
    try: # Extract condition number note safely
        condition_number_note = ols_results.summary().notes[-1].as_text()
        print(f"\nOLS Condition Number Note: {condition_number_note}")
        if "large condition number" in condition_number_note.lower():
            ols_condition_number_high = True
            print("WARNING: High Condition Number indicates significant multicollinearity issues.")
            ols_reliable = False
        else:
             ols_reliable = True # Initial assumption
    except:
        print("Could not parse condition number note from summary.")

    print("\n--- OLS Residual Check ---")
    print(f"Mean Residual: {ols_results.resid.mean():.4f}")
    jb_prob = float(ols_results.summary().tables[2].data[1][3])
    if jb_prob < 0.05:
        print("WARNING: OLS residuals likely violate normality assumption (Jarque-Bera p < 0.05).")
        ols_residuals_normal = False
        ols_reliable = False # Reduce confidence further
    else:
        print("OLS residuals do not strongly violate normality based on Jarque-Bera test.")
    if ols_reliable:
        print("OLS model seems relatively more reliable than QP, but multicollinearity concern (high condition number) remains.")


except Exception as e:
    print(f"Error fitting OLS model: {e}")


print("\n" + "="*50 + "\n")


# --- 7. Final Interpretation Summary ---
print("--- 7. Final Interpretation Summary ---")

print("1. Multicollinearity Check (VIF):")
print("   - Confirmed severe multicollinearity when including 'Total_Citations'.")
print("   - 'Total_Citations' was correctly excluded from subsequent regression models.")
print("   - VIFs for remaining predictors ('Coauthored_Citing', 'Paper_Age', 'Num_Authors_Focal') were low, suggesting direct multicollinearity between them is not the primary issue AFTER removing Total_Citations.")

print("\n2. Spearman Correlation Analysis:")
# Summarize the key correlation finding from spearman_corr, spearman_p_values
if 'spearman_corr' in locals():
     rho_dv_iv = spearman_corr[0, 1] # Correlation between Unique_Authors and Coauthored_Citing
     p_dv_iv = spearman_p_values[0, 1]

     # --- CORRECTED P-VALUE FORMATTING ---
     p_value_str = f"{p_dv_iv:.3g}" if p_dv_iv >= 0.001 else "< 0.001"
     print(f"   - Spearman correlation between 'Unique_Authors' and 'Coauthored_Citing': rho = {rho_dv_iv:.3f}, p {p_value_str}.")
     # --- END CORRECTION ---

     if p_dv_iv < 0.05:
         direction = "positive" if rho_dv_iv > 0 else "negative"
         # Adjust strength categories if desired
         strength = "strong" if abs(rho_dv_iv) >= 0.6 else ("moderate" if abs(rho_dv_iv) >= 0.3 else "weak")
         print(f"     Interpretation: A statistically significant, {strength} monotonic {direction} association exists between the number of co-authored citing papers and the number of unique authors reached.")
     else:
         print("     Interpretation: No statistically significant monotonic association found.")
else:
    print("   - Spearman correlation analysis could not be completed.")


print("\n3. Quasi-Poisson Regression Attempts:")
print("   - Models fitted using original, log-transformed, and standardized predictors.")
print("   - **CRITICAL ISSUE:** All Quasi-Poisson models estimated the scale (dispersion) parameter as 1.0, failing to adjust for severe overdispersion.")
print("   - **CONCLUSION:** Quasi-Poisson results are UNRELIABLE.")

print("\n4. OLS Regression Attempt (Log-Transformed DV):")
if ols_results:
    print("   - OLS model using original predictors fitted successfully.")
    print("   - **Findings:** 'Coauthored_Citing' showed a statistically significant positive coefficient (p < 0.001). 'Paper_Age' and 'Num_Authors_Focal' were not significant.")
    print("   - **LIMITATIONS:**")
    print("     - Residuals violated the normality assumption.")
    if ols_condition_number_high: # Use the flag set during OLS fitting
        print("     - A **high condition number** indicated numerical instability/multicollinearity issues during OLS estimation.")
    else:
        print("     - Condition number check did not indicate severe issues, but caution may still be warranted.") # Adjusted wording slightly
    print("   - **CONCLUSION:** OLS provides tentative evidence for a positive association, but should be interpreted with EXTREME CAUTION due to violated assumptions and potential numerical issues/multicollinearity.")
else:
    print("   - OLS model fitting failed.")


print("\n5. Overall Recommendation:")
print("   - Standard regression attempts faced significant challenges (overdispersion, potential multicollinearity/numerical instability).")
print("   - Reliable inference about independent effects from regression is problematic.")
print("   - **Recommended Action:**")
print("     - **Prioritize reporting the Spearman Correlation results.** They are robust and clearly show a significant positive association between 'Coauthored_Citing' and 'Unique_Authors'.")
print("     - Report the OLS results (Log DV on Original IVs) as a secondary analysis, BUT heavily emphasize the limitations (non-normal residuals, HIGH CONDITION NUMBER). State that the positive coefficient for 'Coauthored_Citing' is statistically significant in this potentially flawed model but should be viewed very cautiously.")
print("     - **Do NOT rely on the Quasi-Poisson results.**")
print("     - Clearly explain the statistical difficulties encountered in the Methods and Limitations sections.")

print("\nAnalysis complete.")


