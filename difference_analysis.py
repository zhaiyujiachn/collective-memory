import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 数据加载
def load_data(before_path, after_path):
    with open(before_path) as f:
        before_data = json.load(f)
        before = {int(item[0]): item[1] for item in before_data}
    with open(after_path) as f:
        after = json.load(f)
    
    indexes = sorted(set(before.keys()) & set(map(int, after.keys())))
    relative_diffs = [(before[i] - after[str(i)]) / before[i] if before[i] != 0 else 0 for i in indexes]
    return np.array(indexes), np.array(relative_diffs)

# 趋势分析
def analyze_trend(indexes, diffs):
    model = LinearRegression()
    model.fit(indexes.reshape(-1,1), diffs)
    return model.coef_[0], model.intercept_

# 可视化
def plot_results(indexes, diffs, slope):
    plt.figure(figsize=(10,6))
    plt.scatter(indexes, diffs, alpha=0.6, label='Actual Difference')
    
    # Plot trend line
    trend = slope * indexes
    plt.plot(indexes, trend, color='r', label=f'Trend Line (slope={slope:.2f})')
    
    plt.xlabel('Index')
    plt.ylabel('Relative Difference ((Before - After) / Before)')
    plt.title('Trend Analysis of Relative Differences')
    plt.legend()
    plt.grid(True)
    
    # 自动创建输出目录
    import os
    os.makedirs('output_figures', exist_ok=True)
    plt.savefig('output_figures/difference_trend.png')
    plt.close()

if __name__ == '__main__':
    indexes, diffs = load_data(
        'final_code_data/Medline/before_remove_converter_citation.json',
        'final_code_data/Medline/after_remove_converter_citation.json'
    )
    slope, intercept = analyze_trend(indexes, diffs)
    
    print(f'Linear Regression Results:\nSlope: {slope:.2f}\nIntercept: {intercept:.2f}')
    print(f'Trend Analysis: Relative difference {"increases" if slope > 0 else "decreases"} with index')
    
    plot_results(indexes, diffs, slope)