# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 16:13:22 2025

@author: Administrator
"""
import pandas as pd

# 生成数据列表（这里修正了doi列表的生成方式）
doi = []
Unique_Authors = []
Coauthored_Citing = []
Total_Citations = []
Paper_Age = []
Num_Authors_Focal = []

for i in adopters:
    # DOI列
    doi.append(i)  # 这里假设adopters中的i就是DOI
    # 独作作者数
    Unique_Authors.append(len(adopters[i]))
    # 核心参数计算
    Num_Authors_Focal.append(len(sd.get_paper_author(i)))
    Paper_Age.append(2022 - int(sd.get_paper_date(i)[:4]))
    Total_Citations.append(len(sd.get_paper_citation(i)))
    
    # 计算合著引用
    coauthornum = 0
    for c in sd.get_paper_citation(i):
        if len(sd.get_paper_author(c)) > 1:
            coauthornum += 1
    Coauthored_Citing.append(coauthornum)

# 创建DataFrame
df = pd.DataFrame({
    'DOI': doi,
    'Unique_Authors': Unique_Authors,
    'Coauthored_Citing': Coauthored_Citing,
    'Paper_Age': Paper_Age,
    'Num_Authors_Focal': Num_Authors_Focal,
    'Total_Citations': Total_Citations
})

# 按指定列顺序排序
df = df[['DOI', 'Unique_Authors', 'Coauthored_Citing', 'Paper_Age', 'Num_Authors_Focal', 'Total_Citations']]

# 保存为CSV
df.to_csv('paper_metrics.csv', index=False)