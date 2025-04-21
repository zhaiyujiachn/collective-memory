# %%
import json
from collections import defaultdict
import scienceplots
from datetime import datetime
import elasticsearch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from methods import SearchData
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

if not os.path.exists('data'):
    os.makedirs('data')

sd = SearchData("semantic_scholar")
# Medline DBLP Sociology
dataset = "DBLP"
datafolder = 'data/' + dataset + "/"


# %%

def write_dict(values, name):
    file_name = name
    with open(file_name, "w") as outfile:
        json.dump(values, outfile, sort_keys=True, indent=4)


def read_dict(name):
    file_name = name
    with open(file_name, "r") as infile:
        dict_values = json.load(infile)
        return dict_values


# %% 最早、最晚的论文时间

#print("数据中发表最早的论文时间是" + sd.get_latest_paper(dataset))
#print("数据中发表最晚的论文时间是" + sd.get_earliest_paper(dataset))

# %% 获取被引量最高100的论文

file1 = 'data/' + dataset + '_top_100.json'

if not os.path.exists(file1):
    top_100 = []
    filtered_paper = []
    for i in sd.get_all_paper(dataset):
        if i in filtered_paper:
            continue
        if not sd.get_paper_date(i):
            continue
        t = (len(sd.get_paper_citation(i)), i)
        top_100.append(t)
        if len(top_100) > 100:
            top_100.sort(reverse=True)
            top_100.pop()
    top_100 = [i[1] for i in top_100]
    write_dict(top_100, file1)
else:
    top_100 = read_dict(file1)

# %% getting adopters

file2 = 'data/' + dataset + '_adopters.json'
if not os.path.exists(file2):
    adopters = {}
    for i in tqdm(top_100, desc="getting adopters"):
        adopters[i] = defaultdict(list)
        for j in sd.get_paper_citation(i):
            if sd.is_paper_has_author(j):
                for au in sd.get_paper_author(j):
                    try:
                        if au:
                            adopters[i][au].append(sd.get_paper_date(j))
                    except elasticsearch.exceptions.NotFoundError:
                        continue
    for a in adopters[i]:
        adopters[i][a] = list(set(adopters[i][a]))
    write_dict(adopters, file2)
else:
    adopters = read_dict(file2)

# %% collaborators,converters,direct_converters,paper_adopter_collaborators


file3 = 'data/' + dataset + '.json'
if not os.path.exists(file3):
    collaborators = {}  # 合作者
    converters = {}  # 转化者
    direct_converters = {}  # 直接转化者
    paper_adopter_collaborators = {}
    for i in tqdm(top_100, desc='collaborators,converters,direct_converters,paper_adopter_collaborators'):
        collaborators[i] = defaultdict(list)
        converters[i] = defaultdict(list)
        direct_converters[i] = defaultdict(list)
        paper_adopter_collaborators[i] = defaultdict(list)  # 存放谁让谁转化了
        for j in tqdm(adopters[i], desc=str(top_100.index(i) + 1) + "/100"):
            try:
                for p in sd.get_author_paper(dataset, j):
                    paper_date = sd.get_paper_date(p)
                    if paper_date > min(adopters[i][j]):
                        for colla in sd.get_paper_author(p):
                            if colla in adopters[i]:
                                if colla != j and paper_date < min(adopters[i][colla]):
                                    # 如果这个人不是引出他的采纳者，且出现在采纳者中，但采纳时间比合作时间晚，那就是一个转化者
                                    collaborators[i][colla].append(paper_date)
                                    converters[i][colla].append(paper_date)
                                    paper_adopter_collaborators[i][colla].append((j, p, paper_date))
                                elif colla != j and paper_date == min(adopters[i][colla]):
                                    # 如果这个人不是引出他的采纳者，且出现在采纳者中，并且采纳时间与合作时间相同，那就是一个直接转化者
                                    collaborators[i][colla].append(paper_date)
                                    converters[i][colla].append(paper_date)
                                    direct_converters[i][colla].append(paper_date)
                                    paper_adopter_collaborators[i][colla].append((j, p, paper_date))
                            elif colla != j:
                                # 如果这个人不是引出他的采纳者，那就是一个合作者
                                collaborators[i][colla].append(paper_date)
            except elasticsearch.exceptions.NotFoundError:
                pass
            for colla in collaborators[i]:
                collaborators[i][colla] = list(set(collaborators[i][colla]))
            for j in converters[i]:
                converters[i][j] = list(set(converters[i][j]))
            for j in direct_converters[i]:
                direct_converters[i][j] = list(set(direct_converters[i][j]))
            for j in paper_adopter_collaborators[i]:
                paper_adopter_collaborators[i][j] = list(set(paper_adopter_collaborators[i][j]))
    dic = {"collaborators": collaborators, "converters": converters, "direct_converters": direct_converters,
           "paper_adopter_collaborators": paper_adopter_collaborators}
    write_dict(dic, file3)
else:
    dic = read_dict(file3)
    collaborators = dic["collaborators"]
    converters = dic["converters"]
    direct_converters = dic["direct_converters"]
    paper_adopter_collaborators = dic["paper_adopter_collaborators"]
# %% 画图

if not os.path.exists('data/' + dataset):
    os.makedirs('data/' + dataset)

df = pd.DataFrame(list(top_100), columns=['DOI'])
df = df.set_index('DOI')
for paper_id in top_100:
    year = sd.get_paper_date(paper_id)
    df.loc[paper_id, "Pub_year"] = int(year[:4])
    df.loc[paper_id, "Total_citation"] = len(sd.get_paper_citation(paper_id))
    df.loc[paper_id, "Total_collaborters"] = len(collaborators[paper_id])
    df.loc[paper_id, "Total_converters"] = len(converters[paper_id])
    df.loc[paper_id, "CRC"] = len(converters[paper_id]) / len(collaborators[paper_id])
    df.loc[paper_id, "CRA"] = len(converters[paper_id]) / len(adopters[paper_id])
    df.loc[paper_id, "Paper_age"] = 2021 - int(year[:4])
    df.loc[paper_id, "Total_adopters"] = len(adopters[paper_id])
df.to_csv(datafolder + dataset+ '_top100.csv')

# 画分布
sns.reset_defaults()
plt.style.use(['science', 'no-latex'])
plt.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(2, 2, figsize=(8, 8))
# Make default histogram of sepal length
sns.histplot(df["Total_citation"], kde=True, ax=ax[0, 0], color=(68 / 255, 144 / 255, 196 / 255));
ax[0, 0].set_xscale('log')
ax[0, 0].set_title('A')
sns.histplot(df["Total_adopters"], kde=True, ax=ax[0, 1], color=(255 / 255, 0 / 255, 0 / 255));
ax[0, 1].set_xscale('log')
ax[0, 1].set_title('B')
sns.histplot(df["Total_collaborters"], kde=True, ax=ax[1, 0], color=(112 / 255, 173 / 255, 71 / 255));
ax[1, 0].set_xscale('log')
ax[1, 0].set_title('C')
sns.histplot(df["Total_converters"], kde=True, ax=ax[1, 1], color=(237 / 255, 125 / 255, 49 / 255));
ax[1, 1].set_xscale('log')
ax[1, 1].set_title('D')
plt.tight_layout()
plt.savefig(datafolder + 'df_overview.svg', dpi=300, bbox_inches='tight')

fig, ax = plt.subplots(figsize=(8, 8))
ax = sns.jointplot(x=df["CRC"], y=df["CRA"], cmap="Reds", shade=True, color='red', kind='kde');
ax.fig.suptitle('E')
plt.tight_layout()
plt.savefig(datafolder + 'df_converters.svg', dpi=300, bbox_inches='tight')

df.hist(column=["CRC", "CRA"], figsize=(16, 4), layout=(1, 2))
plt.tight_layout()
plt.savefig(datafolder + 'df_converters.png', dpi=300, bbox_inches='tight')

# 画相关性
df1 = df[["Total_citation", "Total_collaborters", "Total_converters", "Paper_age", "CRC", "CRA"]]
plt.figure(figsize=(16, 4))
sns.heatmap(df1.corr(), annot=True, fmt=".2f")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(datafolder + 'df_overview_corr.png', dpi=300, bbox_inches='tight')

# 描述统计结果
statistics = df.describe()
statistics.to_csv(datafolder + 'top100_statistics.csv')

# %%
# direct converters
# How many of all converters are direct converters?

df = pd.DataFrame(list(top_100), columns=['DOI'])
df = df.set_index('DOI')
for paper_id in top_100:
    if len(converters[paper_id]) != 0:
        df.loc[paper_id, "Direct_conversion_ratio"] = len(direct_converters[paper_id]) / len(converters[paper_id])
    else:
        df.loc[paper_id, "Direct_conversion_ratio"] = 0

df.to_csv(datafolder +dataset+ '_direct_converters.csv')

# Direct_conversion_ratio
sns.reset_defaults()
plt.style.use(['science', 'no-latex'])
plt.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(figsize=(6, 8))
sns.histplot(df["Direct_conversion_ratio"], kde=True);
ax.set_xlabel('Proportion of direct converters')
ax.set_title('A')
plt.tight_layout()
plt.savefig(datafolder + 'df_direct_converters.png', dpi=300, bbox_inches='tight')

# %% indirect_converters

indirect_converters = defaultdict(set)
for paper_id in top_100:
    indirect_converters[paper_id].update(set(converters[paper_id].keys()) - set(direct_converters[paper_id].keys()))
# How many collaborations does an indirect converter need to convert before citing the original paper?
collaboration_freq_before_indirect = {}
for paper_id in top_100:
    collaboration_freq_before_indirect[paper_id] = defaultdict(int)
# How long does it take for indirect converters to convert?
last_collaboration = defaultdict(dict)
conversion_interval = defaultdict(dict)
for paper_id in top_100:
    for au in indirect_converters[paper_id]:
        adoption_time = min(adopters[paper_id][au])
        last_collaboration[paper_id][au] = min(collaborators[paper_id][au])
        for c_time in collaborators[paper_id][au]:
            if c_time < adoption_time:
                collaboration_freq_before_indirect[paper_id][au] += 1
                if c_time > last_collaboration[paper_id][au]:
                    last_collaboration[paper_id][au] = c_time
        conversion_interval[paper_id][au] = (datetime.strptime(adoption_time, '%Y-%m-%d') - datetime.strptime(
            last_collaboration[paper_id][au], '%Y-%m-%d')).days / 365

# %%

plt.style.use(['science', 'no-latex'])
plt.rcParams.update({'font.size': 16})
# Create figure with three axes
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
data = []
for i in collaboration_freq_before_indirect:
    for j in collaboration_freq_before_indirect[i]:
        data.append(collaboration_freq_before_indirect[i][j])
# Plot violin plot on axes 1

# Histogram plot and bin information acquisition
n, bins, patches = ax1.hist(data, alpha=0.7, bins=100, label='Frequency')
ax1.set_yscale('linear')
ax1.set_ylabel('Number of non-direct converters')
ax1.set_xlabel('Collaboration frequency')
ax1.set_title('B')

# Calculation of values for the 2nd axis
y2 = (np.add.accumulate(n) / n.sum()) * 100
x2 = np.convolve(bins, np.ones(2) / 2, mode="same")[1:]

# 2nd axis plot
ax12 = ax1.twinx()
lines = ax12.plot(x2, y2, ls='--', color='r', marker='o')
ax12.set_ylabel('Cumulative ratio, %')
ax1.grid(visible=False)
data1 = []
for i in conversion_interval:
    for j in conversion_interval[i]:
        data1.append(conversion_interval[i][j])


# Histogram plot and bin information acquisition
n, bins, patches = ax2.hist(data1, alpha=0.7, bins=100, label='Frequency')
ax2.set_yscale('linear')
ax2.set_ylabel('Number of non-direct converters')
ax2.set_xlabel('Conversion interval(years)')
ax2.set_title('C')
# Calculation of values for the 2nd axis
y2 = (np.add.accumulate(n) / n.sum()) * 100
x2 = np.convolve(bins, np.ones(2) / 2, mode="same")[1:]

# 2nd axis plot
ax22 = ax2.twinx()
lines = ax22.plot(x2, y2, ls='--', color='r', marker='o')
ax22.set_ylabel('Cumulative ratio, %')
ax2.grid(visible=False)
fig.tight_layout()
plt.savefig(datafolder + 'conversion_interval.svg', dpi=300, bbox_inches='tight')
plt.show()

# %% growth trending
if not os.path.exists(datafolder + "figure1/"):
    os.makedirs(datafolder + "figure1/")


def get_yearly_distribution(paper_id):
    yearly_citation = defaultdict(int)
    yearly_converters = defaultdict(int)
    yearly_adopters = defaultdict(int)
    yearly_collaborators = defaultdict(int)
    for i in sd.get_paper_citation(paper_id):
        try:
            citation_date = sd.get_paper_date(i)[:4]
            if citation_date != 'None':
                yearly_citation[citation_date] += 1
        except:
            pass
    for i in converters[paper_id]:
        t = min(adopters[paper_id][i])[:4]
        if t != 'None':
            yearly_converters[t] += 1
    for i in adopters[paper_id]:
        t = min(adopters[paper_id][i])[:4]
        if t != 'None':
            yearly_adopters[t] += 1
    for i in collaborators[paper_id]:
        t = min(collaborators[paper_id][i])[:4]
        if t != 'None':
            yearly_collaborators[t] += 1
    return yearly_citation, yearly_converters, yearly_adopters, yearly_collaborators


def get_res_dataframe(yearly_citation, yearly_adopters, yearly_converters, yearly_collaborators):
    df1 = pd.DataFrame(list(yearly_adopters.items()), columns=['Year', 'yearly_adopters'])
    df1 = df1.set_index('Year')
    df2 = pd.DataFrame(list(yearly_citation.items()), columns=['Year', 'yearly_citation'])
    df2 = df2.set_index('Year')
    df3 = pd.DataFrame(list(yearly_converters.items()), columns=['Year', 'yearly_converters'])
    df3 = df3.set_index('Year')
    df4 = pd.DataFrame(list(yearly_collaborators.items()), columns=['Year', 'yearly_collaborators'])
    df4 = df4.set_index('Year')
    res = pd.concat([df1, df2, df3, df4], axis=1)
    # print(res)
    res = res.fillna(0)
    # print(res.index)
    res.index = res.index.astype(int)
    res = res.sort_index()
    # res = res[:20]
    try:
        res = res.drop('2020')
    except:
        pass
    res['yearly_adopters_ratio'] = res['yearly_adopters'] / sum(res['yearly_adopters'])
    res['yearly_citation_ratio'] = res['yearly_citation'] / sum(res['yearly_citation'])
    res['yearly_converters_growth_ratio'] = res['yearly_converters'] / sum(res['yearly_converters'])
    res['yearly_non_converters'] = res['yearly_adopters'] - res['yearly_converters']
    res['yearly_non_converters_growth_ratio'] = (res['yearly_non_converters']) / sum(res['yearly_non_converters'])
    res['yearly_converters_adoptors_ratio'] = res['yearly_converters'] / res['yearly_adopters']
    res['yearly_collaborators_ratio'] = res['yearly_collaborators'] / sum(res['yearly_collaborators'])
    return res


# sns.set(style="darkgrid", palette="muted", color_codes=True)
plt.style.use(['science', 'no-latex'])
plt.rcParams.update({'font.size': 16})


# %matplotlib inline
def plot_citation_collaboration(df, paper_id, rank):
    fig, axs = plt.subplots(2, 1, figsize=(6, 8))
    # citation , new adopters, new collaboration
    """
    fig.suptitle("#"+str(rank)+" "+paper_id+\
                  "\nPublication Time: "+str(paper_year[paper_id])+\
                 "\nTotal Citations: "+str(int(df.yearly_citation.sum()))+\
                 "\nTotal Adoptors: "+str(int(df.yearly_adopters.sum()))+\
                 "\nTotal converters: "+str(int(df.yearly_converters.sum()))+\
                 "\nConvert ratio: "+str(round(df.yearly_converters.sum()/df.yearly_adopters.sum()*100,2)))
    """
    axs[0].plot(df.index.to_list(), np.cumsum(df.yearly_citation.to_list()), label='Citation', marker='o', linewidth=3,
                color=(68 / 255, 144 / 255, 196 / 255))
    axs[0].plot(df.index.to_list(), np.cumsum(df.yearly_adopters.to_list()), label='Adoptors', marker='o', linewidth=3,
                color=(255 / 255, 0 / 255, 0 / 255))
    axs[0].plot(df.index.to_list(), np.cumsum(df.yearly_collaborators.to_list()), label='Collaborators', marker='o',
                linewidth=3, color=(112 / 255, 173 / 255, 71 / 255))
    axs[0].legend(loc='upper left')
    axs[0].set_yscale('log')
    axs[0].set_title('A')
    axs[0].set_ylabel('Cumulative Frequency')
    '''
    axs[1].plot(df.index.to_list(), np.cumsum(df.yearly_non_converters_growth_ratio.to_list()),label = 'yearly_non_converters_growth_ratio', marker='o', linewidth=3)
    axs[1].plot(df.index.to_list(), np.cumsum(df.yearly_converters_growth_ratio.to_list()),label = 'yearly_converters_growth_ratio',  marker='o', linewidth=3)
    axs[1].legend(loc='upper left')
    '''
    axs[1].plot(df.index.to_list(), np.cumsum(df.yearly_non_converters.to_list()), label='yearly_non_converters_growth',
                marker='o', linewidth=3)
    axs[1].plot(df.index.to_list(), np.cumsum(df.yearly_converters.to_list()), label='yearly_converters_growth',
                marker='o', linewidth=3, color=(237 / 255, 125 / 255, 49 / 255))
    axs[1].legend(loc='upper left')
    axs[1].set_title('B')
    axs[1].set_ylabel('Cumulative Frequency')

    fig.tight_layout()
    fig.savefig(datafolder + "figure1/rank" + str(rank) + "_" + paper_id + ".svg", format='svg')
    plt.close()


def plot_citation_converters(df, paper_id, rank):
    fig, axs = plt.subplots(1, figsize=(8, 8))
    # citation , new adopters, new collaboration
    """
    fig.suptitle("#"+str(rank)+" "+paper_id+\
                  "\nPublication Time: "+str(paper_year[paper_id])+\
                 "\nTotal Citations: "+str(int(df.yearly_citation.sum()))+\
                 "\nTotal Adoptors: "+str(int(df.yearly_adopters.sum()))+\
                 "\nTotal converters: "+str(int(df.yearly_converters.sum()))+\
                 "\nConvert ratio: "+str(round(df.yearly_converters.sum()/df.yearly_adopters.sum()*100,2)))
    """
    df['non_converters_proportion'] = df['yearly_non_converters'] / df['yearly_adopters']
    df['converters_proportion'] = df['yearly_converters'] / df['yearly_adopters']
    df = df.fillna(0)
    axs.plot(df.index.to_list(), df.non_converters_proportion.to_list(), label='Non_converters_proportion', marker='o',
             linewidth=3)
    axs.plot(df.index.to_list(), df.converters_proportion.to_list(), label='Converters_proportion', marker='o',
             linewidth=3, color=(237 / 255, 125 / 255, 49 / 255))
    axs.legend(loc='upper left')

    fig.tight_layout()
    fig.savefig(datafolder + "figure1/rank" + str(rank) + "_converters_proportion_" + paper_id + ".svg",
                format='svg')
    plt.close()


for paper_id in top_100:
    yearly_citation, yearly_converters, yearly_adopters, yearly_collaborators = get_yearly_distribution(paper_id)
    df = get_res_dataframe(yearly_citation, yearly_adopters, yearly_converters, yearly_collaborators)
    plot_citation_collaboration(df, paper_id, top_100.index(paper_id) + 1)
    plot_citation_converters(df, paper_id, top_100.index(paper_id) + 1)


# %% cross point
def get_growth_curve_cross_points():
    cross_point = []
    exceed_point = []
    exceed_point_each_paper = defaultdict(list)
    for rank in tqdm(range(100)):
        paper_id = top_100[rank]
        yearly_citation, yearly_converters, yearly_adopters, yearly_collaborators = get_yearly_distribution(paper_id)
        df = get_res_dataframe(yearly_citation, yearly_adopters, yearly_converters, yearly_collaborators)
        yearly_non_converters = np.cumsum(df.yearly_non_converters.to_list())
        yearly_converters = np.cumsum(df.yearly_converters.to_list())

        for i in range(len(yearly_converters)):
            if yearly_converters[i] >= yearly_non_converters[i]:
                cross_point.append((i / len(yearly_converters)))
                break
        slope_converter = []
        slope_non_converter = []
        for i in range(len(yearly_converters) - 1):
            s1 = yearly_converters[i + 1] - yearly_converters[i]
            s2 = yearly_non_converters[i + 1] - yearly_non_converters[i]
            slope_converter.append(s1)
            slope_non_converter.append(s2)
            if s1 > s2:
                exceed_point.append((i / len(yearly_converters)))
                break
            # if s1 > s2:
            #     exceed_point_each_paper[rank].append(i + int(sd.get_paper_date(paper_id)[:4]))
            #     break
    return cross_point, exceed_point


cross_point, exceed_point = get_growth_curve_cross_points()

plt.style.use(['science', 'no-latex'])
plt.rcParams.update({'font.size': 16})
fig, axs = plt.subplots(1, 2, figsize=(8, 8))
axs[0].scatter(x=range(len(cross_point)), y=sorted(cross_point), label='Crossing points of growth curves', marker='o')
axs[1].scatter(x=range(len(exceed_point)), y=sorted(exceed_point), label='Overtake points of growth rates', marker='o',
               color=(237 / 255, 125 / 255, 49 / 255))
# axs[0].legend(loc='upper left')
# axs[1].legend(loc='upper left')
axs[0].set_title('C\nCrossing points\nof growth curves')
axs[1].set_title('D\nOvertake points\nof growth rates')
axs[0].set_xlabel('Reordered rankings')
axs[1].set_xlabel('Reordered rankings')
axs[0].set_ylabel('Proportion of article life cycle')
fig.tight_layout()
fig.savefig(datafolder + "Crossing Points of growth curves.svg", format='svg')

# %% conversion ratio
if not os.path.exists(datafolder + "figure3/"):
    os.makedirs(datafolder + "figure3/")


def get_yearly_converter_distribution(paper_id):
    converter_dis = defaultdict(int)
    for i in converters[paper_id]:
        if i in direct_converters[paper_id]:
            converter_dis[min(adopters[paper_id][i])[:4]] += 1
        elif conversion_interval[paper_id][i] < 5:
            converter_dis[min(adopters[paper_id][i])[:4]] += 1
    return converter_dis


# top 100 CRA/CRC graph
def plot_citation_converters(yearly_cra, yearly_crc, paper_id, rank):
    fig, (axs1, axs2) = plt.subplots(1, 2, figsize=(16, 8))
    axs1.plot(list(yearly_cra.keys()), list(yearly_cra.values()), label='CRA', marker='o', linewidth=3)
    axs1.legend(loc='upper left')
    axs2.plot(list(yearly_crc.keys()), list(yearly_crc.values()), label='CRC', marker='o', linewidth=3)
    axs2.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(datafolder + "figure3/rank" + str(rank) + "_" + paper_id + ".png", format='png')
    plt.close()


# top 100 CRA graph
for paper_id in top_100:
    converter_dis = get_yearly_converter_distribution(paper_id)
    year_range = []
    for i in range(int(sd.get_paper_date(paper_id)[:4]), 2016):
        window = []
        for j in range(5):
            window.append(i + j)
        year_range.append(window)
    yearly_cra = defaultdict()
    yearly_crc = defaultdict()
    yearly_citation, yearly_converters, yearly_adopters, yearly_collaborators = get_yearly_distribution(paper_id)
    for i in year_range:
        x = 0
        y = 0
        z = 0
        for j in i:
            j = str(j)
            x += converter_dis[j]
            y += yearly_adopters[j]
            z += yearly_collaborators[j]
        if y == 0:
            yearly_cra[i[-1]] = 0
        else:
            yearly_cra[i[-1]] = x / y
        if z == 0:
            yearly_crc[i[-1]] = 0
        else:
            yearly_crc[i[-1]] = x / z

    plot_citation_converters(yearly_cra, yearly_crc, paper_id, top_100.index(paper_id) + 1)

# %% conversion ratio in one figure

fig, (axs1, axs2) = plt.subplots(1, 2, figsize=(16, 8))
for paper_id in top_100:
    converter_dis = get_yearly_converter_distribution(paper_id)
    year_range = []
    for i in range(int(sd.get_paper_date(paper_id)[:4]), 2016):
        window = []
        for j in range(5):
            window.append(i + j)
        year_range.append(window)
    yearly_cra = defaultdict()
    yearly_crc = defaultdict()
    yearly_citation, yearly_converters, yearly_adopters, yearly_collaborators = get_yearly_distribution(paper_id)
    for i in year_range:
        x = 0
        y = 0
        z = 0
        for j in i:
            j = str(j)
            x += converter_dis[j]
            y += yearly_adopters[j]
            z += yearly_collaborators[j]
        if y == 0:
            yearly_cra[i[-1]] = 0
        else:
            yearly_cra[i[-1]] = x / y
        if z == 0:
            yearly_crc[i[-1]] = 0
        else:
            yearly_crc[i[-1]] = x / z
    axs1.plot(list(yearly_cra.keys()), list(yearly_cra.values()), marker='o', linewidth=3)
    axs2.plot(list(yearly_crc.keys()), list(yearly_crc.values()), marker='o', linewidth=3)
    fig.tight_layout()
    fig.savefig(datafolder + "figure3/rank_all.png", format='png')
    plt.close()

# %% cumulate conversion ratio

if not os.path.exists(datafolder + "figure4/"):
    os.makedirs(datafolder + "figure4/")


def get_cum_crc(paper_id):
    coll_list = []
    conv_list = []
    adp_list = []
    for i in collaborators[paper_id]:
        t = min(collaborators[paper_id][i])
        coll_list.append((i, t))
        if i in conversion_interval[paper_id] and conversion_interval[paper_id][i] <= 5:
            conv_list.append(i)
        elif i in converters[paper_id]:
            conv_list.append(i)
    for i in adopters[paper_id]:
        adp_list.append((i, min(adopters[paper_id][i])))
    coll_list = list(set(coll_list))
    coll_list = sorted(coll_list, key=lambda x: x[1], reverse=False)
    adp_list = sorted(adp_list, key=lambda x: x[1], reverse=False)
    # 累计采纳转化率
    start = 0
    cra = []
    w = int(len(adp_list) / 10)
    for i in range(1, 10):
        adp = []
        con = []
        for j in adp_list[start:w * i]:
            adp.append(j[0])
            if j[0] in conv_list:
                con.append(j[0])
        cra.append(len(con) / len(adp))
        start += w
    adp = []
    con = []
    for j in adp_list[start:]:
        adp.append(j[0])
        if j[0] in conv_list:
            con.append(j[0])
    adp = list(set(adp))
    cra.append(len(con) / len(adp))

    # 累计合作转化率
    start = 0
    crc = []
    w = int(len(coll_list) / 10)
    for i in range(1, 10):
        col = []
        con = []
        for j in coll_list[start:w * i]:
            col.append(j[0])
            if j[0] in conv_list:
                con.append(j[0])
        col = list(set(col))
        crc.append(len(con) / len(col))
        start += w
    col = []
    con = []
    for j in coll_list[start:]:
        col.append(j[0])
        if j[0] in conv_list:
            con.append(j[0])
    col = list(set(col))
    crc.append(len(con) / len(col))
    return cra, crc


file4 = 'data/' + dataset + '_cra.json'
file5 = 'data/' + dataset + '_crc.json'

if not os.path.exists(file4) and not os.path.exists(file5):
    cra_list = []
    crc_list = []
    for paper_id in tqdm(top_100):
        cra, crc = get_cum_crc(paper_id)
        cra_list.append(cra)
        crc_list.append(crc)
else:
    cra_list = read_dict(file4)
    crc_list = read_dict(file5)

plt.style.use(['science', 'no-latex'])
plt.rcParams.update({'font.size': 16})

fig = plt.figure(figsize=(16, 16))
outer = gridspec.GridSpec(10, 10, wspace=0.2, hspace=0.2)

for rank in range(100):
    inner = gridspec.GridSpecFromSubplotSpec(1, 2,
                                             subplot_spec=outer[rank], wspace=0.1, hspace=0.1)

    ax1 = plt.Subplot(fig, inner[0])
    sns.regplot(ax=ax1, x=list(range(1, 11)), y=list(cra_list[rank]), color=(255 / 255, 0 / 255, 0 / 255))
    ax1.set_xticks([])
    ax1.set_yticks([])
    fig.add_subplot(ax1)
    ax2 = plt.Subplot(fig, inner[1])
    sns.regplot(ax=ax2, x=list(range(1, 11)), y=list(crc_list[rank]), color=(112 / 255, 173 / 255, 71 / 255))
    ax2.set_xticks([])
    ax2.set_yticks([])
    fig.add_subplot(ax2)
fig.savefig(datafolder + "cra_crc.png", format='png', dpi=300, bbox_inches='tight')
fig.show()

write_dict(crc_list,"crc_list.json")
write_dict(cra_list,"cra_list.json")

# 分别画图
fig, axs = plt.subplots(10, 10, figsize=(16, 16), sharex=True)
for rank in range(100):
    sns.regplot(ax=axs[rank % 10][rank // 10], x=list(range(1, 11)), y=list(cra_list[rank]),
                color=(255 / 255, 0 / 255, 0 / 255))
#    axs2.plot(list(range(1,21)),list(crc), marker='o', linewidth=3)
plt.suptitle('CRA')
fig.tight_layout()
fig.savefig(datafolder + "cra.png", format='png')
plt.close()

fig, axs = plt.subplots(10, 10, figsize=(16, 16), sharex=True)
for rank in range(100):
    sns.regplot(ax=axs[rank % 10][rank // 10], x=list(range(1, 11)), y=list(crc_list[rank]),
                color=(112 / 255, 173 / 255, 71 / 255))
#    axs2.plot(list(range(1,21)),list(crc), marker='o', linewidth=3)
plt.suptitle('CRC')
fig.tight_layout()
fig.savefig(datafolder + "crc.png", format='png')
plt.close()

# boxplot
fig, axs = plt.subplots(1, figsize=(16, 8), sharex=True)
df = pd.DataFrame()
steps = []
y = []
for i in range(10):
    for j in crc_list:
        steps.append(str((i + 1) * 10) + "%")
        y.append(j[i])
df['Cumulative collaborator ratio'] = steps
df['CRC'] = y
ax = sns.boxplot(x='Cumulative collaborator ratio', y='CRC', data=df)
fig.tight_layout()
fig.savefig(datafolder + "figure4/crc_boxplot.png", format='png')
plt.close()


# boxplot
fig, axs = plt.subplots(1, figsize=(16, 8), sharex=True)
df = pd.DataFrame()
steps = []
y = []
for i in range(10):
    for j in cra_list:
        steps.append(str((i + 1) * 10) + "%")
        y.append(j[i])
df['Cumulative adopter ratio'] = steps
df['CRA'] = y
ax = sns.boxplot(x='Cumulative adopter ratio', y='CRA', data=df)
fig.tight_layout()
fig.savefig(datafolder + "figure4/cra_boxplot.png", format='png')
plt.close()

# %% remove converters paper
before = {}
after = {}
before_yearly={}
after_yearly={}
top100_before_yearly=[]
top100_after_yearly=[]
for rank, paper_id in enumerate(top_100):
    yearly_citation = defaultdict(int)
    yearly_citation_removed = defaultdict(int)
    for c_id in sd.get_paper_citation(paper_id):
        try:
            yearly_citation[sd.get_paper_date(c_id)[:4]] += 1
            c_index = 0
            for a in sd.get_paper_author(c_id):
                if a in converters[paper_id]:
                    c_index = 1
            if c_index == 0:
                yearly_citation_removed[sd.get_paper_date(c_id)[:4]] += 1
        except elasticsearch.exceptions.NotFoundError:
            pass
    before[rank] = len(sd.get_paper_citation(paper_id))
    after[rank] = sum(yearly_citation_removed.values())
before = sorted(before.items(), key=lambda x: x[1], reverse=True)

fig, axs1 = plt.subplots(10, 10, figsize=(16, 12))
rank = 0
for i in before:
    paper_id = top_100[i[0]]
    yearly_citation = defaultdict(int)
    yearly_citation_removed = defaultdict(int)
    for c_id in sd.get_paper_citation(paper_id):
        try:
            yearly_citation[sd.get_paper_date(c_id)[:4]] += 1
            c_index = 0
            for a in sd.get_paper_author(c_id):
                if a in converters[paper_id]:
                    c_index = 1
            if c_index == 0:
                yearly_citation_removed[sd.get_paper_date(c_id)[:4]] += 1
        except elasticsearch.exceptions.NotFoundError:
            pass
    yearly_citation = sorted(yearly_citation.items(), key=lambda x: x[0], reverse=False)
    yearly_citation_removed = sorted(yearly_citation_removed.items(), key=lambda x: x[0], reverse=False)
    top100_before_yearly.append(yearly_citation)
    top100_after_yearly.append(yearly_citation_removed)
    axs1[rank % 10][rank // 10].fill_between([x[0] for x in yearly_citation][:-1], [x[1] for x in yearly_citation][:-1],
                                             label='Yearly citation', color=(68 / 255, 144 / 255, 196 / 255))
    axs1[rank % 10][rank // 10].fill_between([x[0] for x in yearly_citation_removed][:-1],
                                             [x[1] for x in yearly_citation_removed][:-1],
                                             label='Yearly citation excluding converters', alpha=0.8,
                                             color=(237 / 255, 125 / 255, 49 / 255))
    # axs1[rank%10][rank//10].legend(loc='upper left')
    axs1[rank % 10][rank // 10].set(xticklabels=[])
    axs1[rank % 10][rank // 10].set(xlabel=None)
    rank += 1
plt.tick_params(bottom=False)
plt.suptitle('B')
fig.tight_layout()
fig.savefig(datafolder + "remove_converters_yearly1.png", format='png', dpi=300, bbox_inches='tight')
plt.close()
write_dict(before, "before_remove_converter_citation.json")
write_dict(after, "after_remove_converter_citation.json")
write_dict(top100_before_yearly, "top100_before_remove_converter_citation.json")
write_dict(top100_after_yearly, "top100_after_remove_converter_citation.json")

plt.style.use(['science', 'no-latex'])
plt.rcParams.update({'font.size': 16})
fig, axs1 = plt.subplots(1, figsize=(16, 4))
axs1.fill_between(range(1, 101), [i[1] for i in before], label='Total citation', alpha=0.8,
                  color=(68 / 255, 144 / 255, 196 / 255))
axs1.fill_between(range(1, 101), [after[i[0]] for i in before], label='Citation excluding converters', alpha=0.8,
                  color=(237 / 255, 125 / 255, 49 / 255))
axs1.set_yscale('log')
axs1.set_xlabel('Rankings')
axs1.set_ylabel('Citation')
axs1.set_xlim(1, 100)
plt.suptitle('A')
fig.legend(loc='center left')
fig.tight_layout()
fig.savefig(datafolder + "remove_converters_total.svg", format='svg', dpi=300, bbox_inches='tight')
plt.close()


# %%
# 采纳者的合作者都有哪些人？
def get_adopter_collaborators():
    adopter_collaborators = {}  # 合作者
    for rank, paper_id in enumerate(top_100):
        adopter_collaborators[paper_id] = defaultdict(list)
        for j in tqdm(adopters[paper_id]):
            try:
                for p in sd.get_paper_author(j):
                    if sd.get_paper_date(p) > min(adopters[i][j]):  # 采纳者的文章要发表在采纳之后
                        for colla in sd.get_paper_author(p):
                            adopter_collaborators[i][j].append(colla)
                adopter_collaborators[i][j] = list(set(adopter_collaborators[i][j]))
            except:
                pass
    return adopter_collaborators


adopter_collaborators = get_adopter_collaborators()

# paper_adopter_collaborators
# paper_adopter_collaborators[i][colla].append((j,p,paper_year[p]))
# i 是doi,colla合作者，j是合作的采纳者，p是合作的doi，最后是合作时间
# 这里colla都是转化者


adopter_converter = {}  # 转化者
for rank, paper_id in enumerate(top_100):
    adopter_converter[paper_id] = defaultdict(list)
    for a in adopter_collaborators[paper_id]:
        for c in adopter_collaborators[paper_id][a]:
            if c in adopters[paper_id]:
                adopter_converter[i][a].append(c)

dopter_conversion_power = defaultdict(dict)  # 转化者

for rank, paper_id in enumerate(top_100):
    for j in tqdm(adopters[paper_id]):
        try:
            dopter_conversion_power[paper_id][j] = len(adopter_converter[paper_id][j]) / len(
                adopter_collaborators[paper_id][j])
        except:
            print((i, j))
            pass

sort_adopters = defaultdict(dict)
for rank, paper_id in enumerate(top_100):
    for a in adopters[paper_id]:
        d = adopters[paper_id][a][0]
        for s in adopters[paper_id][a]:
            if d > s:
                d = s
        sort_adopters[paper_id][a] = d
        # print(min(adopters[i][a]))
    sort_adopters[paper_id] = sorted(sort_adopters[paper_id].items(), key=lambda x: x[1], reverse=False)

squence = defaultdict(list)
for rank, paper_id in enumerate(top_100):
    for a in sort_adopters[paper_id]:
        if a[0] in dopter_conversion_power[paper_id]:
            # squence[i].append(dopter_conversion_power[i][a[0]])
            squence[paper_id].append(len(adopter_converter[paper_id][a[0]]))

fig, axs1 = plt.subplots(1, figsize=(16, 8))
for rank, paper_id in enumerate(top_100):
    axs1.scatter(range(len(squence[paper_id])), squence[paper_id])
    # axs1[rank%10][rank//10].legend(loc='upper left')
    # axs1.set(xticklabels=[])
fig.tight_layout()
fig.savefig(datafolder + "conversion_power.png", format='png')
plt.close()
