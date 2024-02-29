import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
d = 128
import json
from xopen import xopen
import random
from scipy.signal import argrelextrema
from matplotlib.pyplot import MultipleLocator
import seaborn as sns
import palettable

def theta(i, base):
    return base ** (-2 * i / d)


def qmkn(m, base):
    result = 0
    for j in range(int(d / 2)):
        result += 2*np.cos(m * theta(j, base))
    return result/np.sqrt(d)

def diff_qmkn(m, base):
    result = 0 
    for j in range(int(d/2)):
        result += -2 * theta(j, base) * np.sin(m * theta(j, base))
    return result/np.sqrt(d)

def get_kv_retrieval_prompt(
    data,
    key: str,
    query_aware_contextualization: bool = False,
):
    with open('../lost-in-the-middle/src/lost_in_the_middle/prompts/kv_retrieval.prompt') as f:
        prompt_template = f.read().rstrip("\n")
    
    # Format the KV data into a string
    formatted_kv_records = ""
    for index, record in enumerate(data):
        start_character = "{" if index == 0 else " "
        data_string = f'"{record[0]}": "{record[1]}"'
        end_character = ",\n" if index != len(data) - 1 else "}"
        formatted_kv_records += start_character + data_string + end_character

    return prompt_template.format(formatted_kv_records=formatted_kv_records, key=key)

def get_tokenizer():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf', use_fast=False, padding_side="left", cache_dir='../llama-2-7b-hf')
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.bos_token_id = 1
    return tokenizer

def get_kv_sample():
    # tokenizer = get_tokenizer()
    with xopen('../lost-in-the-middle/kv_retrieval_data/kv-retrieval-75_keys.jsonl') as fin:
        data = fin.readlines()
        idx = random.choice(range(len(data)))
        input_example = json.loads(data[idx])
        key = input_example["key"]
        value = input_example["value"]
        original_kv_index = input_example["ordered_kv_records"].index([key, value])
        original_kv = input_example["ordered_kv_records"].pop(original_kv_index)

        ordered_kv_records = input_example["ordered_kv_records"][:5]
        
        ordered_kv_records.insert(12, original_kv)

        kv_prompt = get_kv_retrieval_prompt(
            data=ordered_kv_records[:-1], key=key,
        )
        print(kv_prompt)
        print(value)

def print_data_len():
    cnt = 0
    min_ = 20000
    max_ = 0
    with xopen('../kv-retrieval-75_keys.jsonl') as fin:
        for line in fin:
            input_example = json.loads(line)
            key = input_example["key"]
            value = input_example["value"]
            original_kv_index = input_example["ordered_kv_records"].index([key, value])
            original_kv = input_example["ordered_kv_records"].pop(original_kv_index)

            ordered_kv_records = input_example["ordered_kv_records"][:25]
            
            ordered_kv_records.insert(12, original_kv)

            kv_prompt = get_kv_retrieval_prompt(
                data=ordered_kv_records[:-1], key=key,
            )
        
            kv_prompts = kv_prompt.split( f'"{original_kv[1]}"')[0]
           
            prompt_input_pre = tokenizer.encode(kv_prompts) #, return_tensors="pt", padding=True
            left = len(prompt_input_pre)
            prompt_input_pre = tokenizer.encode(kv_prompts+f'"{original_kv[1]}"') #, return_tensors="pt", padding=True
            right = len(prompt_input_pre)
            prompt_input = tokenizer.encode(kv_prompt)
            total = len(prompt_input)
            if total-right > max_:
                max_ = total-right
            if total-right < min_:
                min_ = total-right
            
            print(total)
            cnt += 1
        print(min_)
        print(max_)

          
def caculate_mid_mse(sample, target):
    total = 0
    idx = 0
    idx_target = 0
    mse = 0.0
    while idx_target < len(target) and idx < len(sample)-1:
        if sample[idx] <= target[idx_target] <= sample[idx+1]:
            mse += np.power(target[idx_target] - (sample[idx] + sample[idx+1])/2,  2) 
            total += 1
            idx_target += 1
        elif sample[idx] > target[idx_target]:
            idx_target += 1
        else:
            idx += 1
    mse = np.sqrt(mse/total) if total != 0 else float('inf')
    return mse 

def caculate_cover(peak, trough, cover_rand):
    cover = 0
    dis = 0.0
    for i in peak:
        for j in trough:
            if abs(i-j) <= cover_rand:
                cover += 1
                dis += np.power((i-j), 2)
    return cover, dis

def merge_peak(ori_peak, new_peak):
    # new_peak = [k for k in new_peak if ori_peak[0] < k < ori_peak[-1]]
    result = ori_peak + new_peak
    result.sort()
    return result

def find_peak_points(mn, base):
    all_f_values_short = np.vectorize(qmkn)(mn, base)
    # plt.plot(mn, all_f_values_short)
    peak_all =  [k for k in argrelextrema(all_f_values_short, np.greater)[0]]
    p = [all_f_values_short[k] for k in peak_all]
    res = []
    node = max(p)
  
    while len(res) < 6:
        if abs(all_f_values_short[peak_all[0]] - node) <= 0.001:
            if len(res) < 1 or mn[peak_all[0]] - mn[res[-1]] > 50: 
                res.append(peak_all[0])
            index = min(peak_all[0]+50, peak_all[1])
            # if len(res) > 2:
            #     tmp = all_f_values_short[index: int(index+1.5*(res[-1] - res[-2]))]
            # else:
            tmp = all_f_values_short[index: index+500]
            node = max(tmp)
            peak_all.pop(0)        
        else:
            peak_all.pop(0)
    mn = [mn[k] for k in res]
    # plt.plot(mn, np.vectorize(qmkn)(mn, base), marker='o')
    # plt.savefig('test3.png')
    return mn

def find_trough_points(mn, base):
    all_f_values_short = np.vectorize(qmkn)(mn, base)
    # plt.plot(mn[:-500], all_f_values_short[:-500])
    peak_all =  [k for k in argrelextrema(all_f_values_short, np.less)[0]]
    p = [all_f_values_short[k] for k in peak_all]
    res = []
    node = min(p)
    while len(res) < 6:
        if abs(all_f_values_short[peak_all[-1]] - node) <= 0.001:
            if len(res) < 1 or res[-1] - peak_all[-1]  > 60: 
                res.append(peak_all[-1])
            index = min(peak_all[-1]-50, peak_all[-2])
            if index <=0:
                break
            tmp = all_f_values_short[:index]
            node = min(tmp) 
            peak_all.pop(-1) 
        else:
            peak_all.pop(-1)
    # prinst(res)
    mn = [mn[k] for k in res][::-1]
    # plt.plot(mn, np.vectorize(qmkn)(mn, base), marker='o')
    # plt.savefig('test3.png')
    return mn



def select_cover_max(mn, base_low, base_high, base_gap, final_base, trough, peak, cover_rand):
    max_cover = 0
    min_dis = float('inf')
    chosen_base = None
    for base in range(base_low, base_high+1, base_gap):
        if base in final_base:
            continue
        peak_ = find_peak_points(mn, base)
        trough_ = find_trough_points(mn, base)
        cover1, dis1 = caculate_cover(peak_, trough, cover_rand)
        cover2, dis2 = caculate_cover(peak, trough_, cover_rand)
        cover = cover1 + cover2
        dis = (dis1 + dis2)/cover if cover != 0 else float('inf')
        if cover > max_cover:
            max_cover = cover
            min_dis = dis
            chosen_base = base
        elif cover == max_cover:
            if dis < min_dis:
                min_dis = dis
                chosen_base = base 
    return chosen_base, max_cover, min_dis

def fliter(peak, trough, rand):
    # peak_ = []
    # for k in peak:
    #     flag = 0
    #     for i in range(rand):
    #         if k + i in trough or k - i in trough:
    #             flag = 1
    #     if flag == 0:
    #         peak_.append(k)
    # trough_ = []
    # for k in trough:
    #     flag = 0
    #     for i in range(rand):
    #         if k + i in peak or k - i in peak:
    #             flag = 1
    #     if flag == 0:
    #         trough_.append(k)
    peak_ = [k for k in  peak if k not in trough ]
    trough_ = [k for k in trough if k not in peak]
    return peak_, trough_


def base_choose_method1():
    word_low = 1000
    word_high = 4090
    total_base_num = 6
    # new_base_num = total_base_num - 2
    base_gap = 500
    base_low = 10000 
    base_high = 30000
    ori_base = 10000
    rand = 1
    cover_rand = 3
    final_base = [10000, 18000, 19000]
    # final_base = [10000, 17500, 18000, 19000, 20000, 25000]


    mn = [k for k in range(word_low, word_high+1)]
    
    if len(final_base) == 0:
        total_max_cover = 0
        total_min_dis = float('inf')
        for ori_base in range(base_low, base_high+1, base_gap):
            trough = find_trough_points(mn, ori_base)
            peak = find_peak_points(mn, ori_base)
            tmp_base = []
            chosen_base, max_cover, min_dis = select_cover_max(mn, ori_base+base_gap, base_high, base_gap, tmp_base, trough, peak, cover_rand)
            if max_cover > total_max_cover:
                total_max_cover = max_cover
                final_base = [ori_base, chosen_base]
            elif max_cover == total_max_cover:
                if min_dis < total_min_dis:
                    total_min_dis = min_dis
                    final_base = [ori_base, chosen_base]
        print(total_max_cover)
        print(final_base)
    
    trough = find_trough_points(mn, final_base[0])
    peak = find_peak_points(mn, final_base[0])
    for i in range(1, len(final_base)):
        trough_chosen = find_trough_points(mn, final_base[i])
        peak_chosen = find_peak_points(mn, final_base[i])
        peak = merge_peak(peak, peak_chosen)
        trough = merge_peak(trough, trough_chosen)
        peak, trough = fliter(peak, trough, rand)

    for time in range(total_base_num - len(final_base)):
        chosen_base, max_cover, min_dis = select_cover_max(mn, base_low, base_high, base_gap, final_base, trough, peak, cover_rand)
        print(max_cover)
        print(chosen_base)
        final_base.append(chosen_base)
        peak_chosen = find_peak_points(mn, chosen_base)
        trough_chosen = find_trough_points(mn, chosen_base)
        peak = merge_peak(peak, peak_chosen)
        trough = merge_peak(trough, trough_chosen)
        peak, trough = fliter(peak, trough, rand)
        print(peak)
        print(trough)
    
    final_base.sort()
    print(final_base)
    print(trough)
    print(peak)


def plt_base_list(final_base):
    base_list = final_base#[10000, 12000, 18000, 28000, 30000]
    mn = [k for k in range(900, 4096, 1)]
    for base in base_list:
        plt.plot(mn, np.vectorize(qmkn)(mn,base))
    plt.savefig('test3.png')

def find_high_points(mn, base, ratio=1.5):
    all_f_values_short = np.vectorize(qmkn)(mn, base)
    last_mn = [k for k in mn if qmkn(k, base)/all_f_values_short.mean() > ratio]
    return set(last_mn)


def plot_introduction():
    plt.figure(figsize=(7,8))
    m_values_short = [1,	5,	10,	15,	20,	25,	30,	35]
    f_values_short = [100,	72,	88,	72,	78,	48,	90,	92]
    axs0 = plt.subplot(211)
    axs0.plot(m_values_short, f_values_short, marker='o',  linewidth=3.5, markersize='10', markerfacecolor='none', markeredgewidth='3', 
                color='yellowgreen', alpha=0.6)
    axs0.set_title('(a)',fontsize=20)
    axs0.set_xlabel('Target key-value pair index',fontsize=18,labelpad=10)
    axs0.set_ylabel('Accuary',fontsize=18)
    plt.xlim(-2,38)
    plt.ylim(35,110)
    x_major_locator=MultipleLocator(5)
    axs0.xaxis.set_major_locator(x_major_locator)

    m_values_short = [k for k in range(-1,3005, 1)]
    f_values_short = np.vectorize(qmkn)(m_values_short,10000)
    axs1 = plt.subplot(212)
    axs1.plot(m_values_short, f_values_short,linewidth=2, color='steelblue', alpha=0.8)
    axs1.set_title('(b)',fontsize=20)
    axs1.set_xlabel('Relative token position',fontsize=18,labelpad=10)
    axs1.set_ylabel('Attention score before softmax',fontsize=16, labelpad=5)
    axs0.tick_params(labelsize=15, axis="both", which="major", width=1, length=5)
    axs1.tick_params(labelsize=15)
    # x_major_locator=MultipleLocator(250)
    # axs1.xaxis.set_major_locator(x_major_locator)
    # # plt.legend()
    # fig, axs = plt.subplots(2,1, figsize=(8,10))
    # plt.ylim(40,110)
    # axs[1].plot(m_values_short, f_values_short)
    # axs[1].set_title('(b)',fontsize=20)
    # axs[1].set_xlabel('Distance betwween query and key tokens',fontsize=20,labelpad=10)
    # axs[1].set_ylabel('Attention score before softmax',fontsize=16)
    # # plt.legend()

    # m_values_short = [1,	5,	10,	15,	20,	25,	30,	35]
    # f_values_short = [100,	72,	88,	72,	78,	48,	90,	92]

    # axs[0].plot(m_values_short, f_values_short, marker='s',linewidth=3.0, markersize='12',color='olivedrab')
    # axs[0].set_title('(a)',fontsize=20)
    # axs[0].set_xlabel('Target key-value pair index',fontsize=20,labelpad=10)
    # axs[0].set_ylabel('Accuary',fontsize=18)
    # # # plt.legend()
    # # plt.savefig('test4.png')
    # axs[0].tick_params(labelsize=15, axis="both", which="major",  width=1, length=5)
    # axs[1].tick_params(labelsize=15)
    # plt.tight_layout()
    # axs1.grid(True)
    axs0.yaxis.grid(color='lightgray', linestyle='--')
    bwith = 1.5 #边框宽度设置为2
    # axs0.set_facecolor('aliceblue')
    # axs1.set_facecolor('aliceblue')
    # #设置边框
    axs0.spines['bottom'].set_linewidth(bwith)#图框下边
    axs0.spines['left'].set_linewidth(bwith)#图框左边
    axs0.spines['top'].set_linewidth(bwith)#图框上边
    axs0.spines['right'].set_linewidth(bwith)#图框右边
    axs1.spines['bottom'].set_linewidth(bwith)#图框下边
    axs1.spines['left'].set_linewidth(bwith)#图框左边
    axs1.spines['top'].set_linewidth(bwith)#图框上边
    axs1.spines['right'].set_linewidth(bwith)#图框右边
    plt.tight_layout()
    # plt.savefig('test3.png',dpi=100)
    plt.savefig('test3.pdf')

    
def plot_base_selection_sample():
    import mpl_toolkits.axisartist as axisartist
    #创建画布
    fig = plt.figure(figsize=(6, 4))
    #使用axisartist.Subplot方法创建一个绘图区对象ax
    ax = axisartist.Subplot(fig, 111)  
    #将绘图区对象添加到画布中
    fig.add_axes(ax)
    base_list =  [10000, 20000, 30000]
    mn = [k for k in range(1, 3000, 1)]
    for base in base_list:
        plt.plot(mn, np.vectorize(qmkn)(mn,base))
    #给x坐标轴加上箭头
    ax.axis["bottom"].set_axisline_style("->", size = 1.0)
    #添加y坐标轴，且加上箭头
    ax.axis["left"].set_axisline_style("->", size = 1.0)
    #设置x、y轴上刻度显示方向
    base_list =  [10000, 20000, 30000]
    ax.axis['top'].set_visible(False)
    ax.axis['right'].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()
    bwith = 2 #边框宽度设置为2
    
    # #设置边框
    ax.spines['bottom'].set_linewidth(bwith)#图框下边
    ax.spines['left'].set_linewidth(bwith)#图框左边
    plt.savefig('test3.png')


def plot_base_selection():
    # 加图例
    # from scipy.interpolate import make_interp_spline
    plt.figure(figsize=(12,5))
    axs0 = plt.subplot(121)
    base_list = [10000,  20000]  #[10000, 15000, 18000, 19000, 20000]
    mn = np.array([k for k in range(1400, 2100, 1)])
    # mn =  np.linspace(1500,2000, 10000)
    x_sm = mn

    base = 18000
    # peak = find_trough_points(mn, base)
    # print(peak)
    y_sm0 = np.vectorize(qmkn)(mn,base)
    # y_sm = make_interp_spline(mn, np.vectorize(qmkn)(mn,base))(x_sm) 
    axs0.plot(x_sm, y_sm0, alpha=0.6, color='#1f77b4', label = 'candidate 1')
    axs0.scatter([1690,1805], np.vectorize(qmkn)([1690,1805], base), color='#1f77b4')#[1451,1549,1690,1805,1970],1970, 2103
    axs0.set_title('(a)',fontsize=16)
    i = 0


    # peak = find_peak_points(mn, base)
    # print(peak)
    buff1 = 6
    buff2 = 10
    # trough = find_peak_points(mn, 10000)
    # print(trough)
    # trough = find_trough_points(mn, 20000)
    # print(trough)
    # plt.scatter(trough, np.vectorize(qmkn)(trough, base))W
    y_sm1 = np.vectorize(qmkn)(mn,10000)
    y_sm2 = np.vectorize(qmkn)(mn,20000)
    # y_sm = make_interp_spline(mn, np.vectorize(qmkn)(mn,base))(x_sm) 
    axs0.plot(x_sm, y_sm1+buff1, alpha=0.6, color='#8467bd', label = '$\mathcal{B}_{c} $')
    axs0.plot(x_sm, y_sm2+buff2, alpha=0.6, color='#2ca02c', label = 'candidate 2')
    i += 1
    MIN  = min(min(y_sm0), min(y_sm1),min(y_sm2))
    MAX  = max(max(y_sm0),max(y_sm1),max(y_sm2))
    axs0.scatter([1707,1835], np.vectorize(qmkn)([1707,1835], 10000)+buff1, color='#8467bd') #[1707,1835,1970],1970,2118]
    axs0.scatter([1798,1882], np.vectorize(qmkn)([1798,1882], 20000)+buff2, color='#2ca02c') #[1540,1613,1798,1882,2098],2098, 2197]

    axs0.set_ylim((MIN-3, MAX+13))
    axs0.set_yticks([])
    axs0.legend(loc=3)
    axs1 = plt.subplot(122)
    base_list =   [17500, 18000, 19000, 20000, 10000, 25000] #[10000, 15000,17500, 18000, 19000, 20000]
    mn = [k for k in range(900, 4200, 1)]
    i = 0
    for base in base_list:
        
        buff = 3.9*i
        if base == 10000:
            buff += 0.6
        axs1.plot(mn, np.vectorize(qmkn)(mn,base)+buff, alpha=0.8, label = str(base))
        i += 1
    axs1.set_yticks([])
    axs1.legend(loc=3)
    axs0.set_xlabel('Relative token position',fontsize=14,labelpad=10)
    axs1.set_xlabel('Relative token position',fontsize=14,labelpad=10)
    axs1.set_title('(b)',fontsize=16)
    plt.tight_layout()
    # ax = plt.gca()
    # ax.axes.yaxis.set_ticklabels([])
    # plt.grid()
    # plt.grid(color='lightgray', linestyle='--')
    # plt.savefig('test4.png',dpi=1000)
    plt.savefig('test4.pdf')

    
def plot_ablation_heat():
    f = [float(k.strip()) for k in open('heat.txt').readlines()]
    n = len(f)
    matrix = np.random.random((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i][j] = (f[i] - f[j])
    
    ax = sns.heatmap(matrix, annot=True, annot_kws={"size":6.5}, cmap='summer_r', 
        vmin=-0.9,vmax=0.9, center=0.05, 
        cbar=False,
        xticklabels=['1.00','1.20','1.40', '1.50', '1.65', '1.80', '1.90', '2.00','2.25','2.50', '3.00', '$\mathcal{B}_{A.S.1}$ ','$\mathcal{B}_{A.S.2}$ ','$\mathcal{B}_{c1} $','$\mathcal{B}_{c2} $','$\mathcal{B}_{c3} $'] , #x轴方向刻度标签开关、赋值，可选“auto”, bool, list-like（传入列表）, or int,
        yticklabels=['1.00','1.20','1.40', '1.50', '1.65', '1.80', '1.90', '2.00','2.25','2.50', '3.00', '$\mathcal{B}_{A.S.1}$ ','$\mathcal{B}_{A.S.2}$ ','$\mathcal{B}_{c1} $','$\mathcal{B}_{c2} $','$\mathcal{B}_{c3} $'])
    #['10000','12000','14000', '15000', '16500', '18000', '19000', '20000','22500','25000', '30000', '$\mathcal{B}_{rand1} $','$\mathcal{B}_{rand2} $','$\mathcal{B}_{c1} $','$\mathcal{B}_{c2} $','$\mathcal{B}_{c3} $']
    # cbar = ax.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=9)
    # plt.tight_layout()
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    # plt.savefig('test.png',dpi=1000)
    plt.savefig('test.pdf')

if __name__ == '__main__':
    print('run code')
    # plot_ablation()
    # plot_introduction()
    # get_kv_sample()
    # plot_introduction()
    # plot_base_selection()
    # word_low = 1200
    # word_high = 4090
    # l = [k for k in range(word_low, word_high+1)]
    # # base = 10500
    # # find_trough_points(mn, base)
    # for base in range(10000, 30000, 500):
    #     l = [k for k in range(word_low, word_high+1)]
    #     res = find_peak_points(l, base)
    # base_choose_method1()
    # ## method1 peak+trough cover
    # # gap 2000
    # base_list = [10000, 18000, 20000, 22000, 24000, 26000, 30000] 
    # # gap 1000
    # base_list = [10000, 17000, 18000, 19000, 20000, 23000, 25000]
    # # gap 500
    # base_list = [10000, 17500, 18000, 19000, 20000, 22500, 25000]
    # ## method2 high_points
    # base_list = [10000, 11000, 12000, 13000, 14000, 15000, 16000]

    # plt_base_list(base_list)

    # base_list =  [10000, 20000, 30000]
    # mn = [k for k in range(1, 3000, 1)]
    # for base in base_list:
    #     plt.plot(mn, np.vectorize(qmkn)(mn,base))
    # plt.savefig('test3.png')