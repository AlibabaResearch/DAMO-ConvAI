import json, random
import re

DOMAINS = ['attraction', 'car', 'class', 'flight', 'hospital',
           'hotel', 'movie', 'pc', 'restaurant', 'train', 'tv', 'weather', 'general']
DOMAIN_TOK = ['[旅游景点]', '[汽车]', '[辅导班]', '[飞机]', '[医院]', '[酒店]', '[电影]',
              '[电脑]', '[餐厅]', '[火车]', '[电视剧]', '[天气]', '[通用]']
DOMAIN_RE = r"(\[旅游景点\]|\[汽车\]|\[辅导班\]|\[飞机\]|\[医院\]|\[酒店\]|\[电影\]|\[电脑\]|\[餐厅\]|\[火车\]|\[电视剧\]|\[天气\]|\[通用\])"

DOMAIN_MAP_en2ch = {
    'attraction': '旅游景点',
    'car': '汽车',
    'class': '辅导班',
    'flight': '飞机',
    'hospital': '医院',
    'hotel': '酒店',
    'movie': '电影',
    'pc': '电脑',
    'restaurant': '餐厅',
    'train': '火车',
    'tv': '电视剧',
    'weather': '天气',
    'general': '通用'
}

DOMAIN_MAP_ch2en = dict(zip(DOMAIN_MAP_en2ch.values(), DOMAIN_MAP_en2ch.keys()))

ALL_DA = ['request', 'inform', 'recommend'] + ['affirm', 'negate', 'nooffer', 'general', 'bye', 'fallback']
DA_RE = r"(\[旅游景点\]|\[汽车\]|\[辅导班\]|\[飞机\]|\[医院\]|\[酒店\]|\[电影\]|\[电脑\]|" \
        r"\[餐厅\]|\[火车\]|\[电视剧\]|\[天气\]|\[通用\]|\[request\]|\[inform\]|\[recommend\]|" \
        r"\[affirm\]|\[negate\]|\[nooffer\]|\[general\]|\[bye\]|\[fallback\])"

# {'out_of_know', 'extra_know', 'info_coll', 'confirm', 'reasoning'}


INFORMABLE_SLOTS = {
    '医院': ['区域', '名称', '地铁可达', '性质', '等级', '类别', '重点科室'],
    '天气': ['城市', '日期'],
    '旅游景点': ['区域', '名称', '是否地铁直达', '景点类型', '最适合人群', '消费'],
    '汽车': ['动力水平', '厂商', '座位数', '所属价格区间', '油耗水平', '级别', '能源类型', '车型', '车系', '驱动方式'],
    '火车': ['出发地', '坐席', '日期', '目的地', '车型'],
    '电影': ['主演', '制片国家地区', '年代', '类型'],
    '电脑': ['CPU', '产品类别', '价格区间', '内存容量', '分类', '品牌', '屏幕尺寸', '系列'],
    '电视剧': ['主演', '制片国家地区', '年代', '类型'],
    '辅导班': ['区域', '年级', '时段', '每周', '科目', '难度'],
    '酒店': ['价位', '区域', '名称', '房型', '星级', '酒店类型'],
    '飞机': ['出发地', '日期', '目的地', '舱位档次'],
    '餐厅': ['价位', '区域', '名称', '是否地铁直达', '菜系']
}

BINARY_SLOT = ["是否地铁直达", "DSA", "3.0T MRI", "CT", "地铁可达", "座椅通风", "座椅加热", "定速巡航", "倒车影像", "3.0TMRI"]

NUMBER_MAP = {
    '0': '零',
    '1': '一',
    '2': '二',
    '3': '三',
    '4': '四',
    '5': '五',
    '6': '六',
    '7': '七',
    '8': '八',
    '9': '九',
    '10': '十'
}


# 后面所有 slot 都需要经过
def normalize_slot(s):
    s = re.sub(r"3\.0\s?[Tt]\s?", "", s)
    s = s.lower()
    s = s.replace('/', '').replace(' ', '')
    s = re.sub(r"\(.*\)$", "", s)
    return s.lower()


# use original tokens
UNK_token = '[UNK]'
PAD_token = '[PAD]'

special_tokens = [
    PAD_token, UNK_token,
    '<sos_u>', '<eos_u>',
    '<sos_b>', '<eos_b>',
    '<sos_db>', '<eos_db>',
    '<sos_a>', '<eos_a>',
    '<sos_r>', '<eos_r>',
    '<sos_d>', '<eos_d>',
    '<go_r>', '<go_b>', '<go_a>', '<go_d>']

sos_tokens = {
    'user': '<sos_u>', 'user_delex': '<sos_u>',
    'bspn': '<sos_b>', 'bspn_gen': '<sos_b>', 'pv_bspn': '<sos_b>',
    'aspn': '<sos_a>', 'aspn_gen': '<sos_a>', 'pv_aspn': '<sos_a>',
    'resp': '<sos_r>', 'resp_gen': '<sos_r>', 'pv_resp': '<sos_r>',
    'dspn': '<sos_d>', 'dspn_gen': '<sos_d>', 'pv_dspn': '<sos_d>'}

eos_tokens = {
    'user': '<eos_u>', 'user_delex': '<eos_u>',
    'bspn': '<eos_b>', 'bspn_gen': '<eos_b>', 'pv_bspn': '<eos_b>',
    'aspn': '<eos_a>', 'aspn_gen': '<eos_a>', 'pv_aspn': '<eos_a>',
    'resp': '<eos_r>', 'resp_gen': '<eos_r>', 'pv_resp': '<eos_r>',
    'dspn': '<eos_d>', 'dspn_gen': '<eos_d>', 'pv_dspn': '<eos_d>'}

# if __name__ == '__main__':
#     # import pprint
#     # import copy
#     # import re
#     #
#     # slot_value_set = {}
#     # for domain in DOMAINS:
#     #     with open('../db/%s_db.json' % domain) as f:
#     #         db = json.load(f)
#     #         all_slots = {k: [] for k in db[0]}
#     #         for data in db:
#     #             for k in all_slots:
#     #                 if k == '重点科室':
#     #                     for kk in data[k]:
#     #                         if kk not in all_slots[k]:
#     #                             all_slots[k].append(kk)
#     #                 elif data[k] not in all_slots[k]:
#     #                     all_slots[k].append(data[k])
#     #         slot_value_set[domain] = copy.deepcopy(all_slots)
#     #
#     # with open('../risawoz/all_dense_new_rewrite.json') as f:
#     #     all_data = json.load(f)
#     #     for dial in all_data.values():
#     #         for turn in dial['dialogue']:
#     #             bs = json.loads(turn['belief_state']['slot-values'])
#     #             for k, v in bs.items():
#     #                 dom, slot = k.split('-')
#     #                 if dom in slot_value_set:
#     #                     if v not in slot_value_set[dom][slot]:
#     #                         slot_value_set[dom][slot].append(v)
#     #
#     # for d in slot_value_set:
#     #     for s in slot_value_set[d]:
#     #         if isinstance(slot_value_set[d][s][0], str):
#     #             slot_value_set[d][s] = sorted(slot_value_set[d][s], key=lambda x: len(x), reverse=True)
#     #         else:
#     #             slot_value_set[d][s] = sorted(slot_value_set[d][s], reverse=True)
#     #
#     # with open('slot_value_set.json', 'w') as f:
#     #     json.dump(slot_value_set, f, indent=1, ensure_ascii=False)
#
#
#     # find binary slots
#
#     with open('slot_value_set.json') as f:
#         data = json.load(f)
#
#     # for domain, slot_values in data.items():
#     #     for s, v in  slot_values.items():
#     #         if len(v)< 5:
#     #             print(domain, s, v)
#
#
