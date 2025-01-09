import json
import pickle
import pandas as pd
import os
import csv
import hashlib
import os.path as osp
import time
import numpy as np
import validators
import mimetypes


def LMUDataRoot():
    if 'LMUData' in os.environ and osp.exists(os.environ['LMUData']):
        return os.environ['LMUData']
    home = osp.expanduser('~')
    # root = osp.join(home, 'LMUData')
    # os.makedirs(root, exist_ok=True)
    root = '/mnt/workspace/lr/datasets/testdatasets/LMUData'
    os.makedirs(root, exist_ok=True)
    return root

def MMBenchOfficialServer(dataset_name):
    root = LMUDataRoot()

    if dataset_name in ['MMBench', 'MMBench_V11', 'MMBench_CN', 'MMBench_CN_V11']:
        ans_file = f'{root}/{dataset_name}.tsv'
        if osp.exists(ans_file):
            data = load(ans_file)
            if 'answer' in data and sum([pd.isna(x) for x in data['answer']]) == 0:
                return True

    if dataset_name in ['MMBench_TEST_EN', 'MMBench_TEST_CN', 'MMBench_TEST_EN_V11', 'MMBench_TEST_CN_V11']:
        ans_file1 = f'{root}/{dataset_name}.tsv'
        mapp = {
            'MMBench_TEST_EN': 'MMBench', 'MMBench_TEST_CN': 'MMBench_CN',
            'MMBench_TEST_EN_V11': 'MMBench_V11', 'MMBench_TEST_CN_V11': 'MMBench_CN_V11',
        }
        ans_file2 = f'{root}/{mapp[dataset_name]}.tsv'
        for f in [ans_file1, ans_file2]:
            if osp.exists(f):
                data = load(f)
                if 'answer' in data and sum([pd.isna(x) for x in data['answer']]) == 0:
                    return True
    return False


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None
        return json.JSONEncoder.default(self, obj)


# LOAD & DUMP
def dump(data, f, **kwargs):
    def dump_pkl(data, pth, **kwargs):
        pickle.dump(data, open(pth, 'wb'))

    def dump_json(data, pth, **kwargs):
        json.dump(data, open(pth, 'w'), indent=4, ensure_ascii=False, cls=NumpyEncoder)

    def dump_jsonl(data, f, **kwargs):
        lines = [json.dumps(x, ensure_ascii=False, cls=NumpyEncoder) for x in data]
        with open(f, 'w', encoding='utf8') as fout:
            fout.write('\n'.join(lines))

    def dump_xlsx(data, f, **kwargs):
        data.to_excel(f, index=False, engine='xlsxwriter')

    def dump_csv(data, f, quoting=csv.QUOTE_ALL):
        data.to_csv(f, index=False, encoding='utf-8', quoting=quoting)

    def dump_tsv(data, f, quoting=csv.QUOTE_ALL):
        data.to_csv(f, sep='\t', index=False, encoding='utf-8', quoting=quoting)

    handlers = dict(pkl=dump_pkl, json=dump_json, jsonl=dump_jsonl, xlsx=dump_xlsx, csv=dump_csv, tsv=dump_tsv)
    suffix = f.split('.')[-1]
    return handlers[suffix](data, f, **kwargs)


def load(f):
    def load_pkl(pth):
        return pickle.load(open(pth, 'rb'))

    def load_json(pth):
        return json.load(open(pth, 'r', encoding='utf-8'))

    def load_jsonl(f):
        lines = open(f, encoding='utf-8').readlines()
        lines = [x.strip() for x in lines]
        if lines[-1] == '':
            lines = lines[:-1]
        data = [json.loads(x) for x in lines]
        return data

    def load_xlsx(f):
        return pd.read_excel(f)

    def load_csv(f):
        return pd.read_csv(f)

    def load_tsv(f):
        return pd.read_csv(f, sep='\t')

    handlers = dict(pkl=load_pkl, json=load_json, jsonl=load_jsonl, xlsx=load_xlsx, csv=load_csv, tsv=load_tsv)
    suffix = f.split('.')[-1]
    return handlers[suffix](f)


def download_file(url, filename=None):
    import urllib.request
    from tqdm import tqdm

    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    if filename is None:
        filename = url.split('/')[-1]

    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=filename, reporthook=t.update_to)
    return filename


def ls(dirname='.', match=[], mode='all', level=1):
    if isinstance(level, str):
        assert '+' in level
        level = int(level[:-1])
        res = []
        for i in range(1, level + 1):
            res.extend(ls(dirname, match=match, mode='file', level=i))
        return res

    if dirname == '.':
        ans = os.listdir(dirname)
    else:
        ans = [osp.join(dirname, x) for x in os.listdir(dirname)]
    assert mode in ['all', 'dir', 'file']
    assert level >= 1 and isinstance(level, int)
    if level == 1:
        if isinstance(match, str):
            match = [match]
        for m in match:
            if len(m) == 0:
                continue
            if m[0] != '!':
                ans = [x for x in ans if m in x]
            else:
                ans = [x for x in ans if m[1:] not in x]
        if mode == 'dir':
            ans = [x for x in ans if osp.isdir(x)]
        elif mode == 'file':
            ans = [x for x in ans if not osp.isdir(x)]
        return ans
    else:
        dirs = [x for x in ans if osp.isdir(x)]
        res = []
        for d in dirs:
            res.extend(ls(d, match=match, mode=mode, level=level - 1))
        return res


def mrlines(fname, sp='\n'):
    f = open(fname).read().split(sp)
    while f != [] and f[-1] == '':
        f = f[:-1]
    return f


def mwlines(lines, fname):
    with open(fname, 'w') as fout:
        fout.write('\n'.join(lines))


def md5(s):
    hash = hashlib.new('md5')
    if osp.exists(s):
        with open(s, 'rb') as f:
            for chunk in iter(lambda: f.read(2**20), b''):
                hash.update(chunk)
    else:
        hash.update(s.encode('utf-8'))
    return str(hash.hexdigest())


def last_modified(pth):
    stamp = osp.getmtime(pth)
    m_ti = time.ctime(stamp)
    t_obj = time.strptime(m_ti)
    t = time.strftime('%Y%m%d%H%M%S', t_obj)[2:]
    return t


def parse_file(s):
    if osp.exists(s):
        assert osp.isfile(s)
        suffix = osp.splitext(s)[1].lower()
        mime = mimetypes.types_map.get(suffix, 'unknown')
        return (mime, s)
    elif validators.url(s):
        suffix = osp.splitext(s)[1].lower()
        if suffix in mimetypes.types_map:
            mime = mimetypes.types_map[suffix]
            dname = osp.join(LMUDataRoot(), 'files')
            os.makedirs(dname, exist_ok=True)
            tgt = osp.join(dname, md5(s) + suffix)
            download_file(s, tgt)
            return (mime, tgt)
        else:
            return ('url', s)
    else:
        return (None, s)
