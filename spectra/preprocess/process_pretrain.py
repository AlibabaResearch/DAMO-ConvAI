import os
import sys
import json
import pickle
import librosa
import argparse
import numpy as np
from transformers import RobertaTokenizerFast as RTF
to = RTF.from_pretrained("roberta-base")

def cut_by_limit(words):
    cut = []
    for j, word in enumerate(words):
        st = round(float(word['startTime'][:-1]) * 10)
        et = round(float(word['endTime'][:-1]) * 10)
        # cut long words to avoid BGM in it
        if et - st > 30:
            if word['word'][0].isupper():
                st = et - 30
            else:
                et = st + 30
        if cut and et - cut[0][0][0] > max_len:
            yield cut
            cut = []
        if cut:
            cut[0].append([st, et + 1])
            cut[1].append(word['word'].lower())
        else:
            cut = [[[st, et + 1]], [word['word'].lower()]]
    if cut:
        yield cut

def get_path(f):
    return "/".join(f.split('/')[:-1])

def collect(c, wf, audio=None):
    global length, audio_id, target, audio_path
    audio_start, audio_end = c[0][0][0], c[0][-1][1]
    l = audio_end - audio_start
    wfn = os.path.join(wf, f"{audio_id}.npy")
    data = [wfn, []]
    for (i, word) in enumerate(c[1]):
        st, et = c[0][i]
        tids = to.encode(word)
        CLS, SEP = tids[0], tids[-1]
        # [word, first token index, last token index, start speech frame index, end speech frame index]
        data.append([word, len(data[1]) + 1, len(data[1]) + len(tids) - 1,
                        (st - audio_start) * SAMPLE_RATE, (et - audio_start) * SAMPLE_RATE])
        data[1] += tids[1:-1]
    data[1] = [CLS] + data[1] + [SEP]
    data.extend([audio_start, audio_end])
    if audio is not None:
        audio_piece = audio[c[0][0][0] * SAMPLE_RATE: c[0][-1][1] * SAMPLE_RATE]
        np.save(wfn, audio_piece)
    datas.append(data)
    length += l
    audio_id += 1

parser = argparse.ArgumentParser()
parser.add_argument("--speech_dir", type=str, required=True)
parser.add_argument("--text_dir", type=str, required=True)
parser.add_argument("--save_processed_speech_dir", type=str, required=True)
parser.add_argument("--save_processed_text_filename", type=str, required=True)
parser.add_argument("--hours", type=int, default=60000)
parser.add_argument("--sample_rate", type=int, default=16000)
parser.add_argument("--max_speech_slice_length", type=int, default=10)
args = parser.parse_args()

audio_path = args.speech_dir
text_path = args.text_dir
target = args.save_processed_speech_dir
fn = args.save_processed_text_filename
max_len = args.max_speech_slice_length * 10 - 1
dsecond = args.hours * 36000
SAMPLE_RATE = args.sample_rate // 10
datas = []
audio_id = 0
length = 0

files = 0
for r, ds, fs in os.walk(audio_path):
    if not ds:
        current_path = r[len(audio_path):]
        output_path = target + current_path
        os.makedirs(output_path, exist_ok=True)
        transcript_path = text_path + current_path + "/"
        for f in fs:
            with open(transcript_path + f[:-4] + ".json", "r+") as j:
                words = json.load(j)['results'][-1]['alternatives'][0]['words']
            audio = librosa.load(os.path.join(r, f))
            for cut in cut_by_limit(words):
                collect(cut, output_path, audio)
            del audio
            if length >= dsecond:
                # insert index of next dialog turn
                for i in range(len(datas) - 1, 0, -1):
                    datas[i] = datas[i][:-2] + ([i - 1] if get_path(datas[i][0]) == get_path(datas[i-1][0]) and datas[i-1][-1] + 50 > datas[i][-2] else [-1])
                datas[0] = datas[0][:-2] + [-1]
                with open(os.path.join(target, fn), "wb") as fp:
                    pickle.dump(datas, fp)
                print(f"Total processed {files + 1} files | {len(datas)} audio slices | total audio length {length}s")
                sys.exit(0)
        files += 1
        print(f"Processed {files} files | {audio_id} audio slices | total audio length {length}s")
print(f"Total processed {files} files | {len(datas)} audio slices | total audio length {length}s")
with open(os.path.join(target, fn), "wb") as fp:
    pickle.dump(datas, fp)
