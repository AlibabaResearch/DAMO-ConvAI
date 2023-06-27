import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score


def multiclass_acc(preds, truths):
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def get_centroids(embeddings):
    centroids = embeddings.mean(dim=1)
    return centroids


def get_utterance_centroids(embeddings):
    sum_centroids = embeddings.sum(dim=1)
    sum_centroids = sum_centroids.reshape(sum_centroids.shape[0], 1, sum_centroids.shape[-1])
    num_utterances = embeddings.shape[1] - 1
    centroids = (sum_centroids - embeddings) / num_utterances
    return centroids


def get_cossim(embeddings, centroids):
    num_utterances = embeddings.shape[1]
    utterance_centroids = get_utterance_centroids(embeddings)
    utterance_centroids_flat = utterance_centroids.view(utterance_centroids.shape[0] * utterance_centroids.shape[1], -1)
    embeddings_flat = embeddings.reshape(embeddings.shape[0] * num_utterances, -1)
    cos_same = F.cosine_similarity(embeddings_flat, utterance_centroids_flat)
    centroids_expand = centroids.repeat((num_utterances * embeddings.shape[0], 1))
    embeddings_expand = embeddings_flat.unsqueeze(1).repeat(1, embeddings.shape[0], 1)
    embeddings_expand = embeddings_expand.view(embeddings_expand.shape[0] * embeddings_expand.shape[1],
                                               embeddings_expand.shape[-1])
    cos_diff = F.cosine_similarity(embeddings_expand, centroids_expand)
    cos_diff = cos_diff.view(
        embeddings.size(0),
        num_utterances,
        centroids.size(0))
    same_idx = list(range(embeddings.size(0)))
    cos_diff[same_idx, :, same_idx] = cos_same.view(embeddings.shape[0], num_utterances)
    cos_diff = cos_diff + 1e-6
    return cos_diff


def get_eer(preds, targets, debug=False):
    speaker2embeddings = {}
    for i in range(len(targets)):
        sp = targets[i]
        embedding = preds[i]
        if sp not in speaker2embeddings:
            speaker2embeddings[sp] = []
        speaker2embeddings[sp].append(embedding)
    for sp in speaker2embeddings:
        speaker2embeddings[sp] = np.stack(speaker2embeddings[sp], axis=0)
    N = 4
    M = 50
    avg_EER = 0
    for _ in tqdm(range(10)):
        batch_avg_EER = 0
        for batch_id, _ in enumerate(speaker2embeddings):
            speakers = random.sample(speaker2embeddings.keys(), N)
            all_utterances = []
            for speaker in speakers:
                speaker_npy = np.array(speaker2embeddings[speaker])
                utter_index = np.random.randint(0, speaker_npy.shape[0], M)
                utterance = speaker_npy[utter_index]  # [M, hidden_dim]
                all_utterances.append(utterance)
            all_utterances = np.stack(all_utterances, axis=0)  # [N, M, hidden_dim]
            all_utterances = torch.from_numpy(all_utterances)
            enrollment_embeddings, verification_embeddings = torch.split(all_utterances, int(M / 2), dim=1)
            enrollment_centroids = get_centroids(enrollment_embeddings)
            sim_matrix = get_cossim(verification_embeddings, enrollment_centroids)
            # calculating EER
            diff = 1
            EER = 0
            EER_thresh = 0
            EER_FAR = 0
            EER_FRR = 0
            for thres in [0.01 * i for i in range(101)]:
                sim_matrix_thresh = sim_matrix > thres
                FAR = (sum([sim_matrix_thresh[i].float().sum() - sim_matrix_thresh[i, :, i].float().sum() for i in
                            range(int(N))]) / (N - 1.0) / (float(M / 2)) / N)
                FRR = (sum([M / 2 - sim_matrix_thresh[i, :, i].float().sum() for i in range(int(N))])
                       / (float(M / 2)) / N)
                if diff > abs(FAR - FRR):
                    diff = abs(FAR - FRR)
                    EER = (FAR + FRR) / 2
                    EER_thresh = thres
                    EER_FAR = FAR
                    EER_FRR = FRR
            batch_avg_EER += EER
            if debug:
                print("\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EER, EER_thresh, EER_FAR, EER_FRR))
        avg_EER += batch_avg_EER / (batch_id + 1)
    avg_EER = avg_EER / 10
    return avg_EER


def eval_mosei_classification(test_preds, test_truth):
    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])

    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    mae = np.mean(np.absolute(test_preds - test_truth))
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    binary_truth_non0 = test_truth[non_zeros] > 0
    binary_preds_non0 = test_preds[non_zeros] > 0
    f_score_non0 = f1_score(binary_truth_non0, binary_preds_non0, average='weighted')
    acc_2_non0 = accuracy_score(binary_truth_non0, binary_preds_non0)
    binary_truth_has0 = test_truth >= 0
    binary_preds_has0 = test_preds >= 0
    acc_2 = accuracy_score(binary_truth_has0, binary_preds_has0)
    f_score = f1_score(binary_truth_has0, binary_preds_has0, average='weighted')
    return {'mae': mae, 'corr': corr, 'acc_a7': mult_a7, 'acc_a2': acc_2, 'acc_a2_non0': acc_2_non0, 'f1': f_score,
            'f1_non0': f_score_non0}


def downstream_metrics(pred_label, true_label, task):
    pred_label, true_label = np.array(pred_label), np.array(true_label)
    if task in ['mosi', 'mosei']:
        report_metric = eval_mosei_classification(pred_label, true_label)
    elif task in ['meld', 'test', 'iemocap', 'snipsmart']:
        f1 = f1_score(true_label, pred_label, average='weighted')
        acc = accuracy_score(true_label, pred_label)
        report_metric = {'accuracy': acc, 'weighted f1': f1}
    else:
        acc_twenty = accuracy_score(true_label, pred_label)
        f1_twenty = f1_score(true_label, pred_label, average='macro')
        pred_2class = (pred_label > 10).astype(int)
        true_2class = (true_label > 10).astype(int)
        acc_binary = accuracy_score(true_2class, pred_2class)
        f1_binary = f1_score(true_2class, pred_2class, average='macro')
        report_metric = {'acc_20': acc_twenty, 'f1_20': f1_twenty, 'acc_2': acc_binary, 'f1_2': f1_binary}
    return report_metric
