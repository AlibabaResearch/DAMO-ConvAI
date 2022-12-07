_CITATION = "None"
ECD_CAND_NUM = {'train': 2, 'dev': 2, 'test': 10}


class ECDMetric:
    @staticmethod
    def __mean_reciprocal_rank(sort_data):
        sort_label = [s_d[2] for s_d in sort_data]
        assert 1 in sort_label
        return 1.0 / (1 + sort_label.index(1))

    @staticmethod
    def __precision_at_position_1(sort_data):
        if sort_data[0][2] == 1:
            return 1.0
        else:
            return 0

    @staticmethod
    def __recall_at_position_k_in_10(sort_data, k):
        sort_label = [s_d[2] for s_d in sort_data]
        select_label = sort_label[:k]
        return 1.0 * select_label.count(1) / sort_label.count(1)

    @staticmethod
    def _evaluation_one_session(data):
        '''
        :param data: one conversion session, which layout is [(score1, label1), (score2, label2), ..., (score10, label10)].
        :return: all kinds of metrics.
        '''
        # print(data)
        sort_data = sorted(data, key=lambda x: x[0], reverse=True)
        # print(sort_data)

        m_r_r = ECDMetric.__mean_reciprocal_rank(sort_data)
        r_1   = ECDMetric.__recall_at_position_k_in_10(sort_data, 1)
        r_2   = ECDMetric.__recall_at_position_k_in_10(sort_data, 2)
        r_5   = ECDMetric.__recall_at_position_k_in_10(sort_data, 5)
        return m_r_r, r_1, r_2, r_5


    def compute(self, CAND_NUM, logits, labels):
        sessions = []
        assert len(logits) == len(labels)
        
        for i in range(0, len(logits), CAND_NUM):
            sessions.append(
                list(zip(logits[i:i+CAND_NUM],
                 [ii>0 for ii in logits[i:i+CAND_NUM]],
                labels[i:i+CAND_NUM]))
            )
        
        sum_m_a_p = 0.0
        sum_m_r_r = 0.0
        sum_p_1 = 0.0
        sum_r_1 = 0.0
        sum_r_2 = 0.0
        sum_r_5 = 0.0

        total_s = len(sessions)
        for session in sessions:
            m_r_r, r_1, r_2, r_5 = ECDMetric._evaluation_one_session(session)

            # sum_m_a_p += m_a_p
            sum_m_r_r += m_r_r
            # sum_p_1 += p_1
            sum_r_1 += r_1
            sum_r_2 += r_2
            sum_r_5 += r_5

        return {
            # "map": sum_m_a_p/total_s,
            "mrr": sum_m_r_r/total_s,
            # "p_at_1": sum_p_1/total_s,
            "r_at_1": sum_r_1/total_s,
            "r_at_2": sum_r_2/total_s,
            "r_at_5": sum_r_5/total_s
            }


class ECDMetric_with_hard_data(ECDMetric):
    def compute(self, CAND_NUM, logits, hard_ids=None):
        sessions = []
        # assert len(logits) == len(labels)

        for i in range(0, len(logits), CAND_NUM):
            sessions.append(
                list(zip(logits[i:i + CAND_NUM],
                         [ii > 0 for ii in logits[i:i + CAND_NUM]],
                         [1] + [0] * 9))
            )

        sum_m_a_p = 0.0
        sum_m_r_r = 0.0
        sum_p_1 = 0.0
        sum_r_1 = 0.0
        sum_r_2 = 0.0
        sum_r_5 = 0.0

        total_s = 0
        bad_things = []
        for sess_id, session in enumerate(sessions):
            if sess_id not in hard_ids: continue
            total_s += 1
            m_r_r, r_1, r_2, r_5 = ECDMetric._evaluation_one_session(session)
            if m_r_r != 1: bad_things.append(sess_id)
            # sum_m_a_p += m_a_p
            sum_m_r_r += m_r_r
            # sum_p_1 += p_1
            sum_r_1 += r_1
            sum_r_2 += r_2
            sum_r_5 += r_5

        return {
            # "map": sum_m_a_p/total_s,
            "mrr": sum_m_r_r / total_s * 100,
            # "p_at_1": sum_p_1/total_s,
            "r_at_1": sum_r_1 / total_s * 100,
            "r_at_2": sum_r_2 / total_s * 100,
            "r_at_5": sum_r_5 / total_s * 100,
            # "bad": bad_things
        }


if __name__ == "__main__":
    import json
    # 计算整个测试集
    # main.py 的 evaluate 已计算

    # 计算 test-hard 子集
    with open('hard_dialog_id.json') as f:
        hard_ids = json.load(f)
    metric = ECDMetric_with_hard_data()
    with open('previous_pred_results/predict_results_PCM.json') as f:
        pred_logits = json.load(f)
    print(metric.compute(CAND_NUM=10, logits=pred_logits, hard_ids=hard_ids))
