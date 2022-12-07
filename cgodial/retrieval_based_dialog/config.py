# chinese-roberta-wwm-ext 自行下载
# https://huggingface.co/hfl/chinese-roberta-wwm-ext

class Config:
    batch_size = 32
    max_seq_len = 128
    max_dial_len = 60
    use_cuda = False
    gradient_accumulation_steps = 1
    num_train_epochs = 20
    save_epochs = 2
    warmup_proportion = 0.1
    learning_rate = 1e-5
    adam_epsilon = 1e-6
    seed = 1234
    embedding_dim = 768
    hidden_dim = 128
    rnn_layers = 1
    lstm_dropout_ratio = 0.1
    dropout_ratio = 0.1
    logging_steps = 20
    max_history = 300
    max_response = 50

    

# pre-train
class PreTrainConfig:
    model_name_or_path = '../chinese-roberta-wwm-ext'
    learning_rate = 1e-5
    adam_epsilon = 1e-8
    pre_train_epochs = 1
    grad_accum = 2
    max_seq_length = 500
    max_grad_norm = 1.0
    train_batch_size = 10
    logging_steps = 100
    output_dir = './chinese-roberta-wwm-ext-pretrained'


if __name__ == '__main__':
    config = Config()
    print(config.max_history)
    # import numpy as np
    # pred_tag = [0.10395336151123047, 0.12665358185768127, 0.04394872486591339, 0.038305968046188354, 0.008290484547615051,
    #  0.056823063641786575, 0.0708710178732872, 0.04301300644874573, 0.08311615884304047, 0.06961376219987869,
    #  -0.009892445057630539, -0.012072302401065826, -0.017415516078472137, 0.02824031561613083, -0.011212978512048721,
    #  0.012474387884140015, 0.007045187056064606, -0.011557381600141525, -0.01093737781047821, -0.007164008915424347,
    #  -0.002002079039812088, 0.003886282444000244, -0.012671604752540588, -0.009088821709156036, 0.016604673117399216,
    #  0.023836523294448853, 0.035443030297756195, 0.008360665291547775, 0.07397391647100449, 0.006389342248439789]
    # true_tag = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #
    # pred_tag_bool = [i>0 for i in pred_tag]
    # print(pred_tag_bool)
    # print(np.mean(np.equal(pred_tag_bool, true_tag)))
    #
    
    