import sys

def round_for_list(x, precision):
    return [round(data, precision) for data in x]

def make_n_gram(words, n):
    lenw = len(words)
    words = ["BBBB"] * (n-1) + words + ["EEEE"] * (n-1)
#    print "9:", words
    ngrams = []

    for i in range(lenw):
        str = ""
        for j in range(n):
            str += words[i + j] + "_"
        str = str[:-1]
        ngrams.append(str)
        i += n
#    print "17:", ngrams
    return ngrams

def make_all_n_gram(datas):
    all_words = len(datas)
    all_ngram = [[] for i in range(4)]
    all_ngram_len = []
    all_ngram_score = []

    all_ngram[0] += make_n_gram(datas, 1)
    all_ngram[1] += make_n_gram(datas, 2)
    all_ngram[2] += make_n_gram(datas, 3)
    all_ngram[3] += make_n_gram(datas, 4)
#    print all_ngram[0]
#    print all_ngram[1]
#    print all_ngram[2]
#    print all_ngram[3]

    for i in range(4):
        all_ngram_len.append(get_uniq_length(all_ngram[i]))
        all_ngram_score.append(all_ngram_len[i] * 1.0 / all_words)

    return all_ngram, all_ngram_len, all_words, all_ngram_score

def get_uniq_length(ngram):
#    print ngram
#    print set(ngram)
    return len(list(set(ngram)))

def uniq_n_gram_length(words, n):
    ngrams = make_n_gram(words, n)
    return len(list(set(ngrams)))

def distinct0(words):
    dis = []

    dis.append(uniq_n_gram_length(words, 1))
    dis.append(uniq_n_gram_length(words, 2))
    dis.append(uniq_n_gram_length(words, 3))
    dis.append(uniq_n_gram_length(words, 4))

    word_count = len(words)
    if 0 == word_count:
        return [], 0, []
    dis_score = []

    for diss in dis:
        dis_score.append(diss * 1.0 / word_count)

    return dis, word_count, dis_score

def readinputall():
    datas = []
    for line in sys.stdin:
        #words = line.strip().split()
        words = line.strip("\n").replace(' </s>', '').split() 
        #print("words:", words, "words-1:", words[-1], file=sys.stderr)
        datas.append(words)
    return datas

def readinput(k):
    datas = []
    count = 0
    for line in sys.stdin:
        #words = line.strip().split()
        words = line.strip("\n").replace(' </s>', '').split() 
        if len(words) == 0:
            count = 0
            continue
        if count >= k:
            continue
        count += 1
            
        datas.append(words)
    return datas

def distinct(datas):
    dataall = []
    for data in datas:
        dataall += data
    ngram, dist_scores_unnorm, word_count, dist_scores = make_all_n_gram(dataall)

    dist_scores = round_for_list(dist_scores, 4)
    return dist_scores, word_count, dist_scores_unnorm

def main(argv):
    if len(argv) <= 1:
        datas = readinputall()
    else:
        k = int(argv[1])
        if k == -1:
            datas = readinputall()
        else:
            datas = readinput(k)
    dist_scores, word_count, dist_scores_unnorm = distinct(datas)
    print("dist 1~4(unnorm):", dist_scores_unnorm, \
          "word_count:", word_count, \
          "dist_score 1~4:", "\t".join(list(map(str, dist_scores))))

if "__main__" == __name__:
    main(sys.argv)
