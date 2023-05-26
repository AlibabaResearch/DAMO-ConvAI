import random

# seed_list = [2, 102, 22, 32, 42]
seed_list = [2, 32, 42]
# intent_list = ['atis', 'banking', 'clinc', 'hwu', 'snips', 'top_split1', 'top_split2' ,'top_split3']
# res_order=open('six_order_intent.txt','w')
# prelist = ['atis', 'dstc8', 'mit_movie_eng', 'snips', 'mit_movie_trivia', 'mit_restaurant']
# prelist = ['yahoo','yelp','ag','dbpedia','amazon']
# prelist = ['srl','sst','wikisql','woz.en','squad']
prelist = ['srl','sst','wikisql','woz.en','squad','yahoo','yelp','ag','dbpedia','amazon']
# res_order=open('text_classification_order.txt','w')
res_order=open('mix_order.txt','w')

res_list = []
for i in range(len(seed_list)):
    random.seed(seed_list[i])
    random.shuffle(prelist)
    # print(intent_list)
    # res_list.append(intent_list)
    # print(intent_list,file=res_order)
    print('('+' '.join(prelist)+')',file=res_order)
    # print('\n',file=res_order)

res_order.close()
    