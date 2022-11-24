import random

seed_list = [32, 42, 52]
intent_list = ['atis', 'banking', 'clinc', 'hwu', 'snips', 'top_split1', 'top_split2' ,'top_split3']
res_order=open('six_order_intent.txt','w')
prelist = ['atis', 'dstc8', 'mit_movie_eng', 'snips', 'mit_movie_trivia', 'mit_restaurant']
# prelist=['MWOZ_attraction','MWOZ_hotel','MWOZ_restaurant','MWOZ_taxi','MWOZ_train',
        # 'sgd_alarm','sgd_banks','sgd_buses','sgd_calendar','sgd_events','sgd_flights','sgd_homes','sgd_hotels','sgd_media','sgd_movies','sgd_music','sgd_payment','sgd_rentalcars','sgd_restaurants',
        # 'sgd_ridesharing','sgd_services','sgd_trains','sgd_travel','sgd_weather',
        # 'TMA_auto','TMA_coffee','TMA_movie','TMA_pizza','TMA_restaurant','TMA_uber',
        # 'TMB_flight','TMB_food-ordering','TMB_hotel','TMB_movie','TMB_music','TMB_restaurant','TMB_sport']
# res_order=open('big_order.txt','w')

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
    