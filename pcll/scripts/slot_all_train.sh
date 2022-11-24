gpuid=$1
data_type="slot"
epoch=10
pseudo_ratio=0.2
exp=vae_slot0

echo $exp

declare -a all_orders
ord0=(mit_movie_eng dstc8 mit_restaurant snips atis)
ord1=(mit_movie_eng snips dstc8 mit_restaurant atis)
ord2=(atis mit_movie_eng dstc8 mit_restaurant snips)
ord3=(dstc8 mit_restaurant mit_movie_eng atis snips)
ord4=(mit_movie_eng atis snips mit_restaurant dstc8)
ord5=(snips atis mit_restaurant mit_movie_eng dstc8)

all_orders[0]=${ord0[@]}
all_orders[1]=${ord1[@]}
all_orders[2]=${ord2[@]}
all_orders[3]=${ord3[@]}
all_orders[4]=${ord4[@]}
all_orders[5]=${ord5[@]}

#* test
# i=0
# sh scripts/train_one.sh $gpuid $exp ord_$i $data_type "${all_orders[$i]}" $epoch $pseudo_ratio
# sh scripts/score_one.sh $exp ord_$i $data_type "${all_orders[$i]}" 

for i in 0 1 2 3 4 5; 
do
    echo 'BEGIN' $i '=========='
    echo "Order:" ${all_orders[$i]}
    sh scripts/train_one.sh $gpuid $exp ord_$i $data_type "${all_orders[$i]}" $epoch $pseudo_ratio
    echo "Finish training!"
    # sh scripts/score_one.sh $exp ord_$i $data_type "${all_orders[$i]}" 
    # echo "Finish scoring!" $i 
done
