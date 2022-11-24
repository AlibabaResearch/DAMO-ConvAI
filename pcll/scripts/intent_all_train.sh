gpuid=$1
pseudo_ratio=0.2

exp=test_intent0
data_type="intent"
epoch=1
echo $exp

declare -a all_orders
# ord0=(hwu atis)
ord0=(top_split1 hwu snips banking clinc top_split2 top_split3 atis)
ord1=(banking hwu top_split1 top_split3 clinc top_split2 snips atis) 
ord2=(snips atis top_split2 top_split3 clinc banking hwu top_split1) 
ord3=(clinc snips top_split3 banking top_split2 hwu top_split1 atis)
ord4=(banking top_split2 top_split1 atis top_split3 hwu clinc snips)
ord5=(clinc top_split1 top_split2 atis snips hwu banking top_split3)

all_orders[0]=${ord0[@]}
all_orders[1]=${ord1[@]}
all_orders[2]=${ord2[@]}
all_orders[3]=${ord3[@]}
all_orders[4]=${ord4[@]}
all_orders[5]=${ord5[@]}


# test
# i=0
# sh scripts/train_one.sh $gpuid $exp ord_$i $data_type "${all_orders[$i]}" $epoch
# sh scripts/score_one.sh $exp ord_$i $data_type "${all_orders[$i]}" 

# for i in 0;
for i in 0 1 2 3 4 5; 
do
    echo 'BEGIN' $i '=========='
    echo "Order:" ${all_orders[$i]}
    sh scripts/train_one.sh $gpuid $exp ord_$i $data_type "${all_orders[$i]}" $epoch $pseudo_ratio
    echo "Finish training!"
    # sh scripts/score_one.sh $exp ord_$i $data_type "${all_orders[$i]}" 
    # echo "Finish scoring!" $i 
done
