task=lgesql_large
seed=999
device=0
testing='' #'--testing'
read_model_path=''

model=lgesql
output_model=with_pruning # without_pruning
local_and_nonlocal=$1 # mmc, msde, local
plm=$2
subword_aggregation=attentive-pooling
schema_aggregation=head+tail
gnn_hidden_size=512
gnn_num_layers=8
relation_share_heads='--relation_share_heads'
score_function='affine'
num_heads=8
dropout=0.2
attn_drop=0.0
drop_connect=0.2

lstm=onlstm
chunk_size=8
att_vec_size=512
sep_cxt=''
lstm_hidden_size=512
lstm_num_layers=1
action_embed_size=128
field_embed_size=64
type_embed_size=64
no_context_feeding='--no_context_feeding'
no_parent_production_embed=''
no_parent_field_embed=''
no_parent_field_type_embed=''
no_parent_state=''

batch_size=20
grad_accumulate=5
lr=1e-4
layerwise_decay=0.8
l2=0.1
smoothing=0.15
warmup_ratio=0.1
lr_schedule=linear
eval_after_epoch=120
max_epoch=200
max_norm=5
beam_size=5

python scripts/text2sql.py --task $task --seed $seed --device $device $testing $read_model_path \
    --plm $plm --gnn_hidden_size $gnn_hidden_size --dropout $dropout --attn_drop $attn_drop --att_vec_size $att_vec_size \
    --model $model --output_model $output_model --local_and_nonlocal $local_and_nonlocal --score_function $score_function $relation_share_heads \
    --subword_aggregation $subword_aggregation --schema_aggregation $schema_aggregation --gnn_num_layers $gnn_num_layers --num_heads $num_heads $sep_cxt \
    --lstm $lstm --chunk_size $chunk_size --drop_connect $drop_connect --lstm_hidden_size $lstm_hidden_size --lstm_num_layers $lstm_num_layers \
    --action_embed_size $action_embed_size --field_embed_size $field_embed_size --type_embed_size $type_embed_size \
    $no_context_feeding $no_parent_production_embed $no_parent_field_embed $no_parent_field_type_embed $no_parent_state \
    --batch_size $batch_size --grad_accumulate $grad_accumulate --lr $lr --l2 $l2 --warmup_ratio $warmup_ratio --lr_schedule $lr_schedule --eval_after_epoch $eval_after_epoch \
    --smoothing $smoothing --layerwise_decay $layerwise_decay --max_epoch $max_epoch --max_norm $max_norm --beam_size $beam_size   
