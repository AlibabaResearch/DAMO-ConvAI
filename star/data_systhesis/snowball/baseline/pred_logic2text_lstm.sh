fairseq-generate data-bin/logic2text.sql-en \
	    --path checkpoints/logic2text_lstm/checkpoint_best.pt \
	        --batch-size 128 --beam 5 --results-path res/logic2text_lstm_best

fairseq-generate data-bin/logic2text.sql-en \
            --path checkpoints/logic2text_lstm/checkpoint_last.pt \
                --batch-size 128 --beam 5 --results-path res/logic2text_lstm_last
