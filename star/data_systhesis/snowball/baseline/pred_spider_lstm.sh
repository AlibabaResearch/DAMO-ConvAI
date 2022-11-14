fairseq-generate data-bin/spider.sql-en \
	    --path checkpoints/spider_lstm/checkpoint_best.pt \
	        --batch-size 128 --beam 5 --results-path res/spider_lstm_best

fairseq-generate data-bin/spider.sql-en \
	            --path checkpoints/spider_lstm/checkpoint_last.pt \
		                    --batch-size 128 --beam 5 --results-path res/spider_lstm_last
