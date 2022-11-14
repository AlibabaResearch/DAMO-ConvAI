fairseq-generate data-bin/spider.sql-en \
	    --path checkpoints/spider_trans/checkpoint_best.pt \
	        --batch-size 128 --beam 5 --results-path res/spider_trans_best

fairseq-generate data-bin/spider.sql-en \
	            --path checkpoints/spider_trans/checkpoint_last.pt \
		                    --batch-size 128 --beam 5 --results-path res/spider_trans_last
