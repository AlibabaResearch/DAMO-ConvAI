echo '''injecting syntax relations into graphs'''
sh data_all_in/run/run_syntax.sh

echo '''preprocessing training set'''
sh data_all_in/run/run_preprocessing_training.sh

echo '''preprocessing dev set'''
sh data_all_in/run/run_preprocessing_dev.sh

echo '''composing graph pedia for both train and dev'''
sh data_all_in/run/run_graph_all.sh