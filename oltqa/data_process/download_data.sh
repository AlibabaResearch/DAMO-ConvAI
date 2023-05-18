BASE_URL=https://storage.googleapis.com/danielk-files/data
# datasets for pretraining
for dataset in squad1_1 squad2  newsqa; do
    for data_type in train dev test; do
        wget  ${BASE_URL}/${dataset}/${data_type}_ans.jsonl -O data/${dataset}/${data_type}_ans.json
    done
done


#for dataset in  prost_multiple_choice_with_no_context; do
#    mkdir -p data/${dataset}
#    for data_type in train dev test ; do
#        wget ${BASE_URL}/${dataset}/${data_type}.tsv -O data/${dataset}/${data_type}.tsv
#    done
#done

#arc_easy_with_ir arc_hard_with_ir  contrast_sets_boolq contrast_sets_drop      contrast_sets_quoref      contrast_sets_ropes      race_string      commonsenseqa      arc_hard      arc_easy      mctest_corrected_the_separator      natural_questions      quoref      squad1_1      squad2      boolq      multirc      newsqa      ropes      ropes_test      drop      narrativeqa      openbookqa      qasc      boolq_np      arc_hard_dev      arc_easy_dev      qasc_test      openbookqa_dev      narrativeqa_dev      commonsenseqa_test      qasc_with_ir      qasc_with_ir_test      openbookqa_with_ir      openbookqa_with_ir_dev      arc_easy_with_ir_dev      arc_hard_with_ir_dev      race_string_dev      ambigqa      natural_questions_with_dpr_para      natural_questions_direct_ans_test winogrande_xl      social_iqa      social_iqa_test      physical_iqa      physical_iqa_test      adversarialqa_dbert_dev      adversarialqa_dbert_test      adversarialqa_dbidaf_dev      adversarialqa_dbidaf_test      adversarialqa_droberta_dev      adversarialqa_droberta_test      aqua_rat_dev      aqua_rat_test      codah_dev      codah_test      head_qa_en_dev      head_qa_en_test      processbank_test      csqa2      strategyqa      pubmedqa_pqal_short_ans reclor      race_c      quail      onestopqa_elementry      onestopqa_intermediate      onestopqa_advanced      mcscript      mcscript2      record_extractive      record_multiple_choice      cosmosqa      tweetqa      measuring_massive_multitask_language_understanding      dream      qaconv #
#
#exit
#
#wget https://nlp.cs.washington.edu/ambigqa/data/nqopen.zip -O data/nqopen.zip
#unzip -d data data/nqopen.zip
#rm data/nqopen.zip

