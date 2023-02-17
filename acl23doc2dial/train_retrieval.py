import os
import json

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.msdatasets import MsDataset
from modelscope.trainers.nlp.document_grounded_dialog_retrieval_trainer import \
    DocumentGroundedDialogRetrievalTrainer
from modelscope.utils.constant import DownloadMode

fr_train_dataset = MsDataset.load(
    'DAMO_ConvAI/FrDoc2BotRetrieval',
    download_mode=DownloadMode.FORCE_REDOWNLOAD)

vi_train_dataset = MsDataset.load(
    'DAMO_ConvAI/ViDoc2BotRetrieval',
    download_mode=DownloadMode.FORCE_REDOWNLOAD)

train_dataset = [x for dataset in [fr_train_dataset, vi_train_dataset] for x in dataset]

all_passages = []
for file_name in ['fr', 'vi']:
    with open(f'all_passages/{file_name}.json') as f:
        all_passages += json.load(f)

cache_path = snapshot_download('DAMO_ConvAI/nlp_convai_retrieval_pretrain', cache_dir='./')
trainer = DocumentGroundedDialogRetrievalTrainer(
    model=cache_path,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
    all_passages=all_passages)
trainer.train(
    batch_size=128,
    total_epoches=50,
)
trainer.evaluate(
    checkpoint_path=os.path.join(trainer.model.model_dir,
                                 'finetuned_model.bin'))
