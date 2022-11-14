import torch
import torch.nn as nn
from transformers.modeling_bart import BartForConditionalGeneration
from transformers.tokenization_bart import BartTokenizer
from generator.keywords.keywords import SKETCH_KEYWORDS, KEYWORDS
import logging
import os

logger = logging.getLogger(__name__)
WEIGHTS_NAME = "pytorch_model.bin"

def pad_and_tensorize_sequence(sequences, padding_value = None, tensorize = False):
    if tensorize:
        return torch.tensor(sequences, dtype=torch.long)
    max_size = max([len(sequence) for sequence in sequences])
    padded_sequences = []
    for sequence in sequences:
        padded_sequence = sequence + [padding_value] * (max_size - len(sequence))
        padded_sequences.append(padded_sequence)
    return torch.tensor(padded_sequences, dtype=torch.long)

class RelogicModel(nn.Module):
  """
  output: tuple: (loss, ) in training
  """
  def __init__(self, pretrain_config):
    super().__init__()
    self.bert = BartForConditionalGeneration.from_pretrained(pretrain_config)
    self.tokenizer = BartTokenizer.from_pretrained(pretrain_config)
    special_tokens_dict = {'additional_special_tokens': ['<SQL>', '<LOGIC>']}
    self.tokenizer.add_special_tokens(special_tokens_dict)
    self.bert.resize_token_embeddings(len(self.tokenizer))

  def forward(self, *input, **kwargs):
    input_ids = kwargs.pop("input_ids")
    pad_token_id = kwargs.pop("pad_token_id")
    attention_mask = (input_ids != pad_token_id).long()
    num_return_sequences = 5

    if self.training:
        output_ids = kwargs.pop('labels')
        y_ids = output_ids[:, :-1].contiguous()
        lm_labels = output_ids[:, 1:].clone()
        lm_labels[output_ids[:, 1:] == pad_token_id] = -100

        outputs = self.bert(input_ids,
                          attention_mask=attention_mask, decoder_input_ids=y_ids, lm_labels=lm_labels, )
        return (outputs[0],)

    else:
        reranker = kwargs.pop("reranker")
        label_eos_id = kwargs.pop("label_eos_id")
        label_bos_id = kwargs.pop("label_bos_id")
        label_padding_id = kwargs.pop("label_padding_id")
        
        if not reranker:
            generated_ids = self.bert.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=5,
            max_length=60,
            length_penalty=3.0,
            early_stopping=True,
            use_cache=True,
            decoder_start_token_id=label_bos_id,
            eos_token_id=label_eos_id,
            pad_token_id=label_padding_id
            )
            output_ids = kwargs.pop('labels')
            y_ids = output_ids[:, :-1].contiguous()
            lm_labels = output_ids[:, 1:].clone()
            lm_labels[output_ids[:, 1:] == pad_token_id] = -100

            outputs = self.bert(input_ids,
                              attention_mask=attention_mask, decoder_input_ids=y_ids, lm_labels=lm_labels, )
            return (outputs[0].detach(), generated_ids)
        else:
            generated_ids = self.bert.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=5,
            max_length=60,
            length_penalty=4.0,
            early_stopping=True,
            use_cache=True,
            decoder_start_token_id=label_bos_id,
            eos_token_id=label_eos_id,
            pad_token_id=label_padding_id,
            num_return_sequences=num_return_sequences
            )
            
            reranker_inputs = []
            for i in range(input_ids.shape[0]):
                input_logic = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
                for j in range(num_return_sequences):
                    pred_text = self.tokenizer.decode(generated_ids[num_return_sequences*i + j], skip_special_tokens=True)
                    reranker_input_token = [self.tokenizer.cls_token] + self.tokenizer.tokenize(input_logic) + [self.tokenizer.eos_token] + self.tokenizer.tokenize(pred_text) + [self.tokenizer.sep_token]

                    reranker_input_token_ids = self.tokenizer.convert_tokens_to_ids(reranker_input_token)
                    reranker_inputs.append(reranker_input_token_ids)
            
            reranker_input_ids = pad_and_tensorize_sequence(reranker_inputs, padding_value=self.tokenizer.pad_token_id)
            reranker_input_ids = reranker_input_ids.to(input_ids.device)
            reranker_labels = torch.zeros(reranker_input_ids.shape[0],dtype=torch.long).to(input_ids.device)
            
            rerank_inputs = {
                "encoder_input_ids":reranker_input_ids,
                "pad_token_id":pad_token_id,
                "labels":reranker_labels,
            }
            rerank_result = reranker(**rerank_inputs)
            
            seq_len = generated_ids.shape[-1]
            rerank_score = rerank_result[1]
            rerank_mat = rerank_score.view(-1, num_return_sequences)
            batch_num = rerank_mat.shape[0]
            max_rerank_id = rerank_mat.argmax(dim=1).view(batch_num, 1, 1).repeat(1, 1, seq_len)
            
            generated_ids = generated_ids.view(-1, num_return_sequences, seq_len)
            reranked_generated_ids = generated_ids.gather(1, max_rerank_id).squeeze()
            loss = torch.FloatTensor(0).to(generated_ids.device).detach()
            return (loss, reranked_generated_ids)



  def save_pretrained(self, save_directory):
    """ Save a model and its configuration file to a directory, so that it
        can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.

        Arguments:
            save_directory: directory to which to save.
    """
    assert os.path.isdir(
      save_directory
    ), "Saving path should be a directory where the model and configuration can be saved"

    # Only save the model itself if we are using distributed training
    model_to_save = self.module if hasattr(self, "module") else self

    # Attach architecture to the config
    # model_to_save.config.architectures = [model_to_save.__class__.__name__]

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(save_directory, WEIGHTS_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)

    logger.info("Model weights saved in {}".format(output_model_file))



