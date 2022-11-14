import torch
import torch.nn as nn
from generator.keywords.keywords import SKETCH_KEYWORDS, KEYWORDS
from transformers import BartForSequenceClassification
from transformers.tokenization_bart import BartTokenizer
import logging
import os

logger = logging.getLogger(__name__)
WEIGHTS_NAME = "pytorch_model.bin"

class AdversarialModel(nn.Module):
  """
  output: tuple: (loss, ) in training
  """
  def __init__(self, config_name):
    super().__init__()
    self.bart = BartForSequenceClassification.from_pretrained(config_name)
    self.tokenizer = BartTokenizer.from_pretrained(config_name)
    special_tokens_dict = {'additional_special_tokens': ['<SQL>', '<LOGIC>']}
    self.tokenizer.add_special_tokens(special_tokens_dict)
    self.bart.resize_token_embeddings(len(self.tokenizer))
    self.prelu = nn.PReLU()
    self.fc = nn.Linear(3, 1)
    self.sigmoid = nn.Sigmoid()
    self.loss = nn.BCELoss()

  def label_smoothing(self, labels, epsilon=0.1):
    K = 2 # number of channels
    return ((1-epsilon) * labels) + (epsilon / K) 
 
  def forward(self, *input, **kwargs):
    encoder_inputs = kwargs.pop("encoder_input_ids").contiguous()
    labels = kwargs.pop('labels').unsqueeze(1)
    
    
    pad_token_id = kwargs.pop("pad_token_id")
    attention_mask = (encoder_inputs != pad_token_id).long()
    
    outputs = self.bart(encoder_inputs,
                      attention_mask=attention_mask)
#     for i in range(encoder_inputs.shape[0]):
#         print("Input", self.tokenizer.decode(encoder_inputs[i], skip_special_tokens=True))
    #3 logits -> 1 score
    #print("outputs", outputs)

#     score = self.prelu(outputs[0])
#     score = self.fc(score)
#     score = self.sigmoid(score)

    score = torch.sum(outputs[0], 1).view(-1, 1)
    score = self.sigmoid(score)
    
    labels = labels.float()
#     print("score", score.view(1, -1))
#     print("labels", labels.view(1, -1))
    labels = self.label_smoothing(labels)
    loss = self.loss(score, labels)
    
    
    
    if self.training:        
        return (loss, score)
    
    else:
        return (loss.detach(), score)

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



