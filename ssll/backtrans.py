from transformers import MarianMTModel, MarianTokenizer
import torch

# target_model_name = 'Helsinki-NLP/opus-mt-en-ROMANCE'
# tgt_tokz_path='out_BT/tgt_tokz'
# tgt_model_path='out_BT/tgt_model'
# target_tokenizer = MarianTokenizer.from_pretrained(tgt_tokz_path)
# target_model = MarianMTModel.from_pretrained(tgt_model_path)

# # en_model_name = 'Helsinki-NLP/opus-mt-ROMANCE-en'
# en_tokz_path='out_BT/en_tokz'
# en_model_path='out_BT/en_model'
# en_tokenizer = MarianTokenizer.from_pretrained(en_tokz_path)
# en_model = MarianMTModel.from_pretrained(en_model_path)

# target_tokenizer.save_pretrained('out_BT/tgt_tokz.pt')
# target_model.save_pretrained('out_BT/tgt_model.pt')
# en_tokenizer.save_pretrained('out_BT/en_tokz.pt')
# en_model.save_pretrained('out_BT/en_model.pt')

# target_model, en_model = target_model.to('cuda'), en_model.to('cuda')

def translate(texts, model, tokenizer, language="fr"):
    # Prepare the text data into appropriate format for the model
    def template(
        text): return f"{text}" if language == "en" else f">>{language}<< {text}"
    src_texts = [template(text) for text in texts]

    # Tokenize the texts
    encoded = tokenizer.prepare_seq2seq_batch(src_texts)
    # input_ids = tokenizer.encode(src_texts)
    for k, v in encoded.items():
        encoded[k] = torch.tensor(v)
    encoded = encoded.to('cuda')  

    # Generate translation using model
    translated = model.generate(**encoded)

    # Convert the generated tokens indices back into text
    translated_texts = tokenizer.batch_decode(
        translated, skip_special_tokens=True)

    return translated_texts


def back_translate(texts, target_model, target_tokenizer, en_model, en_tokenizer, source_lang="en", target_lang="fr"):
    # Translate from source to target language
    fr_texts = translate(texts, target_model, target_tokenizer,
                         language=target_lang)

    # Translate from target language back to source language
    back_translated_texts = translate(fr_texts, en_model, en_tokenizer,
                                      language=source_lang)

    return back_translated_texts

# en_texts = ['This is so cool', 'I hated the food', 'They were very helpful']
# # en_texts = 'I love you.'
# aug_texts = back_translate(en_texts, target_model, target_tokenizer, en_model, en_tokenizer, source_lang="en", target_lang="fr")
# print(aug_texts)
