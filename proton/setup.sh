conda create -n text2sql python=3.6
source activate text2sql
pip install torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
python -c "import stanza; stanza.download('en')"
python -c "from embeddings import GloveEmbedding; emb = GloveEmbedding('common_crawl_48', d_emb=300)"
python -c "import nltk; nltk.download('stopwords')"
mkdir -p pretrained_models && cd pretrained_models
git lfs install
git clone https://huggingface.co/bert-large-uncased-whole-word-masking
git clone https://huggingface.co/google/electra-large-discriminator
mkdir -p glove.42b.300d && cd glove.42b.300d
wget -c http://nlp.stanford.edu/data/glove.42B.300d.zip && unzip glove.42B.300d.zip
awk -v FS=' ' '{print $1}' glove.42B.300d.txt > vocab_glove.txt