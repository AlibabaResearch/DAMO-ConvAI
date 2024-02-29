import huggingface_hub 
huggingface_hub.login("")
from huggingface_hub import snapshot_download
snapshot_download(repo_id="meta-llama/Llama-2-7b")