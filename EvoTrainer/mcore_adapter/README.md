# MCoreAdapter

MCoreAdapter is a lightweight bridge toolkit for scalable LLM/VLM training, combining NVIDIA Megatron-LM's distributed training efficiency with HuggingFace Transformers-like API simplicity.

Developed as Roll Framework's Megatron-LM integration layer, it enables seamless interoperability between Roll's reinforcement learning workflows and Megatron's distributed training capabilities.

## Installation

```bash
pip install "git+https://github.com/alibaba/roll.git#subdirectory=mcore_adapter"
```

## Usage

Except reinforcement learning with Roll, MCoreAdapter can also be applied for LLMs and VLMs in PreTraining, SFT and DPO/ORPO.

See [examples](examples/README.md) for fine-tuning examples used [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) library.


### Convert between HuggingFace and Megatron

Convert a Megatron model to HuggingFace model:
```bash
python tools/convert.py --checkpoint_path path_to_megatron_model --output_path path_to_output_hf_model
```

MCoreAdapter can directly load a HuggingFace model, so you can skip converting the model to Megatron.
