import torch
from torch import nn
from utils import ATConfig
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from modeling_spectra.wavlm import WavLMMAMHead, WavLMForMultiTurn
from transformers.models.roberta.modeling_roberta import RobertaLMHead, RobertaModel, RobertaLayer


class ATModel(PreTrainedModel):
    config_class = ATConfig
    _keys_to_ignore_on_load_missing = ["mlm_head", "mam_head", "selection_head", "start_prediction_head", "end_prediction_head"]
    _keys_to_ignore_on_save = ["mlm_head", "mam_head", "selection_head", "start_prediction_head", "end_prediction_head"]

    def __init__(self, config: ATConfig, audio=None, text=None):
        super(ATModel, self).__init__(config)
        self.hidden_size = config.text.hidden_size
        if audio is None:
            self.audio_encoder = WavLMForMultiTurn(config.audio)
            self.text_encoder = RobertaModel(config.text)
        else:
            self.audio_encoder = WavLMForMultiTurn.from_pretrained(audio, config=config.audio)
            self.text_encoder = RobertaModel.from_pretrained(text, config=config.text)
        self.token_type_embeddings = nn.Embedding(2, self.hidden_size)
        self.fused_encoder = nn.ModuleList(RobertaLayer(config.text) for _ in range(config.text.num_fused_layers))

    def fuse_four(self, text, audio, bs, text_len, audio_len, token_type_ids=None):
        text = text.unsqueeze(2).repeat(1, 1, 2, 1, 1).view(4 * bs, text_len, -1)
        audio = audio.unsqueeze(1).repeat(1, 2, 1, 1, 1).view(4 * bs, audio_len, -1)
        fused_input = torch.cat([text, audio], dim=1)
        if token_type_ids is not None:
            fused_input += self.token_type_embeddings(token_type_ids)
        else:
            fused_input = fused_input.squeeze(-1)
        return fused_input

    def forward(self, audio_input, text_input, audio_attention_mask, text_attention_mask, bs, turn_id=None):
        device = audio_input.device
        # audio: 3B * 160000  text: 2B * 514  mlm_label: B * 514  turn_id: B * 514
        out = self.audio_encoder(audio_input, audio_attention_mask, bs, perform_mam=True,
                                 token_embedding=self.text_encoder.embeddings.token_type_embeddings)
        audio_features, audio_mask, mam_label, a_masked = out
        # audio_features: 2B * 200 * 768  audio_mask: 2B * 200  mam_label: B * 200  a_masked: B * 200
        text_features = self.text_encoder(text_input, text_attention_mask, token_type_ids=turn_id)[0]
        # text_features: 2B * 514 * 768
        bs, text_len = text_input.shape
        bs //= 2
        audio_features = audio_features.view(bs, 2, -1, self.hidden_size)
        text_features = text_features.view(bs, 2, text_len, self.hidden_size)
        audio_len = audio_features.shape[2]
        modal_ids = torch.zeros([bs * 4, text_len + audio_len], dtype=torch.long).to(device)
        modal_ids[:, text_len:] = 1
        fused_input = self.fuse_four(text_features, audio_features, bs, text_len, audio_len, modal_ids)
        fused_attention_mask = self.fuse_four(text_attention_mask, audio_mask, bs, text_len, audio_len)
        fused_attention_mask = (1.0 - fused_attention_mask[:, None, None, :]) * torch.finfo(text_features.dtype).min
        for layer in self.fused_encoder:
            fused_input = layer(fused_input, fused_attention_mask)[0]
        return fused_input, mam_label, a_masked


class ATForPreTraining(PreTrainedModel):
    config_class = ATConfig
    _keys_to_ignore_on_load_missing = ["mlm_head", "mam_head", "selection_head", "start_prediction_head", "end_prediction_head"]
    _keys_to_ignore_on_save = ["mlm_head", "mam_head", "selection_head", "start_prediction_head", "end_prediction_head"]

    def __init__(self, config: ATConfig, audio=None, text=None):
        super(ATForPreTraining, self).__init__(config)
        self.hidden_size = config.text.hidden_size
        self.model = ATModel(config, audio, text)
        self.mlm_head = RobertaLMHead(config.text)
        self.mam_head = WavLMMAMHead(self.hidden_size, config.audio.conv_dim[-1])
        self.selection_head = nn.Linear(self.hidden_size, 4)
        self.start_prediction_head = nn.Sequential(nn.Linear(self.hidden_size, 1))
        self.end_prediction_head = nn.Sequential(nn.Linear(self.hidden_size, 1))
        self.vocab_size = config.text.vocab_size
        self.conv_dim = config.audio.conv_dim[-1]
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, audio_input, text_input, audio_attention_mask=None, text_attention_mask=None, mlm_label=None,
                turn_id=None, start_valid=None, end_valid=None, starts=None, ends=None):
        bs, text_len = mlm_label.shape
        fused_input, mam_label, a_masked = self.model(audio_input, text_input, audio_attention_mask, text_attention_mask, bs, turn_id)
        fused_input = fused_input.view(bs, 4, -1, self.hidden_size)
        mam_pre = self.mam_head(fused_input[:, 0, text_len:])
        text_fused = fused_input[:, 0, :text_len]
        mlm_pre = self.mlm_head(text_fused)
        mlm_loss = self.ce(mlm_pre.view(-1, self.vocab_size), mlm_label.view(-1))  # 未mask的位置，label为-100。
        if torch.isnan(mlm_loss):
            mlm_loss = torch.tensor(0.0, device=text_input.device)
        mam_loss = torch.tensor(0.0, device=text_input.device)
        if torch.sum(a_masked[1]) != 0:
            l1 = torch.nn.L1Loss()
            mam_loss = l1(mam_pre.masked_select(a_masked[0]), mam_label.masked_select(a_masked[1]))
        response_select = self.selection_head(fused_input[:, :, 0].view(4 * bs, self.hidden_size))
        rs_loss = self.ce(response_select, torch.arange(4).to(mlm_pre.device).repeat(bs))
        words = text_fused.masked_select(start_valid.unsqueeze(-1)).view(-1, self.hidden_size)
        pred_start = self.start_prediction_head(words).squeeze(-1)
        words = text_fused.masked_select(end_valid.unsqueeze(-1)).view(-1, self.hidden_size)
        pred_end = self.end_prediction_head(words).squeeze(-1)
        span_loss = torch.mean(torch.pow(torch.cat([starts, ends]) - torch.cat([pred_start, pred_end]), 2))
        return mlm_loss, mam_loss, rs_loss, span_loss


class ATForSequenceClassification(PreTrainedModel):
    config_class = ATConfig
    _keys_to_ignore_on_load_missing = ["head"]

    def __init__(self, config: ATConfig, audio=None, text=None):
        super(ATForSequenceClassification, self).__init__(config)
        self.hidden_size = config.text.hidden_size
        self.model = ATModel(config, audio, text)
        hidden_size = config.text.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            ACT2FN['gelu'],
            nn.Linear(hidden_size, self.num_class))
        self.config = config
        if self.num_class == 1:
            self.loss_fct = nn.L1Loss()
        else:
            self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, audio_input, text_input, audio_attention_mask, text_attention_mask, turn_ids=None, labels=None):
        bs, text_len = text_input.shape[:2]
        audio_features, audio_mask = self.model.audio_encoder(audio_input, audio_attention_mask, bs, False, self.model.text_encoder.embeddings.token_type_embeddings if self.config.audio.multi_turn else None)
        text_features = self.model.text_encoder(text_input, text_attention_mask, token_type_ids=turn_ids)[0]
        modal_ids = torch.zeros([bs, text_len + audio_features.shape[1]], dtype=torch.long).to(text_input.device)
        modal_ids[:, text_len:] = 1
        fused_input = torch.cat([text_features, audio_features], dim=1) + self.model.token_type_embeddings(modal_ids)
        fused_attention_mask = torch.cat([text_attention_mask, audio_mask], dim=1).to(dtype=text_features.dtype)
        fused_attention_mask = (1.0 - fused_attention_mask[:, None, None, :]).to(dtype=text_features.dtype) * torch.finfo(text_features.dtype).min
        if hasattr(self.config.text, "num_fusion_layers") or hasattr(self.config.text, "num_fused_layers"):
            for layer in self.model.fused_encoder:
                fused_input = layer(fused_input, fused_attention_mask)[0]
        else:
            fused_input = self.model.fused_encoder(fused_input, fused_attention_mask)[0]
        fused_input = fused_input[:, 0]
        logits = self.head(fused_input).squeeze(1)
        if labels is None:
            return logits
        return logits, self.loss_fct(logits, labels)
