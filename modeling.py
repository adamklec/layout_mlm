from dataclasses import dataclass
from typing import Optional, Tuple

import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers import LayoutLMv3PreTrainedModel, LayoutLMv3Model
from transformers.activations import ACT2FN
from transformers.models.layoutlmv3.modeling_layoutlmv3 import LayoutLMv3ClassificationHead
from transformers.utils import logging, ModelOutput

logger = logging.get_logger(__name__)


@dataclass
class LayoutLMv3PretrainingOutput(ModelOutput):
    """
    Base class for outputs of token classification models.

    Args:
        mlm_loss
        mim_loss
        wpa_loss
        sd_loss
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    mlm_loss: Optional[torch.FloatTensor] = None
    mim_loss: Optional[torch.FloatTensor] = None
    wpa_loss: Optional[torch.FloatTensor] = None
    sd_loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# Copied from transformers.models.bert.modeling_bert.BertPredictionHeadTransform with Bert->LayoutLMv3
class LayoutLMv3PredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLMPredictionHead with Bert->LayoutLMv3
class LayoutLMv3MLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = LayoutLMv3PredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLMPredictionHead with Bert->LayoutLMv3
class LayoutLMv3MIMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = LayoutLMv3PredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.codebook_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.codebook_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class LayoutLMv3MLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = LayoutLMv3MLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class LayoutLMv3SDHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = LayoutLMv3MLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class LayoutLMv3MIMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = LayoutLMv3MIMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class LayoutLMv3ForPretraining(LayoutLMv3PreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.layoutlmv3 = LayoutLMv3Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.text_classifier = LayoutLMv3MLMHead(config)
        self.image_classifier = LayoutLMv3MIMHead(config)
        self.wpa_classifier = nn.Linear(config.hidden_size, 2)
        self.sd_classifier = nn.Bilinear(config.hidden_size, config.hidden_size, 1)

        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        text_labels: Optional[torch.LongTensor] = None,
        image_labels: Optional[torch.LongTensor] = None,
        wpa_labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.LongTensor] = None,
        doc_id: Optional[torch.LongTensor] = None,
    ) -> LayoutLMv3PretrainingOutput:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlmv3(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
        )
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        text_sequence_output = outputs.last_hidden_state[:, :seq_length]
        text_sequence_output = self.dropout(text_sequence_output)

        image_sequence_output = outputs.last_hidden_state[:, seq_length:]
        image_sequence_output = self.dropout(image_sequence_output)

        cls_encoding = outputs.last_hidden_state[:, 0]
        sd_labels = (doc_id.unsqueeze(0) == doc_id.unsqueeze(1)).float()

        batch_size = sd_labels.shape[0]
        n_pairs = batch_size * (batch_size - 1) / 2
        # TODO register buffer?
        bce_weight = torch.tril(torch.ones_like(sd_labels), -1) / n_pairs

        text_logits = self.text_classifier(text_sequence_output)
        image_logits = self.image_classifier(image_sequence_output)
        wpa_logits = self.wpa_classifier(text_sequence_output)

        # TODO verify
        sd_logits = self.sd_classifier(
            cls_encoding.unsqueeze(0).repeat_interleave(batch_size, 0),
            cls_encoding.unsqueeze(1).repeat_interleave(batch_size, 1)).squeeze()

        # TODO functional?
        loss_fct = CrossEntropyLoss()
        binary_loss_fct = BCEWithLogitsLoss(weight=bce_weight.view(-1))

        mlm_loss = loss_fct(text_logits.view(-1, self.config.vocab_size), text_labels.view(-1))
        mim_loss = loss_fct(image_logits.view(-1, self.config.codebook_size), image_labels.view(-1))
        wpa_loss = loss_fct(wpa_logits.view(-1, 2), wpa_labels.view(-1))
        sd_loss = binary_loss_fct(sd_logits.view(-1), sd_labels.view(-1))

        return LayoutLMv3PretrainingOutput(
            mlm_loss=mlm_loss,
            mim_loss=mim_loss,
            wpa_loss=wpa_loss,
            sd_loss=sd_loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
