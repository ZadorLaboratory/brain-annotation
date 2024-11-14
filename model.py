from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import (
    PreTrainedModel,
    BertPreTrainedModel,
    BertModel, 
    BertConfig,
    PretrainedConfig
)

@dataclass
class HierarchicalBertConfig(PretrainedConfig):
    """Configuration class for HierarchicalBert."""
    
    def __init__(
        self,
        num_labels: int = 2,
        bert_config: Optional[Union[BertConfig, str]] = None,
        num_set_layers: int = 2,
        set_hidden_size: int = 768,
        num_attention_heads: int = 8,
        dropout_prob: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
        pool_weight: Union[str, float] = 0.5,
        single_cell_augmentation: bool = False,
        detach_bert_embeddings: bool = False,
        detach_single_cell_logits: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        # Load BERT config if string is provided
        if isinstance(bert_config, str):
            self.bert_config = BertConfig.from_pretrained(bert_config)
        else:
            self.bert_config = bert_config or BertConfig()
            
        self.num_labels = num_labels
        self.num_set_layers = num_set_layers
        self.set_hidden_size = set_hidden_size
        self.num_attention_heads = num_attention_heads
        self.dropout_prob = dropout_prob
        self.class_weights = class_weights
        self.pool_weight = pool_weight
        self.single_cell_augmentation = single_cell_augmentation
        self.detach_bert_embeddings = detach_bert_embeddings
        self.detach_single_cell_logits = detach_single_cell_logits

class SetTransformerLayer(nn.Module):
    """Simple Set Transformer layer."""
    
    def __init__(self, config: HierarchicalBertConfig):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=config.set_hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.dropout_prob,
            batch_first=True
        )
        self.layer_norm1 = nn.LayerNorm(config.set_hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.set_hidden_size)
        self.feedforward = nn.Sequential(
            nn.Linear(config.set_hidden_size, config.set_hidden_size * 4),
            nn.GELU(),
            nn.Dropout(config.dropout_prob),
            nn.Linear(config.set_hidden_size * 4, config.set_hidden_size),
            nn.Dropout(config.dropout_prob)
        )

    def forward(
        self,
        x: torch.Tensor,
        output_attentions: Optional[bool] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Ensure output_attentions is boolean
        need_weights = bool(output_attentions) if output_attentions is not None else False
        
        # Self-attention
        attended, attn_weights = self.attention(x, x, x, need_weights=need_weights)
        x = self.layer_norm1(x + attended)
        
        # Feedforward
        ff_output = self.feedforward(x)
        x = self.layer_norm2(x + ff_output)
        
        return x, attn_weights if need_weights else None

class HierarchicalBert(BertPreTrainedModel):

    config_class = HierarchicalBertConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: HierarchicalBertConfig):
        super().__init__(config)
        
        # Initialize BERT
        self.bert = BertModel(config.bert_config)
        
        # Initialize set transformer layers
        self.set_layers = nn.ModuleList([
            SetTransformerLayer(config)
            for _ in range(config.num_set_layers)
        ])
        
        # Project if needed
        if config.bert_config.hidden_size != config.set_hidden_size:
            self.hidden_projection = nn.Linear(
                config.bert_config.hidden_size,
                config.set_hidden_size
            )
        else:
            self.hidden_projection = nn.Identity()
            
        # Classification head
        self.classifier = nn.Linear(config.set_hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.dropout_prob)

        # Single-cell classification head
        if config.single_cell_augmentation or config.also_single_cell_loss:
            self.single_cell_classifier = nn.Linear(config.bert_config.hidden_size, config.num_labels)
        else:
            self.single_cell_classifier = nn.Identity()
        
        # Store class weights
        self.class_weights = config.class_weights

        if config.pool_weight == "learned":
            self.pool_weight = nn.Parameter(torch.ones(1)*0.5, requires_grad=True)
            raise NotImplementedError("Learned pooling weights are not yet implemented")
        else:
            self.pool_weight = torch.tensor(config.pool_weight)
            self.pool_weight.requires_grad = False

        self.single_cell_augmentation = config.single_cell_augmentation
        self.detach_bert_embeddings = config.detach_bert_embeddings
        self.detach_single_cell_logits = config.detach_single_cell_logits
        self.also_single_cell_loss = config.also_single_cell_loss
        assert not (self.also_single_cell_loss and self.single_cell_augmentation), \
            "also_single_cell_loss can only be used without single_cell_augmentation"

        if self.pool_weight < 0 or self.pool_weight > 1:
            raise ValueError("pool_weight must be in range [0, 1]")        
        
        
    def _init_weights(self, module):
        """Initialize the weights - called by BertPreTrainedModel"""
        if isinstance(module, nn.Linear):
            # Initialize linear layers (non-BERT ones)
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            single_cell_labels: Optional[torch.Tensor] = None,
            indices: Optional[torch.Tensor] = None,
            ) -> Union[Tuple, SequenceClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else False

        batch_size, num_sentences, seq_length = input_ids.shape

        assert input_ids.shape == (batch_size, num_sentences, seq_length), \
            f"Expected input_ids shape (batch_size, num_sentences, seq_length), got {input_ids.shape}"
        if attention_mask is not None:
            assert attention_mask.shape == input_ids.shape, \
                f"Expected attention_mask shape {input_ids.shape}, got {attention_mask.shape}"
        if labels is not None:
            assert labels.shape == (batch_size,), \
                f"Expected labels shape (batch_size,), got {labels.shape}"
        
        # Reshape for BERT processing
        flat_input_ids = input_ids.view(-1, seq_length)
        flat_attention_mask = attention_mask.view(-1, seq_length) if attention_mask is not None else None
        flat_token_type_ids = token_type_ids.view(-1, seq_length) if token_type_ids is not None else None
        flat_position_ids = position_ids.view(-1, seq_length) if position_ids is not None else None
        
        # Process through BERT
        bert_outputs = self.bert(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids,
            position_ids=flat_position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
            
        # Get CLS tokens and reshape
        sentence_embeddings = bert_outputs[1]  # Pooled output

        # Get single-cell classifications from base BERT?
        if self.single_cell_augmentation:
            single_cell_logits = self.single_cell_classifier(sentence_embeddings) # shape (batch_size * num_sentences, num_labels)

        # Potentially detach embeddings
        if self.detach_bert_embeddings:
            sentence_embeddings = sentence_embeddings.detach()

        # Reshape to (batch_size, num_sentences, hidden_size) 
        sentence_embeddings = sentence_embeddings.view(batch_size, num_sentences, -1)
        assert sentence_embeddings.shape == (batch_size, num_sentences, bert_outputs.last_hidden_state.size(-1)), \
            f"Expected sentence_embeddings shape (batch_size, num_sentences, hidden_size), got {sentence_embeddings.shape}"
        
        # Project if necessary
        sentence_embeddings = self.hidden_projection(sentence_embeddings)
        assert sentence_embeddings.shape == (batch_size, num_sentences, self.config.set_hidden_size), \
            f"Expected projected embeddings shape (batch_size, num_sentences, set_hidden_size), got {sentence_embeddings.shape}"
        
        # Process through Set Transformer layers
        hidden_states = sentence_embeddings
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        
        if self.pool_weight < 1:
            for layer in self.set_layers:
                hidden_states, attn_weights = layer(
                    hidden_states,
                    output_attentions=output_attentions
                )
                
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
                if output_attentions and attn_weights is not None:
                    all_self_attentions = all_self_attentions + (attn_weights,)
            
            # Pool sentences (mean pooling)
            pooled = torch.mean(hidden_states, dim=1)
            pooled = self.dropout(pooled)
            
            # Classification
            logits = self.classifier(pooled) # shape (batch_size, num_labels)

            # maybe try to classify each sentence (cell) separately
            if self.also_single_cell_loss:
                hidden_states_reshaped = hidden_states.view(batch_size*num_sentences, -1)
                single_cell_logits = self.single_cell_classifier(hidden_states_reshaped)
        else:
            logits = 0
            all_hidden_states = None
            all_self_attentions = None

        # For the final prediction, average in pooled single-cell logits
        if self.single_cell_augmentation:
            single_cell_logits_reshaped = self.dropout(single_cell_logits)
            single_cell_logits_reshaped = single_cell_logits_reshaped.view(batch_size, num_sentences, -1)
            pooled_single_cell_logits = torch.mean(single_cell_logits_reshaped, dim=1)
            if self.detach_single_cell_logits:
                pooled_single_cell_logits = pooled_single_cell_logits.detach()
            logits = pooled_single_cell_logits * self.pool_weight + logits * (1 - self.pool_weight)       

        loss = None
        if labels is not None:
            if labels.min() < 0 or labels.max() >= self.config.num_labels:
                raise ValueError(
                    f"Labels must be in range [0, {self.config.num_labels-1}], "
                    f"but found range [{labels.min().item()}, {labels.max().item()}]"
                )
            loss_fct = nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device) if self.class_weights is not None else None
            )
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

            if self.single_cell_augmentation or self.also_single_cell_loss:
                single_cell_loss = loss_fct(single_cell_logits.view(-1, self.config.num_labels), single_cell_labels.view(-1))
                loss += single_cell_loss

        
        if not return_dict:
            output = (logits,) + (hidden_states,)
            if output_hidden_states:
                output = output + (all_hidden_states,)
            if output_attentions:
                output = output + (all_self_attentions,)
            return ((loss,) + output) if loss is not None else output
            
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=all_hidden_states if output_hidden_states else None,
            attentions=all_self_attentions if output_attentions else None,
        )
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (BertModel, SetTransformerLayer)):
            module.gradient_checkpointing = value
