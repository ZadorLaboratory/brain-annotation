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
import torch.nn.functional as F

@dataclass
class SequenceClassifierOutputWithSingleCell(SequenceClassifierOutput):
    single_cell_logits: Optional[torch.Tensor] = None

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
        single_cell_vs_group_weight: Union[str, float] = 0.5,
        detach_bert_embeddings: bool = False,
        single_cell_loss_after_set: bool = False,
        use_relative_positions: bool = False,
        position_encoding_dim: int = 32,  # Must ensure (set_hidden_size + position_encoding_dim) is divisible by num_attention_heads
        position_encoding_type: str = "mlp",
        rms_layernorm: bool = False,
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
        self.single_cell_vs_group_weight = single_cell_vs_group_weight
        self.detach_bert_embeddings = detach_bert_embeddings
        self.single_cell_loss_after_set = single_cell_loss_after_set
        self.use_relative_positions = use_relative_positions
        self.position_encoding_dim = position_encoding_dim
        self.position_encoding_type = position_encoding_type
        self.rms_layernorm = rms_layernorm

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
        if config.rms_layernorm:
            self.layer_norm1 = nn.RMSNorm(config.set_hidden_size) 
            self.layer_norm2 = nn.RMSNorm(config.set_hidden_size)
        else:
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

class SinusoidalEncoder(nn.Module):
    """Encode positions using sinusoidal embeddings"""
    def __init__(self, output_dim: int):
        super().__init__()
        assert output_dim % 6 == 0, f"output_dim must be divisible by 6 for sinusoidal encoding. Got {output_dim}"
        self.dim_per_component = output_dim // 3  # 2 components each for r, theta, z
        self.frequencies = torch.exp(
            torch.arange(0, self.dim_per_component, 2) * -(4.605 / self.dim_per_component)
        )
    
    def _sinusoidal_embedding(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, num_points)
        freqs = self.frequencies.to(x.device)
        emb = x.unsqueeze(-1) * freqs
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        # Convert x,y to polar coordinates
        x, y, z = positions[..., 0], positions[..., 1], positions[..., 2]
        r = torch.sqrt(x**2 + y**2)
        theta = torch.atan2(y, x)
        
        # Generate embeddings for each component
        r_emb = self._sinusoidal_embedding(r)
        theta_emb = self._sinusoidal_embedding(theta)
        z_emb = self._sinusoidal_embedding(z)
        
        # Concatenate all embeddings
        return torch.cat([r_emb, theta_emb, z_emb], dim=-1)

class PositionalEncoder(nn.Module):
    """Encode 3D positions into higher dimensional space"""
    def __init__(self, output_dim: int, encoding_type: str = "mlp"):
        super().__init__()
        self.encoding_type = encoding_type
        
        if encoding_type == "mlp":
            self.encoder = nn.Sequential(
                nn.Linear(3, output_dim // 2),
                nn.ReLU(),
                nn.Linear(output_dim // 2, output_dim),
                nn.LayerNorm(output_dim)
            )
        elif encoding_type == "sinusoidal":
            self.encoder = SinusoidalEncoder(output_dim)
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.encoder(positions)

class HierarchicalBert(BertPreTrainedModel):
    config_class = HierarchicalBertConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: HierarchicalBertConfig):
        super().__init__(config)
        
        # Initialize BERT 
        if isinstance(config.bert_config, BertConfig):
            self.bert = BertModel(config.bert_config)
        else:
            # Handle the case where bert_config is a dict
            self.bert = BertModel(BertConfig(**config.bert_config))
            self.config.bert_config = self.bert.config

        # Position handling and calculating final dimension size for set transformer
        self.use_relative_positions = config.use_relative_positions
        self.original_set_dim = final_dim = config.set_hidden_size
        if self.use_relative_positions:
            self.position_encoder = PositionalEncoder(
                config.position_encoding_dim,
                encoding_type=getattr(config, 'position_encoding_type', 'mlp')
            )
            # Adjust set transformer input dimension
            final_dim = config.set_hidden_size + config.position_encoding_dim
            if final_dim % config.num_attention_heads != 0:
                raise ValueError(
                    f"Combined dimension (set_hidden_size + position_encoding_dim = {final_dim}) "
                    f"must be divisible by num_attention_heads ({config.num_attention_heads})"
                )
            config.set_hidden_size = final_dim

        # Initialize set transformer layers
        self.set_layers = nn.ModuleList([
            SetTransformerLayer(config)
            for _ in range(config.num_set_layers)
        ])
        
        # Project if needed
        if config.bert_config.hidden_size != self.original_set_dim:
            self.hidden_projection = nn.Linear(
                config.bert_config.hidden_size,
                self.original_set_dim
            )
        else:
            self.hidden_projection = nn.Identity()
            
        # Classification head
        self.classifier = nn.Linear(config.set_hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.dropout_prob)

        # Single-cell classification head
        if config.single_cell_loss_after_set:
            self.single_cell_classifier = nn.Linear(final_dim, config.num_labels)
        else:
            self.single_cell_classifier = nn.Identity()
        
        # Store class weights
        self.class_weights = config.class_weights

        self.single_cell_vs_group_weight = config.single_cell_vs_group_weight
        self.detach_bert_embeddings = config.detach_bert_embeddings
        self.single_cell_loss_after_set = config.single_cell_loss_after_set

        if self.single_cell_vs_group_weight < 0 or self.single_cell_vs_group_weight > 1:
            raise ValueError("single_cell_vs_group_weight must be in range [0, 1]")        

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
            relative_positions: Optional[torch.Tensor] = None,
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

        # Potentially detach embeddings
        if self.detach_bert_embeddings:
            sentence_embeddings = sentence_embeddings.detach()

        # Project if necessary
        sentence_embeddings = self.hidden_projection(sentence_embeddings)

        # Reshape to (batch_size, num_sentences, hidden_size) 
        sentence_embeddings = sentence_embeddings.view(batch_size, num_sentences, -1)
        
        # Handle relative positions if enabled
        if self.use_relative_positions and relative_positions is not None:
            position_features = self.position_encoder(relative_positions)
                
            # Concatenate position features with sentence embeddings
            sentence_embeddings = torch.cat([sentence_embeddings, position_features], dim=-1)

        # Process through Set Transformer layers
        hidden_states = sentence_embeddings
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        
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
        if self.single_cell_loss_after_set:
            hidden_states_reshaped = hidden_states.view(batch_size*num_sentences, -1)
            single_cell_logits = self.single_cell_classifier(hidden_states_reshaped)    

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

            if self.single_cell_loss_after_set:
                single_cell_loss = loss_fct(single_cell_logits.view(-1, self.config.num_labels), single_cell_labels.view(-1))
                loss = self.single_cell_vs_group_weight * single_cell_loss + (1 - self.single_cell_vs_group_weight) * loss
        
        if not return_dict:
            output = (logits,) + (hidden_states,)
            if output_hidden_states:
                output = output + (all_hidden_states,)
            if output_attentions:
                output = output + (all_self_attentions,)
            return ((loss,) + output) if loss is not None else output
            
        return SequenceClassifierOutputWithSingleCell(
                loss=loss,
                logits=logits,
                hidden_states=all_hidden_states if output_hidden_states else None,
                attentions=all_self_attentions if output_attentions else None,
                single_cell_logits=single_cell_logits if self.single_cell_loss_after_set else None
            )
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (BertModel, SetTransformerLayer)):
            module.gradient_checkpointing = value
