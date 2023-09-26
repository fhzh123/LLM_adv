# Import PyTorch
import torch
import torch.nn as nn
from torch.nn import functional as F
# Import Huggingface
from transformers import BertModel, BertConfig

class TransformerModel(nn.Module):
    def __init__(self, label_num: int = 2, dropout: float = 0.3):
        super().__init__()

        """
        Initialize augmenter model

        Args:
            encoder_config (dictionary): encoder transformer's configuration
            d_latent (int): latent dimension size
            device (torch.device):
        Returns:
            log_prob (torch.Tensor): log probability of each word
            mean (torch.Tensor): mean of latent vector
            log_var (torch.Tensor): log variance of latent vector
            z (torch.Tensor): sampled latent vector
        """
        self.dropout = nn.Dropout(dropout)
        self.label_num = label_num

        self.model = BertModel.from_pretrained('bert-base-uncased')

        self.d_hidden = self.model.config.hidden_size
        self.d_embedding = int(self.d_hidden / 2)
        self.vocab_num = self.model.config.vocab_size

        # Linear model setting
        self.classifier = nn.Linear(self.d_hidden, self.label_num)

        # Adv model setting
        self.adv_linear = nn.Linear(self.d_hidden, self.d_embedding)
        self.adv_norm = nn.LayerNorm(self.d_embedding, eps=1e-12)
        self.adv_linear2 = nn.Linear(self.d_embedding, self.vocab_num)

    def encode(self, src_input_ids, src_attention_mask=None, src_token_type=None):
        if src_input_ids.dtype == torch.int64:
            encoder_out = self.model(input_ids=src_input_ids,
                                     attention_mask=src_attention_mask,
                                     token_type_ids=src_token_type)
        else:
            encoder_out = self.model(inputs_embeds=src_input_ids,
                                     attention_mask=src_attention_mask,
                                     token_type_ids=src_token_type)
        encoder_out = encoder_out['pooler_output'] # (batch_size, d_hidden)

        return encoder_out
    
    def pca_reduction(self, encoder_hidden_states):
        U, _, _ = torch.pca_lowrank(encoder_hidden_states.transpose(1,2), q=3)
        pca_encoder_out = U.transpose(1,2)
        return pca_encoder_out
    
    def classify(self, encoder_hidden_states):
        logit = self.classifier(self.dropout(encoder_hidden_states))

        return logit
    
    def adversarial_generate(self, encoder_hidden_states):
        adv_ex = self.dropout(F.gelu(self.adv_linear(encoder_hidden_states)))
        adv_ex = self.adv_linear2(self.adv_norm(adv_ex))

        return adv_ex