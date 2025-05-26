import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BertModel, BertTokenizer
from typing import Optional, Tuple

class CausalLVLM(nn.Module):
    """Large Vision-Language Model for Causal Narrative Generation 
    
    Args:
        vision_backbone (str): Pretrained vision encoder name (default: "google/vit-base-patch16-224")
        llm_backbone (str): Pretrained LLM name (default: "Qwen1.5 14B")
        bert_model (str): BERT model for text encoding (default: "bert-base-uncased")
        max_text_length (int): Maximum generated text length (default: 128)
        freeze_vision (bool): Freeze vision encoder weights (default: True)
        freeze_llm (bool): Freeze LLM decoder weights (default: True)
    
    Input:
        images (torch.Tensor): GAF images from PCNN output (B, C, H, W)
    
    Output:
        text_embeddings (torch.Tensor): Causal text embeddings (B, text_dim)
    """
    def __init__(self,
                 vision_backbone: str = "google/vit-base-patch16-224",
                 llm_backbone: str = "Qwen1.5 14B",
                 bert_model: str = "bert-base-uncased",
                 max_text_length: int = 128,
                 freeze_vision: bool = True,
                 freeze_llm: bool = True):
        super().__init__()
        self.max_text_length = max_text_length
        
        # Phase 1: Vision-to-Text Generation Components
        self.vision_encoder = AutoModel.from_pretrained(vision_backbone)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_backbone)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_backbone)
        
        # Project vision features to LLM space
        self.vision_proj = nn.Linear(
            self.vision_encoder.config.hidden_size,
            self.llm.config.n_embd
        )
        
        # Phase 2: Text Encoding Components
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.bert_encoder = BertModel.from_pretrained(bert_model)
        
        # Freeze pretrained weights
        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad_(False)
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad_(False)
                
        # Initialize soft prompt tokens
        self.prompt_embeddings = nn.Parameter(
            torch.randn(4, self.llm.config.n_embd)
        )

    def _generate_causal_text(self, 
                            pixel_values: torch.Tensor
                           ) -> Tuple[torch.Tensor, list]:
        """Text Generation Pipeline 
        
        Mathematical Formulation:
            T = LVLM(M; θ_{LVLM}) 
            where M is the GAF image matrix
        
        Implementation Details:
        1. Vision encoder extracts image features
        2. Project features to LLM embedding space
        3. Generate text with soft prompt tuning
        """
        # Extract vision features (B, seq_len, hid_dim)
        vision_features = self.vision_encoder(pixel_values).last_hidden_state
        
        # Pool features (B, hid_dim)
        pooled_features = vision_features.mean(dim=1)
        
        # Project to LLM space (B, llm_dim)
        projected_features = self.vision_proj(pooled_features)
        
        # Concatenate with learnable prompts (B, 4 + 1, llm_dim)
        inputs_embeds = torch.cat([
            self.prompt_embeddings.unsqueeze(0).repeat(pooled_features.size(0), 
            projected_features.unsqueeze(1)
        ], dim=1)
        
        # Generate causal text
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            max_length=self.max_text_length,
            num_beams=3,
            early_stopping=True
        )
        
        # Decode generated text
        generated_text = [
            self.llm_tokenizer.decode(g, skip_special_tokens=True) 
            for g in outputs
        ]
        return outputs, generated_text

    def _encode_text(self, 
                   generated_ids: torch.Tensor
                  ) -> torch.Tensor:
        """Text Embedding Pipeline 
        
        Mathematical Formulation:
            v_text = BERT(T; θ_BERT)
        
        Implementation Details:
        1. Use BERT's [CLS] token embeddings
        2. Apply mean pooling as fallback
        """
        # Get BERT-compatible input IDs
        encoded = self.bert_tokenizer(
            generated_ids,
            padding=True,
            return_tensors="pt",
            truncation=True
        ).to(generated_ids.device)
        
        # Extract embeddings (B, seq_len, bert_dim)
        bert_output = self.bert_encoder(**encoded).last_hidden_state
        
        # Use [CLS] token embedding (B, bert_dim)
        return bert_output[:, 0, :]

    def forward(self, 
              images: torch.Tensor
             ) -> Tuple[torch.Tensor, list]:
        """End-to-End Processing Pipeline
        
        Workflow:
        1. GAF images → Causal text generation
        2. Generated text → BERT embeddings
        """
        # Validate input dimensions (B, C, H, W)
        if images.dim() != 4:
            raise ValueError(f"Input images must be 4D tensor, got {images.shape}")
            
        # Phase 1: Generate causal text
        generated_ids, generated_text = self._generate_causal_text(images)
        
        # Phase 2: Semantic vectorization 
        text_embeddings = self._encode_text(generated_ids)
        
        return text_embeddings, generated_text

    def align_modalities(self,
                       image_embeddings: torch.Tensor,
                       text_embeddings: torch.Tensor,
                       temperature: float = 0.07
                      ) -> torch.Tensor:
        """Cross-Modal Alignment Loss 
        
        Mathematical Formulation:
            L_align = Σ ||v_image - g(v_text)||^2_2
        
        Implementation Details:
        - Uses cosine similarity contrastive loss
        - Temperature-scaled softmax
        """
        # Project text to image space
        projected_text = self.vision_proj(text_embeddings)
        
        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        projected_text = F.normalize(projected_text, dim=-1)
        
        # Compute similarity matrix
        logits = (image_embeddings @ projected_text.T) / temperature
        labels = torch.arange(image_embeddings.size(0), device=image_embeddings.device)
        
        return F.cross_entropy(logits, labels)
