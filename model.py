"""
Model Module for News Headline Classification
==============================================
This module implements a DistilBERT-based binary classifier for
distinguishing between Fox News and NBC News headlines.

Architecture:
- Base: DistilBERT (distilbert-base-uncased)
- Classification Head: Linear layer (768 -> 2)

Key Features:
- Fast inference using DistilBERT (40% smaller, 60% faster than BERT)
- Automatic GPU/CPU detection
- Batch processing support

Author: Chih Yu Tsai, Aditya Pratap Singh, Xinjie Hu
Course: CIS 5190 Applied Machine Learning Fall 2025
"""

import torch
import torch.nn as nn
from typing import List, Any, Iterable
from transformers import DistilBertTokenizer, DistilBertModel


class Model(nn.Module):
    """
    DistilBERT-based News Headline Classifier.
    
    This model classifies news headlines as either 'foxnews' or 'nbcnews'
    using a fine-tuned DistilBERT encoder with a linear classification head.
    
    Attributes:
        tokenizer: DistilBERT tokenizer for text encoding
        bert: DistilBERT base model for feature extraction
        classifier: Linear layer for binary classification
        device: Current device (cuda or cpu)
        label_map: Mapping from class indices to label strings
    """
    
    # Class-level constants for label mapping
    LABEL_TO_ID = {'foxnews': 0, 'nbcnews': 1}
    ID_TO_LABEL = {0: 'foxnews', 1: 'nbcnews'}
    
    def __init__(self) -> None:
        """
        Initialize the DistilBERT classifier.
        
        The model is initialized with:
        - Pre-trained DistilBERT tokenizer and encoder
        - Random classification head (to be loaded from checkpoint)
        """
        super().__init__()
        
        # Set device (GPU if available, otherwise CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer (always from pretrained)
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Initialize DistilBERT encoder
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Classification head: 768 (hidden size) -> 2 (num classes)
        self.classifier = nn.Linear(768, 2)
        
        # Dropout for regularization (used during training)
        self.dropout = nn.Dropout(0.1)
        
        # Move model to device
        self.to(self.device)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Tokenized input tensor of shape (batch_size, seq_length)
            attention_mask: Attention mask tensor of shape (batch_size, seq_length)
        
        Returns:
            torch.Tensor: Logits of shape (batch_size, 2)
        """
        # Get DistilBERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation (first token)
        # outputs.last_hidden_state shape: (batch_size, seq_length, hidden_size)
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, 768)
        
        # Apply dropout and classification head
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)  # (batch_size, 2)
        
        return logits
    
    def eval(self) -> 'Model':
        """
        Set model to evaluation mode.
        
        This disables dropout and other training-specific behaviors.
        
        Returns:
            Model: Self reference for method chaining
        """
        super().eval()
        return self
    
    def predict(self, batch: Iterable[Any]) -> List[str]:
        """
        Predict labels for a batch of text inputs.
        
        This is the main inference method called by the evaluation backend.
        
        Args:
            batch: Iterable of text strings (headlines extracted from URLs)
        
        Returns:
            List[str]: Predicted labels ('foxnews' or 'nbcnews') for each input
        
        Example:
            >>> model = Model()
            >>> model.eval()
            >>> preds = model.predict(['trump announces policy', 'biden visits europe'])
            >>> print(preds)
            ['foxnews', 'nbcnews']
        """
        # Convert batch to list if needed
        texts = list(batch)
        
        if len(texts) == 0:
            return []
        
        # Tokenize all texts in the batch
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,  # Headlines are typically short
            return_tensors='pt'
        )
        
        # Move tensors to device
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Forward pass (no gradient computation for inference)
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
        
        # Get predicted class indices
        predictions = torch.argmax(logits, dim=-1).cpu().tolist()
        
        # Convert indices to label strings
        labels = [self.ID_TO_LABEL[pred] for pred in predictions]
        
        return labels


def get_model() -> Model:
    """
    Factory function required by the evaluation backend.
    
    Returns:
        Model: An instance of the DistilBERT classifier
    
    Note:
        The evaluator will optionally load weights via load_state_dict()
        after calling this function.
    """
    return Model()


# ============================================================================
# Testing / Debug Section (not used by backend)
# ============================================================================
if __name__ == '__main__':
    print("Testing Model initialization and inference...")
    print("-" * 60)
    
    # Initialize model
    model = get_model()
    model.eval()
    
    print(f"Device: {model.device}")
    print(f"Model loaded successfully!")
    
    # Test inference with sample texts
    test_texts = [
        'trump announces new immigration policy changes',
        'biden administration unveils climate plan',
        'stock market reaches record high today',
    ]
    
    print(f"\nTest predictions:")
    predictions = model.predict(test_texts)
    for text, pred in zip(test_texts, predictions):
        print(f"  '{text[:40]}...' -> {pred}")
