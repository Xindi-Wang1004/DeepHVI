from typing import Union, List, Optional

import torch
from transformers import AutoTokenizer, AutoModel

local_model_path = "./esm2"


class ESMSequenceTransformer:
    def __init__(self,
                 model_name: str = "facebook/esm2_t6_8M_UR50D",
                 max_length: int = 1024,
                 encoder_output_dim: int = 320,
                 decoder_input_dim: int = 256):
        """
        Initialize the sequence transformer with ESM2 model and tokenizer

        Args:
            model_name: Name or path of the ESM2 model
            max_length: Maximum sequence length for tokenization
            encoder_output_dim: Dimension of encoder output
            decoder_input_dim: Dimension of decoder input
        """
        self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        self.model = AutoModel.from_pretrained(local_model_path)
        self.max_length = max_length
        self.encoder_to_decoder_proj = torch.nn.Linear(encoder_output_dim, decoder_input_dim)

        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.encoder_to_decoder_proj.to(self.device)

    def process_single_sequence(self, sequence: str) -> torch.Tensor:
        """
        Process a single protein sequence through the ESM2 model

        Args:
            sequence: Input protein sequence string

        Returns:
            torch.Tensor: Processed sequence embeddings with shape [1, hidden_dim]
        """
        # Tokenize sequence
        tokens = self.tokenizer(
            sequence,
            return_tensors="pt",
            max_length=self.max_length,
            padding=True,
            truncation=True
        ).to(self.device)

        # Generate attention mask
        attention_mask = tokens.input_ids.ne(self.model.config.pad_token_id).int()

        # Get encoder outputs
        with torch.no_grad():
            encoder_outputs = self.model(
                input_ids=tokens.input_ids,
                attention_mask=attention_mask
            ).last_hidden_state

        # Project to decoder dimension
        projected_outputs = self.encoder_to_decoder_proj(encoder_outputs)

        # Take mean over sequence length dimension
        mean_outputs = torch.mean(projected_outputs, dim=1)  # Shape: [1, 256]

        return mean_outputs

    def transform_sequences(self,
                            sequences: Union[str, List[str]],
                            batch_size: int = 8) -> torch.Tensor:
        """
        Transform protein sequences into encoder outputs

        Args:
            sequences: Single sequence string or list of sequence strings
            batch_size: Batch size for processing multiple sequences

        Returns:
            torch.Tensor: Transformed sequence embeddings with shape [batch_size, hidden_dim]
        """
        # Handle single sequence
        if isinstance(sequences, str):
            return self.process_single_sequence(sequences)

        # Handle multiple sequences
        all_outputs = []

        # Process in batches
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]

            # Tokenize batch
            batch_tokens = self.tokenizer(
                batch_sequences,
                return_tensors="pt",
                max_length=self.max_length,
                padding=True,
                truncation=True
            ).to(self.device)

            # Generate attention mask
            attention_mask = batch_tokens.input_ids.ne(self.model.config.pad_token_id).int()

            # Get encoder outputs
            with torch.no_grad():
                encoder_outputs = self.model(
                    input_ids=batch_tokens.input_ids,
                    attention_mask=attention_mask
                ).last_hidden_state

            # Project to decoder dimension
            projected_outputs = self.encoder_to_decoder_proj(encoder_outputs)

            # Take mean over sequence length dimension
            mean_outputs = torch.mean(projected_outputs, dim=1)  # Shape: [batch_size, 256]
            all_outputs.append(mean_outputs)

        # Concatenate all batches
        return torch.cat(all_outputs, dim=0)


global_transformer = ESMSequenceTransformer()


# Usage example
def transform_sequence_by_esm2(
        sequences: Optional[Union[str, List[str]]],
        transformer: Optional[ESMSequenceTransformer] = None
) -> torch.Tensor:
    """
    Transform protein sequences to embeddings using ESM2 model

    Args:
        sequences: Input sequence(s) to transform
        transformer: Optional pre-initialized SequenceTransformer

    Returns:
        torch.Tensor: Sequence embeddings
    """
    if transformer is None:
        transformer = global_transformer

    if sequences is None:
        raise ValueError("Input sequences cannot be None")

    return transformer.transform_sequences(sequences)
