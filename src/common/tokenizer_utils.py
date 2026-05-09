import torch
from typing import Any, List, Optional, Union, Dict

def get_pad_id(tokenizer: Any, default: int = 0) -> int:
    """Reliably extracts PAD token ID from various tokenizer implementations."""
    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    
    # Try MIRON-style char_to_id
    char_to_id = getattr(tokenizer, "char_to_id", None)
    if char_to_id:
        for tag in ["<pad>", "[PAD]", "pad", "PAD"]:
            if tag in char_to_id:
                return char_to_id[tag]
                
    # Try standard vocab
    vocab = getattr(tokenizer, "vocab", None)
    pad_token = getattr(tokenizer, "pad_token", None)
    if vocab and pad_token and pad_token in vocab:
        return vocab[pad_token]
        
    return default

def get_eos_id(tokenizer: Any, default: Optional[int] = None) -> Optional[int]:
    """Reliably extracts EOS token ID from various tokenizer implementations."""
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
        
    char_to_id = getattr(tokenizer, "char_to_id", None)
    if char_to_id:
        for tag in ["<eos>", "[EOS]", "eos", "EOS", "<|endoftext|>", "</s>"]:
            if tag in char_to_id:
                return char_to_id[tag]
                
    vocab = getattr(tokenizer, "vocab", None)
    eos_token = getattr(tokenizer, "eos_token", None)
    if vocab and eos_token and eos_token in vocab:
        return vocab[eos_token]
        
    return default

def get_prefix_lengths_from_offsets(
    tokenizer: Any,
    combined_texts: List[str],
    prefix_char_ends: List[int],
    add_bos: bool = True
) -> Optional[List[int]]:
    """
    Get prefix token lengths from full encodings using character offsets.
    Fixes BPE boundary mismatch: "prompt " and "prompt target" can tokenize differently
    (e.g. " foo" vs " foot" merged as one token). Returns None if tokenizer lacks offset support.
    """
    raw = getattr(tokenizer, "tokenizer", tokenizer)  # UniversalTokenizer.tokenizer = raw
    inner = getattr(raw, "tokenizer", raw)  # ByteBPE/BPE have .tokenizer = HF Tokenizer
    if not hasattr(inner, "encode"):
        return None
    has_bos = add_bos and (getattr(tokenizer, "bos_token_id", None) is not None or getattr(raw, "bos_token_id", None) is not None)
    result = []
    for combined, end in zip(combined_texts, prefix_char_ends):
        enc = inner.encode(combined)
        if not hasattr(enc, "char_to_token"):
            return None
        idx = enc.char_to_token(end) if end < len(combined) else len(enc.ids)
        if idx is None:
            idx = len(enc.ids)
        plen = (1 if has_bos else 0) + idx
        result.append(plen)
    return result


def get_bos_id(tokenizer: Any, default: Optional[int] = None) -> Optional[int]:
    """Reliably extracts BOS token ID from various tokenizer implementations (HF, BPE, ByteBPE, MIRON)."""
    if hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None:
        return tokenizer.bos_token_id
    bos_token = getattr(tokenizer, "bos_token", None)
    # MIRON-style: char_to_id mapping
    char_to_id = getattr(tokenizer, "char_to_id", None)
    if char_to_id and bos_token and bos_token in char_to_id:
        return char_to_id[bos_token]
    # Standard vocab (BPE, etc.)
    vocab = getattr(tokenizer, "vocab", None)
    if vocab and bos_token and bos_token in vocab:
        return vocab[bos_token]
    if vocab:
        for tag in ["<bos>", "[BOS]", "<s>", "<|endoftext|>"]:
            if tag in vocab:
                return vocab[tag]
    # HuggingFace tokenizers library: token_to_id(bos_token)
    token_to_id = getattr(tokenizer, "token_to_id", None)
    if callable(token_to_id) and bos_token:
        try:
            return token_to_id(bos_token)
        except (KeyError, TypeError):
            pass
    return default

class UniversalTokenizer:
    """
    Standardizes interface for all supported tokenizers (MIRON, BPE, Byte).
    Handles conversion between text and tensors (2D or 3D).
    """
    def __init__(self, tokenizer: Any):
        if isinstance(tokenizer, UniversalTokenizer):
            tokenizer = tokenizer.tokenizer
        self.tokenizer = tokenizer
        self.type = getattr(tokenizer, 'tokenizer_type', 'standard')
        self.is_miron = (self.type == 'miron')
        self.is_byte = 'byte' in self.type.lower()
        
        self.vocab_size = getattr(tokenizer, 'vocab_size', 0)
        self.pad_token_id = get_pad_id(tokenizer)
        self.eos_token_id = get_eos_id(tokenizer)
        self.unk_token_id = getattr(tokenizer, 'unk_token_id', None)
        self.bos_token_id = get_bos_id(tokenizer)

        # For MIRON, we often need max word length
        self.max_word_length = getattr(tokenizer, 'max_word_length', 32)

    def __call__(self, text: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        """HF-compatible call: forwards to encode with kwargs (return_tensors, padding, truncation, max_length)."""
        return self.encode(text, **kwargs)

    def get_token_ids(self, text: str) -> List[int]:
        """Return token IDs for a string (for bias/penalties)."""
        enc = self.tokenizer.encode(text, add_special_tokens=False)
        if not enc:
            return []
        if isinstance(enc, dict):
            enc = enc.get("input_ids", enc)
            if isinstance(enc, list) and enc and isinstance(enc[0], list):
                enc = enc[0]
        if hasattr(enc, "ids"):
            enc = enc.ids
        enc = list(enc) if isinstance(enc, (list, tuple)) else []
        if not enc:
            return []
        if isinstance(enc[0], int):
            return enc
        return [x for w in enc for x in (w if isinstance(w, list) else [w])]

    def encode(
        self,
        text: Union[str, List[str]],
        return_tensors: Optional[str] = None,
        padding: bool = True,
        add_special_tokens: bool = True,
        add_bos_only: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        padding_side: str = "right",
    ) -> Dict[str, torch.Tensor]:
        if isinstance(text, str):
            text = [text]

        add_spec = add_special_tokens and not add_bos_only

        def _to_ids(enc):
            if isinstance(enc, dict):
                enc = enc.get("input_ids", enc)
                if isinstance(enc, list) and enc and isinstance(enc[0], list):
                    enc = enc[0]
            if hasattr(enc, "ids"):
                enc = enc.ids
            return list(enc) if isinstance(enc, (list, tuple)) else enc

        encoded_list = [_to_ids(self.tokenizer.encode(t, add_special_tokens=add_spec)) for t in text]

        if add_bos_only and self.bos_token_id is not None:
            if self.is_miron:
                encoded_list = [[[self.bos_token_id]] + ids for ids in encoded_list]
            else:
                encoded_list = [[self.bos_token_id] + ids for ids in encoded_list]
        if truncation and max_length is not None:
            if self.is_miron:
                encoded_list = [seq[-max_length:] if len(seq) > max_length else seq for seq in encoded_list]
            else:
                encoded_list = [ids[-max_length:] if len(ids) > max_length else ids for ids in encoded_list]

        if return_tensors != "pt":
            return {"input_ids": encoded_list}

        if padding_side not in ("left", "right") and self.is_miron:
            raise ValueError("MIRON batch encoding supports 'left' and 'right' padding.")

        if self.is_miron:
            max_seq_len = max(len(seq) for seq in encoded_list)
            max_word_len = self.max_word_length
            
            batch_size = len(encoded_list)
            input_ids = torch.full((batch_size, max_seq_len, max_word_len), self.pad_token_id, dtype=torch.long)
            
            for i, seq in enumerate(encoded_list):
                curr_len = len(seq)
                start_idx = max_seq_len - curr_len if padding_side == "left" else 0
                for j, word in enumerate(seq):
                    if j < max_seq_len:
                        w_len = min(len(word), max_word_len)
                        input_ids[i, start_idx + j, :w_len] = torch.tensor(word[:w_len], dtype=torch.long)
            
            attention_mask = (input_ids != self.pad_token_id).any(dim=-1).long()
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        else:
            max_seq_len = max(len(ids) for ids in encoded_list)
            batch_size = len(encoded_list)
            input_ids = torch.full((batch_size, max_seq_len), self.pad_token_id, dtype=torch.long)
            if padding_side not in ("left", "right"):
                raise ValueError("padding_side must be 'left' or 'right'")

            for i, ids in enumerate(encoded_list):
                row = torch.tensor(ids, dtype=torch.long)
                curr_len = len(ids)
                if padding_side == "right":
                    input_ids[i, :curr_len] = row
                else:
                    input_ids[i, -curr_len:] = row

            attention_mask = (input_ids != self.pad_token_id).long()
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, ids: Union[List, torch.Tensor], skip_special_tokens: bool = True) -> Union[str, List[str]]:
        """Decodes IDs back to text, handling 2D and 3D tensors."""
        if isinstance(ids, torch.Tensor):
            if ids.dim() == 3:
                return self.batch_decode(ids, skip_special_tokens=skip_special_tokens)
            elif ids.dim() == 2:
                return self.batch_decode(ids, skip_special_tokens=skip_special_tokens)
            else:
                ids = ids.cpu().tolist()

        if self.is_miron and isinstance(ids, list) and ids and isinstance(ids[0], list):
            return self._decode_miron_word_slots(ids)

        return self.tokenizer.decode(ids)

    def _decode_miron_word_slots(self, words: List[List[int]]) -> str:
        parts: List[str] = []
        for word in words:
            filtered = [tid for tid in word if tid != self.pad_token_id]
            if filtered:
                parts.append(self.tokenizer.decode(filtered))
        return "".join(parts)

    def batch_decode(self, sequences: Union[List, torch.Tensor], skip_special_tokens: bool = True) -> List[str]:
        """Decodes a batch of sequences."""
        if isinstance(sequences, torch.Tensor):
            if sequences.dim() == 3:
                results = []
                for i in range(sequences.size(0)):
                    word_ids = []
                    for j in range(sequences.size(1)):
                        word = sequences[i, j].cpu().tolist()
                        filtered = [tid for tid in word if tid != self.pad_token_id]
                        if filtered:
                            decoded_word = self.tokenizer.decode(filtered)
                            word_ids.append(decoded_word)
                    results.append("".join(word_ids))
                return results
            else:
                sequences = sequences.cpu().tolist()
                
        return [self.tokenizer.decode(seq) for seq in sequences]
