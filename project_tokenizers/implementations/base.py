import os
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import torch


class BaseTokenizer(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.language = config.get("language") or "ru"
        self.vocab_size = config.get("vocab_size")
        # Tokenizer trainers expect an int here.
        _mf = config.get("min_frequency")
        self.min_frequency = 2 if _mf is None else _mf
        
        self.unk_token = config.get("unk_token", "[UNK]")
        self.pad_token = config.get("pad_token", "[PAD]")
        self.bos_token = config.get("bos_token", "[CLS]")
        self.eos_token = config.get("eos_token", "[SEP]")
        
        self.special_tokens = [self.unk_token, self.pad_token, self.bos_token, self.eos_token]
        
        self.vocab = {}
        self.id_to_token = {}
        
        self.tokenizer_type = self.__class__.__name__.replace("Tokenizer", "").lower()
        self.padding_side = "right"

    @property
    def pad_token_id(self) -> Optional[int]:
        return self.vocab.get(self.pad_token)

    @property
    def eos_token_id(self) -> Optional[int]:
        return self.vocab.get(self.eos_token)

    @property
    def bos_token_id(self) -> Optional[int]:
        return self.vocab.get(self.bos_token)

    @property
    def unk_token_id(self) -> Optional[int]:
        return self.vocab.get(self.unk_token)

    def __call__(self, text: Union[str, List[str]], return_tensors: Optional[str] = None, padding: bool = False, truncation: bool = False, max_length: Optional[int] = None, add_special_tokens: bool = True) -> Dict[str, Any]:
        if isinstance(text, str):
            text = [text]
        
        batch_ids = [self.encode(t, add_special_tokens=add_special_tokens) for t in text]
        
        if truncation and max_length is not None:
            batch_ids = [ids[:max_length] for ids in batch_ids]
            
        if padding:
            if max_length is None:
                max_length = max(len(ids) for ids in batch_ids)
            
            padded_ids = []
            attention_mask = []
            for ids in batch_ids:
                pad_len = max_length - len(ids)
                if self.padding_side == "right":
                    padded_ids.append(ids + [self.pad_token_id or 0] * pad_len)
                    attention_mask.append([1] * len(ids) + [0] * pad_len)
                else:
                    padded_ids.append([self.pad_token_id or 0] * pad_len + ids)
                    attention_mask.append([0] * pad_len + [1] * len(ids))
            batch_ids = padded_ids
        else:
            attention_mask = [[1] * len(ids) for ids in batch_ids]
            
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(batch_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
            }
        
        return {
            "input_ids": batch_ids,
            "attention_mask": attention_mask
        }

    def batch_decode(self, sequences: Union[List[List[int]], torch.Tensor], skip_special_tokens: bool = False) -> List[str]:
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()
        
        decoded = []
        for ids in sequences:
            if skip_special_tokens:
                special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id, self.unk_token_id}
                ids = [tid for tid in ids if tid not in special_ids and tid is not None]
            decoded.append(self.decode(ids))
        return decoded

    @abstractmethod
    def train(self, file_path: str) -> None:
        pass

    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        pass

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        pass
    
    def tokenize(self, text: str) -> List[str]:
        ids = self.encode(text)
        return [self.id_to_token.get(id, self.unk_token) for id in ids]
    
    def encode_words(self, text: str) -> torch.Tensor:
        raise NotImplementedError(f"{self.__class__.__name__} does not support word vector encoding")
    
    def decode_words(self, word_vectors: torch.Tensor) -> str:
        raise NotImplementedError(f"{self.__class__.__name__} does not support word vector decoding")

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
            json.dump({
                "type": self.tokenizer_type,
                "language": self.language,
                "vocab_size": self.vocab_size,
                "min_frequency": self.min_frequency,
                "special_tokens": self.special_tokens
            }, f, indent=2, ensure_ascii=False)

    def load(self, path: str) -> None:
        with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)
            self.language = config.get("language") or "ru"
            self.vocab_size = config.get("vocab_size")
            _mf = config.get("min_frequency")
            self.min_frequency = 2 if _mf is None else _mf
            self.special_tokens = config.get("special_tokens", [self.unk_token, self.pad_token, self.bos_token, self.eos_token])
    
    def get_vocab_size(self) -> int:
        return self.vocab_size if self.vocab_size is not None else len(self.vocab)
    
    def __len__(self) -> int:
        return self.get_vocab_size()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.tokenizer_type}, vocab_size={self.get_vocab_size()})"
