import os
import pyarrow.ipc
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from .base import BaseTokenizer


class ByteBPETokenizer(BaseTokenizer):
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = Tokenizer(models.BPE(unk_token=self.unk_token))
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = decoders.ByteLevel()
        self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    def train(self, file_path):
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )
        
        if file_path.endswith('.arrow'):
            table = pyarrow.ipc.open_file(file_path).read_all()
            
            def batch_iterator(batch_size=1000):
                for i in range(0, len(table), batch_size):
                    yield table[0][i : i + batch_size].to_pylist()
            
            self.tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
        else:
            self.tokenizer.train([file_path], trainer=trainer)
            
        self.vocab_size = self.tokenizer.get_vocab_size()
        vocab = self.tokenizer.get_vocab()
        self.vocab = {token: idx for token, idx in vocab.items()}
        self.id_to_token = {idx: token for token, idx in vocab.items()}

    def encode(self, text, add_special_tokens=False):
        encoding = self.tokenizer.encode(text)
        ids = encoding.ids
        
        if add_special_tokens:
            bos_id = self.tokenizer.token_to_id(self.bos_token)
            eos_id = self.tokenizer.token_to_id(self.eos_token)
            ids = [bos_id] + ids + [eos_id]
            
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def save(self, path):
        super().save(path)
        self.tokenizer.save(os.path.join(path, "tokenizer.json"))
        
        project_path = f"project_tokenizers/trained/{self.language}/byte_bpe"
        os.makedirs(project_path, exist_ok=True)
        self.tokenizer.save(os.path.join(project_path, "tokenizer.json"))

    def load(self, path):
        super().load(path)
        tokenizer_path = os.path.join(path, "tokenizer.json")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.vocab_size = self.tokenizer.get_vocab_size()
        
        vocab = self.tokenizer.get_vocab()
        self.vocab = {token: idx for token, idx in vocab.items()}
        self.id_to_token = {idx: token for token, idx in vocab.items()}
