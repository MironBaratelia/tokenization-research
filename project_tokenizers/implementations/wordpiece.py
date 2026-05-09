import os
import pyarrow.ipc
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
from .base import BaseTokenizer


class WordPieceTokenizer(BaseTokenizer):
    def __init__(self, config):
        super().__init__(config)
        self.continuing_subword_prefix = "##"
        self.tokenizer = Tokenizer(models.WordPiece(unk_token=self.unk_token))
        self.tokenizer.normalizer = normalizers.NFC()
        self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern="\n", behavior="isolated"),
            pre_tokenizers.Metaspace(replacement=" ") 
        ])

    def _get_alphabet(self):
        metaspace_char = " " 
        newline_char = "\n"
        
        if self.language == "ru":
            ru_letters = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
        else:
            ru_letters = ""
        
        en_letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        digits = "0123456789"
        symbols = " -.,:;!?\"'()[]{}%#@*/\\+=<>|_"
        
        return list(ru_letters + en_letters + digits + symbols + metaspace_char + newline_char)

    def train(self, file_path):
        texts = []

        if os.path.exists(file_path) and file_path.endswith(".arrow"):
            table = pyarrow.ipc.open_file(file_path).read_all()
            texts = table.column(0).to_pylist()
        
        test_file = os.path.join(os.path.dirname(file_path), "..", "..", "..", "data", "base", self.language, "tokenizer_check.txt")
        if os.path.exists(test_file):
            with open(test_file, 'r', encoding='utf-8') as f:
                t_content = f.read()
                texts.append(t_content)
        
        if not texts:
            texts = ["Пример 123.", "Тест\nПеренос."]

        trainer = trainers.WordPieceTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            continuing_subword_prefix=self.continuing_subword_prefix,
            initial_alphabet=self._get_alphabet(), 
            min_frequency=self.min_frequency,
            show_progress=True,
        )

        self.tokenizer.train_from_iterator(texts, trainer=trainer)
        
        self.vocab_size = self.tokenizer.get_vocab_size()
        vocab = self.tokenizer.get_vocab()
        self.vocab = {token: idx for token, idx in vocab.items()}
        self.id_to_token = {idx: token for token, idx in vocab.items()}

    def encode(self, text, add_special_tokens=False):
        enc = self.tokenizer.encode(text)
        ids = enc.ids
        if add_special_tokens:
            bos_id = self.tokenizer.token_to_id(self.bos_token)
            eos_id = self.tokenizer.token_to_id(self.eos_token)
            ids = [bos_id] + ids + [eos_id]
        return ids

    def decode(self, ids):
        tokens = []
        for id in ids:
            token = self.tokenizer.id_to_token(id)
            tokens.append(token)
        
        return self._manual_decode(tokens)
    
    def _manual_decode(self, tokens):
        decoded = []
        
        for i, token in enumerate(tokens):
            if token in [self.unk_token, self.pad_token, self.bos_token, self.eos_token]:
                continue
                
            if token.startswith("##"):
                decoded.append(token[2:])
                continue

            if token == " \n" or token == "\n":
                decoded.append("\n")
                
            elif token.startswith(" "):
                cleaned_token = token.replace(" ", " ")
                
                is_prev_newline = (i > 0 and (tokens[i-1] == " \n" or tokens[i-1] == "\n"))
                is_first_token = (i == 0)
                
                if is_prev_newline or is_first_token:
                    decoded.append(cleaned_token.lstrip())
                else:
                    decoded.append(cleaned_token)
            
            else:
                decoded.append(token)
                
        return "".join(decoded)

    def save(self, path):
        super().save(path)
        self.tokenizer.save(os.path.join(path, "tokenizer.json"))
        
        project_path = f"project_tokenizers/trained/{self.language}/wordpiece"
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
        
        self.tokenizer.normalizer = normalizers.NFC()
        self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern="\n", behavior="isolated"),
            pre_tokenizers.Metaspace(replacement=" ") 
        ])
