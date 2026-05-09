import os
import tempfile
import sentencepiece as spm
import logging
from .base import BaseTokenizer

logger = logging.getLogger(__name__)


class UnigramTokenizer(BaseTokenizer):
    def __init__(self, config):
        super().__init__(config)
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.special_tokens = [self.unk_token, self.pad_token, self.bos_token, self.eos_token]
        self.model_prefix = None
        self._temp_training_dir = None
        self.sp = None

    def train(self, file_path):
        self._temp_training_dir = tempfile.mkdtemp(prefix="unigram_train_")
        self.model_prefix = os.path.join(self._temp_training_dir, "spm_unigram")
        
        # SentencePiece trains from plain text.
        temp_txt_path = None
        if file_path.endswith('.arrow'):
            import pyarrow.ipc as ipc
            temp_txt_path = file_path + ".temp.txt"
            with ipc.open_file(file_path) as reader:
                with open(temp_txt_path, 'w', encoding='utf-8') as f:
                    for i in range(reader.num_record_batches):
                        batch = reader.get_batch(i)
                        texts = batch.column(0).to_pylist()
                        for text in texts:
                            if text:
                                lines = text.split('\n')
                                for line in lines:
                                    line = line.replace('\0', '').strip()
                                    if len(line) > 10000:
                                        for k in range(0, len(line), 10000):
                                            chunk = line[k:k+10000].strip()
                                            if chunk:
                                                f.write(chunk + '\n')
                                    elif line:
                                        f.write(line + '\n')
            file_path = temp_txt_path

        vocab_size = self.vocab_size
        try:
            num_sentences = 0
            if file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    for _ in f:
                        num_sentences += 1
            
            if num_sentences > 0:
                # Clamp vocab size on tiny corpora.
                max_vocab = max(100, num_sentences * 10) 
                if vocab_size > max_vocab:
                    import warnings
                    warnings.warn(f"Vocab size {vocab_size} potentially too large for {num_sentences} sentences. Reducing to {max_vocab}.")
                    vocab_size = max_vocab
        except Exception as e:
            logger.warning(f"Error during vocab size estimation: {e}")
            pass

        cmd = " ".join(
            [
                f"--input={file_path}",
                f"--model_prefix={self.model_prefix}",
                f"--vocab_size={vocab_size}",
                "--model_type=unigram",
                "--character_coverage=0.9998",
                "--normalization_rule_name=identity",
                "--remove_extra_whitespaces=false",
                "--byte_fallback=true",
                "--split_digits=true",
                "--split_by_unicode_script=true",
                "--split_by_number=true",
                "--split_by_whitespace=true",
                "--allow_whitespace_only_pieces=false",
                "--max_sentence_length=16384",
                "--max_sentencepiece_length=16",
                "--shuffle_input_sentence=true",
                "--train_extremely_large_corpus=true",
                f"--num_threads={os.cpu_count()}",
                "--unk_id=0",
                "--bos_id=1",
                "--eos_id=2",
                "--pad_id=3",
                f"--unk_piece={self.unk_token}",
                f"--bos_piece={self.bos_token}",
                f"--eos_piece={self.eos_token}",
                f"--pad_piece={self.pad_token}",
            ]
        )
        spm.SentencePieceTrainer.Train(cmd)

        if temp_txt_path and os.path.exists(temp_txt_path):
            os.remove(temp_txt_path)

        model_file = self.model_prefix + ".model"
        self.sp = spm.SentencePieceProcessor(model_file=model_file)
        self.vocab_size = self.sp.get_piece_size()
        self.vocab = {self.sp.id_to_piece(i): i for i in range(self.vocab_size)}
        self.id_to_token = {i: self.sp.id_to_piece(i) for i in range(self.vocab_size)}

    def encode(self, text, add_special_tokens=False):
        ids = self.sp.encode(text, out_type=int)

        if add_special_tokens:
            ids = [self.sp.bos_id()] + ids + [self.sp.eos_id()]
        return ids

    def decode(self, ids):
        return self.sp.decode(ids)

    def save(self, path):
        super().save(path)
        os.makedirs(path, exist_ok=True)
        src_model = self.model_prefix + ".model"
        dst_model = os.path.join(path, "tokenizer.model")
        if os.path.exists(src_model):
            with open(src_model, "rb") as fin, open(dst_model, "wb") as fout:
                fout.write(fin.read())
        if self._temp_training_dir and os.path.isdir(self._temp_training_dir):
            try:
                import shutil
                shutil.rmtree(self._temp_training_dir, ignore_errors=True)
            except OSError:
                pass
            self._temp_training_dir = None

    def load(self, path):
        super().load(path)
        model_file = os.path.join(path, "tokenizer.model")
        self.sp = spm.SentencePieceProcessor(model_file=model_file)
        self.vocab_size = self.sp.get_piece_size()
        self.vocab = {self.sp.id_to_piece(i): i for i in range(self.vocab_size)}
        self.id_to_token = {i: self.sp.id_to_piece(i) for i in range(self.vocab_size)}
