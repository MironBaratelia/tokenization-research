import os
import json
import unicodedata
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, normalizers
from .base import BaseTokenizer

_MAX_MORPHEME_LEN = 20
_MAX_VOCAB_TOKEN_LEN = 128


_END = object()


class _RadixNode:
    __slots__ = ("edges", "word")

    def __init__(self) -> None:
        self.edges: Dict[str, Tuple[str, "_RadixNode"]] = {}
        self.word: Optional[str] = None


def _radix_insert_morph(node: _RadixNode, s: str, full: str) -> None:
    if not s:
        node.word = full
        return
    c0 = s[0]
    if c0 not in node.edges:
        leaf = _RadixNode()
        leaf.word = full
        node.edges[c0] = (s[1:], leaf)
        return
    edge_rest, ch = node.edges[c0]
    edge = c0 + edge_rest
    i = 0
    lim = min(len(s), len(edge))
    while i < lim and s[i] == edge[i]:
        i += 1
    if i == len(edge) and i == len(s):
        ch.word = full
    elif i == len(edge):
        _radix_insert_morph(ch, s[i:], full)
    elif i == len(s):
        mid = _RadixNode()
        mid.word = full
        mid.edges[edge[i]] = (edge[i + 1 :], ch)
        node.edges[c0] = (s[1:i], mid)
    else:
        mid = _RadixNode()
        mid.edges[edge[i]] = (edge[i + 1 :], ch)
        node.edges[c0] = (s[1:i], mid)
        _radix_insert_morph(mid, s[i:], full)


def _build_morpheme_radix(strings: Set[str], max_len: int = _MAX_MORPHEME_LEN) -> _RadixNode:
    root = _RadixNode()
    for s in strings:
        if not s or len(s) > max_len:
            continue
        _radix_insert_morph(root, s, s)
    return root


def _build_vocab_token_trie(tokens: Set[str], max_len: int = _MAX_VOCAB_TOKEN_LEN) -> dict:
    root: dict = {}
    for tok in tokens:
        if not tok or len(tok) > max_len:
            continue
        node = root
        for c in tok:
            node = node.setdefault(c, {})
        node[_END] = tok
    return root


def _radix_longest_morph(root: _RadixNode, text: str, start: int, max_len: int = _MAX_MORPHEME_LEN) -> Tuple[int, str]:
    best = ""
    node = root
    pos = start
    end_limit = min(len(text), start + max_len)
    while pos < end_limit:
        c = text[pos]
        if c not in node.edges:
            break
        edge_rest, ch = node.edges[c]
        edge = c + edge_rest
        el = len(edge)
        if pos + el > end_limit:
            break
        if not text.startswith(edge, pos):
            break
        pos += el
        node = ch
        if node.word is not None:
            best = node.word
    return len(best), best


def _build_morpheme_trie(morpheme_to_id: Dict[str, int]):
    trie = {}
    for m, mid in morpheme_to_id.items():
        if not m or len(m) > _MAX_VOCAB_TOKEN_LEN:
            continue
        node = trie
        for c in m:
            node = node.setdefault(c, {})
        node[_END] = (len(m), mid)
    return trie


def _longest_match_id(trie, text: str, start: int):
    node = trie
    best = (0, None)
    end = min(start + _MAX_VOCAB_TOKEN_LEN, len(text))
    for i in range(start, end):
        c = text[i]
        if c not in node:
            break
        node = node[c]
        if _END in node:
            best = node[_END]
    return best


def merge_space_after_morphemes(raw: List[str], metaspace: str = " ") -> List[str]:
    if not raw:
        return []
    out: List[str] = []
    i = 0
    n = len(raw)
    while i < n:
        t = raw[i]
        if t == metaspace and i + 1 < n:
            out.append(metaspace + raw[i + 1])
            i += 2
        elif t == metaspace:
            out.append(metaspace)
            i += 1
        else:
            out.append(t)
            i += 1
    return out


class MorphemeTokenizer(BaseTokenizer):
    def __init__(self, config):
        self.id_to_token = {}
        super().__init__(config)
        self.continuing_subword_prefix = "##"
        self.word_to_morphemes = {}
        self.morpheme_to_id = {}
        self.id_to_morpheme = {}
        self.word_to_ids = {}
        self.tokenizer = None
        self.metaspace_char = " "
        self.newline_char = "\n"
        self.ru_letters = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
        self.en_letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.digits = "0123456789"
        self.symbols = " -.,:;!?\"'()[]{}%#@*/\\+=<>|_"
        self._symbols_set = frozenset(self.symbols)
        self._digits_set = frozenset(self.digits)
        self._raw_morpheme_strings: Set[str] = set()
        self._oov_morph_radix: _RadixNode = _RadixNode()
        self._vocab_token_trie: dict = {}
        self._corpus_vocab_trained = False

    @property
    def id_to_token(self):
        return self.id_to_morpheme

    @id_to_token.setter
    def id_to_token(self, value):
        self.id_to_morpheme = value

    def _get_alphabet(self):
        if self.language == "ru":
            ru_letters = self.ru_letters
        else:
            ru_letters = ""
        return list(ru_letters + self.en_letters + self.digits + self.symbols + self.metaspace_char + self.newline_char)

    def _nfc(self, text: str) -> str:
        return unicodedata.normalize("NFC", text)

    def _morph_match_key(self, text: str) -> str:
        return self._nfc(text).lower()

    def _split_to_words(self, text: str) -> List[str]:
        words = []
        current_word = ""
        for char in text:
            if char == self.newline_char:
                if current_word:
                    words.append(current_word)
                    current_word = ""
                words.append(self.newline_char)
            elif char == self.metaspace_char:
                if current_word:
                    words.append(current_word)
                    current_word = ""
                words.append(self.metaspace_char)
            elif char in self._symbols_set:
                if current_word:
                    words.append(current_word)
                    current_word = ""
                words.append(char)
            elif char in self._digits_set:
                if current_word:
                    words.append(current_word)
                    current_word = ""
                words.append(char)
            else:
                current_word += char
        if current_word:
            words.append(current_word)
        return words

    def _segment_word_greedy(self, word: str) -> List[str]:
        out = []
        pos = 0
        n = len(word)
        unk_char_fallback = word.__getitem__
        while pos < n:
            ln, s = _radix_longest_morph(self._oov_morph_radix, word, pos, _MAX_MORPHEME_LEN)
            if ln > 0:
                out.append(s)
                pos += ln
            else:
                out.append(unk_char_fallback(pos))
                pos += 1
        return out

    def _word_to_raw_morpheme_strings(self, word: str, *, use_lexicon: bool = True) -> List[str]:
        if use_lexicon and word in self.word_to_morphemes:
            return list(self.word_to_morphemes[word])
        return self._segment_word_greedy(word)

    def _line_raw_from_match_key(self, match_key: str, *, use_lexicon: bool = True) -> List[str]:
        acc: List[str] = []
        for w in self._split_to_words(match_key):
            if w in (self.metaspace_char, self.newline_char):
                acc.append(w)
            else:
                acc.extend(self._word_to_raw_morpheme_strings(w, use_lexicon=use_lexicon))
        return acc

    def _surface_tokens_from_match_pieces(self, orig: str, key: str, key_pieces: List[str]) -> List[str]:
        if len(orig) != len(key):
            return list(key_pieces)
        i = 0
        out: List[str] = []
        for piece in key_pieces:
            L = len(piece)
            if L == 0 or i + L > len(key) or key[i : i + L] != piece:
                return list(key_pieces)
            out.append(orig[i : i + L])
            i += L
        if i != len(key):
            return list(key_pieces)
        return out

    def _ids_for_merged_tokens(self, tokens: List[str]) -> List[int]:
        unk = self.morpheme_to_id[self.unk_token]
        return [self.morpheme_to_id.get(t, unk) for t in tokens]

    def _rebuild_encode_caches(self):
        self.word_to_ids = {}
        for w, morphs in self.word_to_morphemes.items():
            morphs = list(morphs)
            merged_bos = merge_space_after_morphemes(morphs)
            self.word_to_ids[w] = self._ids_for_merged_tokens(merged_bos)

    def _load_word_to_morphemes_dataset(self) -> None:
        dataset_path = f"project_tokenizers/trained/{self.language}/morpheme/dataset.json"
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        with open(dataset_path, "r", encoding="utf-8") as f:
            self.word_to_morphemes = json.load(f)

    def _rebuild_tries_after_vocab(self) -> None:
        alphabet = self._get_alphabet()
        self._raw_morpheme_strings = set(alphabet)
        for _w, morphs in self.word_to_morphemes.items():
            self._raw_morpheme_strings.update(morphs)
        self._oov_morph_radix = _build_morpheme_radix(self._raw_morpheme_strings, _MAX_MORPHEME_LEN)
        self._vocab_token_trie = _build_vocab_token_trie(set(self.morpheme_to_id.keys()))
        self._morpheme_trie = _build_morpheme_trie(self.morpheme_to_id)

    def _attach_hf_wordpiece(self) -> None:
        self.tokenizer = Tokenizer(models.WordPiece(vocab=self.vocab, unk_token=self.unk_token))
        self.tokenizer.normalizer = normalizers.NFC()
        self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(pattern="\n", behavior="isolated"),
                pre_tokenizers.Metaspace(replacement=" "),
            ]
        )

    def train(self, file_path: Optional[str] = None):
        self._load_word_to_morphemes_dataset()

        alphabet = self._get_alphabet()

        target_size = self.vocab_size or 32768
        use_corpus = file_path and os.path.isfile(file_path)

        if use_corpus:
            counter = self._count_merged_tokens_corpus(file_path)
            vocab = self._build_vocab_from_counter(counter, alphabet, target_size)
            self._corpus_vocab_trained = True
        else:
            vocab = self._build_vocab_legacy(alphabet)
            self._corpus_vocab_trained = False

        self.vocab = vocab
        self.id_to_morpheme = {idx: token for token, idx in vocab.items()}
        self.morpheme_to_id = {morpheme: idx for morpheme, idx in vocab.items()}
        self.vocab_size = len(vocab)

        self._rebuild_tries_after_vocab()
        self._rebuild_encode_caches()
        self._attach_hf_wordpiece()

    def _build_vocab_legacy(self, alphabet: List[str]) -> Dict[str, int]:
        all_morphemes = set(self._raw_morpheme_strings)
        vocab = {}
        idx = 0
        for token in self.special_tokens:
            vocab[token] = idx
            idx += 1
        for morpheme in sorted(all_morphemes):
            if morpheme not in vocab:
                vocab[morpheme] = idx
                idx += 1
        return vocab

    def _build_vocab_from_counter(self, counter: Counter, alphabet: List[str], target_size: int) -> Dict[str, int]:
        vocab: Dict[str, int] = {}
        idx = 0
        for token in self.special_tokens:
            vocab[token] = idx
            idx += 1
        for ch in sorted(alphabet):
            if ch not in vocab:
                vocab[ch] = idx
                idx += 1
        if idx > target_size:
            raise ValueError(
                f"vocab_size={target_size} is smaller than special+alphabet={idx}; increase vocab_size in config."
            )
        for tok, _cnt in counter.most_common():
            if tok in vocab:
                continue
            if idx >= target_size:
                break
            vocab[tok] = idx
            idx += 1
        return vocab

    def _count_merged_line(self, line: str, counter: Counter) -> None:
        orig = self._nfc(line)
        key = orig.lower()
        raw_key = self._line_raw_from_match_key(key, use_lexicon=True)
        surface_raw = self._surface_tokens_from_match_pieces(orig, key, raw_key)
        merged = merge_space_after_morphemes(surface_raw, self.metaspace_char)
        counter.update(merged)

    def _count_merged_tokens_corpus(self, file_path: str) -> Counter:
        counter: Counter = Counter()
        if file_path.endswith(".arrow"):
            import pyarrow.ipc as ipc

            reader = ipc.open_file(file_path)
            total_rows = sum(reader.get_batch(bi).num_rows for bi in range(reader.num_record_batches))
            pbar = tqdm(
                total=total_rows,
                desc="Morpheme vocab: corpus",
                unit="row",
            )
            try:
                for bi in range(reader.num_record_batches):
                    batch = reader.get_batch(bi)
                    col = batch.column(0)
                    for j in range(len(col)):
                        line = col[j].as_py()
                        pbar.update(1)
                        if not line or not isinstance(line, str):
                            continue
                        line = line.strip()
                        if not line:
                            continue
                        self._count_merged_line(line, counter)
            finally:
                pbar.close()
        else:
            with open(file_path, "r", encoding="utf-8", errors="replace", buffering=1024 * 1024) as f:
                for line in tqdm(f, desc="Morpheme vocab: corpus", unit="line"):
                    line = line.strip()
                    if not line:
                        continue
                    self._count_merged_line(line, counter)
        return counter

    def _greedy_tokenize_continuous(self, text: str) -> List[int]:
        trie = self._vocab_token_trie
        m2i = self.morpheme_to_id
        unk_id = m2i[self.unk_token]
        get = m2i.get
        end_sym = _END
        ml = _MAX_VOCAB_TOKEN_LEN
        n = len(text)
        ids: List[int] = []
        append = ids.append
        pos = 0
        while pos < n:
            node = trie
            best = ""
            i = pos
            end_limit = pos + ml
            if end_limit > n:
                end_limit = n
            while i < end_limit:
                nxt = node.get(text[i])
                if nxt is None:
                    break
                node = nxt
                w = node.get(end_sym)
                if w is not None:
                    best = w
                i += 1
            if best:
                append(get(best, unk_id))
                pos += len(best)
            else:
                append(get(text[pos], unk_id))
                pos += 1
        return ids

    def encode(self, text, add_special_tokens=False):
        if not self.word_to_morphemes:
            raise ValueError("Tokenizer not trained. Call train() first.")

        text = self._nfc(text)
        ids = self._greedy_tokenize_continuous(text)

        if add_special_tokens:
            ids = [self.morpheme_to_id[self.bos_token]] + ids + [self.morpheme_to_id[self.eos_token]]

        return ids

    def decode(self, ids: List[int]) -> str:
        if not self.id_to_morpheme:
            raise ValueError("Tokenizer not trained or not loaded.")
        tokens = []
        for i in ids:
            tokens.append(self.id_to_morpheme.get(i, self.unk_token))
        return self._manual_decode(tokens)

    def _manual_decode(self, tokens: List[str]) -> str:
        compact: List[str] = []
        i = 0
        n = len(tokens)
        while i < n:
            t = tokens[i]
            if t in (self.pad_token, self.bos_token, self.eos_token):
                i += 1
                continue
            if t == self.unk_token:
                while i < n and tokens[i] == self.unk_token:
                    i += 1
                compact.append("<unk>")
                continue
            compact.append(t)
            i += 1

        decoded: List[str] = []
        for i, token in enumerate(compact):
            if token == "<unk>":
                decoded.append(" ")
                continue
            if token.startswith(self.continuing_subword_prefix):
                decoded.append(token[len(self.continuing_subword_prefix) :])
                continue
            if token in (" \n", "\n"):
                decoded.append("\n")
                continue
            if token.startswith(self.metaspace_char):
                is_prev_newline = i > 0 and compact[i - 1] in (" \n", "\n")
                is_first = i == 0
                rest = token.lstrip(self.metaspace_char)
                if not rest:
                    decoded.append(self.metaspace_char)
                    continue
                if is_prev_newline or is_first:
                    decoded.append(rest)
                else:
                    decoded.append(token)
            else:
                decoded.append(token)
        return "".join(decoded)

    def save(self, path: str) -> None:
        super().save(path)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        meta = {
            "corpus_vocab_trained": self._corpus_vocab_trained,
            "metaspace_char": self.metaspace_char,
            "newline_char": self.newline_char,
        }
        with open(os.path.join(path, "morpheme_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        if self.tokenizer is not None:
            self.tokenizer.save(os.path.join(path, "tokenizer.json"))
        trained_dir = f"project_tokenizers/trained/{self.language}/morpheme"
        os.makedirs(trained_dir, exist_ok=True)
        with open(os.path.join(trained_dir, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        with open(os.path.join(trained_dir, "morpheme_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        super().load(path)
        vocab_path = os.path.join(path, "vocab.json")
        tokenizer_path = os.path.join(path, "tokenizer.json")
        meta_path = os.path.join(path, "morpheme_meta.json")

        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self._corpus_vocab_trained = bool(meta.get("corpus_vocab_trained", False))
            self.metaspace_char = meta.get("metaspace_char", " ")
            self.newline_char = meta.get("newline_char", "\n")

        if os.path.isfile(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.vocab = {str(k): int(v) for k, v in raw.items()}
        elif os.path.isfile(tokenizer_path):
            tmp = Tokenizer.from_file(tokenizer_path)
            self.vocab = dict(tmp.get_vocab())
        else:
            raise FileNotFoundError(f"Need vocab.json or tokenizer.json in {path}")

        self.morpheme_to_id = dict(self.vocab)
        self.id_to_morpheme = {idx: tok for tok, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        self._load_word_to_morphemes_dataset()
        self._rebuild_tries_after_vocab()
        self._rebuild_encode_caches()

        if os.path.isfile(tokenizer_path):
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            self.tokenizer.normalizer = normalizers.NFC()
            self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
                [
                    pre_tokenizers.Split(pattern="\n", behavior="isolated"),
                    pre_tokenizers.Metaspace(replacement=" "),
                ]
            )
        else:
            self._attach_hf_wordpiece()
