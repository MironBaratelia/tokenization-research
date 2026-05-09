# MIRON Library

`miron_lib` contains the final MIRON implementation used by the experiments.
The model works at word level in the CoreLM, while each word is represented as a
fixed character slot.

## Architecture

```text
word characters
  -> ReferenceCharEncoder
  -> flatten all slot states: (max_word_length + 1) * char_dim
  -> enc_proj: flattened slot -> CoreLM hidden size
  -> SmolLM2 CoreLM
  -> dec_proj: CoreLM hidden size -> flattened decoder context
  -> ReferenceCharDecoder
  -> next-word characters
```

The public implementation intentionally supports only the final architecture:

- `encoder_pooling = "flatten"`
- `decoder_conditioning = "flatten"`
- default character width: `128`
- default CoreLM hidden size: `768`
- projectors: `(max_word_length + 1) * 128 <-> 768`

For the current experiments `max_word_length = 24`, so the bridge is
`25 * 128 = 3200 -> 768` in the encoder and `768 -> 3200` in the decoder.
The positional table must cover the decoder context plus teacher-forced
characters, therefore the effective minimum is `2 * max_word_length + 1`.

## Usage

```python
from src.miron_lib import MironConfig, MironForCausalLM
from src.models.smollm2 import SmolLM2Config, SmolLM2Model

config = MironConfig(
    vocab_size=50000,
    pad_token_id=0,
    eow_token_id=1,
    bow_token_id=0,
    max_word_length=24,
    encoder_char_dim=128,
    decoder_char_dim=128,
)

lm_config = SmolLM2Config(hidden_size=config.d_model)
lm_model = SmolLM2Model(lm_config)
model = MironForCausalLM(config, lm_model=lm_model)
```

`forward(input_ids)` expects `input_ids` with shape
`(batch, words, max_word_length)` and returns `loss`, `lm_char_loss`, and
`lm_char_acc`.
