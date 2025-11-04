MODEL_ARGS=(
  --swiglu
  --num-layers 36
  --hidden-size 4096
  --ffn-hidden-size 12288
  --num-attention-heads 32
  --group-query-attention
  --num-query-groups 8
  --use-rotary-position-embeddings
  --rotary-base 5000000
  --disable-bias-linear
  --normalization "RMSNorm"
  --norm-epsilon 1e-6
  --vocab-size 151936
  --kv-channels 128
  --qk-layernorm
  --untie-embeddings-and-output-weights
)


