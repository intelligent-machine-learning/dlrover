import tensorflow as tf
from tensorflow.python.platform import test
from tfplus.flash_attn.python.ops.flash_attn_ops import FlashAttentionLayer
num_heads = 2
seq_len_q = 4
seq_len_v = 4
d_model = 32
dtype = tf.half

class FlashAttentionTest(test.TestCase):

  def __init__(self, method_name='FlashAttentionTest'):
    super().__init__(method_name)

  @staticmethod
  def _run(instance):
    with tf.device("/gpu:0"):
        # Create input tensors
        queries = tf.random.normal((1, seq_len_q, d_model), dtype=dtype)
        values = tf.random.normal((1, seq_len_v, d_model), dtype=dtype)
        # Create an attention_mask filled with 1 to mask out all positions
        attention_mask = tf.ones((1, num_heads, seq_len_q, seq_len_v))
        # Create MultiHeadAttention layer
        multihead_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model//num_heads)
        # Apply attention_mask to mask attention weights
        attention_output = multihead_attention(queries, values, attention_mask)
        flash_attention_layer = FlashAttentionLayer(
            seq_len_q, seq_len_v, num_heads, d_model//num_heads)
        flash_attention_output = flash_attention_layer(
            queries, values, values, attention_mask)
        instance.assertTrue(
            (tf.abs(flash_attention_output - attention_output) <= 1e-4).all())

if __name__ == '__main__':
  test.main()
