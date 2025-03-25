import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, Dense, MultiHeadAttention

# Define a Transformer Block
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim)]
        )
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        out1 = self.norm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return self.norm2(out1 + ffn_output)

# Initialize Transformer Block
transformer_block = TransformerBlock(embed_dim=128, num_heads=4, ff_dim=512)

# Dummy Input
inputs = tf.random.uniform((32, 10, 128))  # Batch 32, Sequence 10, Embedding 128
output = transformer_block(inputs)
print(output.shape)  # Output shape
