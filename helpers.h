#include <iostream>
#include <vector>

#include <fstream>
#include <stdexcept>
#include <string>

#include <cmath>
#include <limits>


struct BlockWeights {
    std::vector<float> ln1_gamma, ln1_beta;
    std::vector<float> qkv_weight, qkv_bias;
    std::vector<float> attn_out_weight, attn_out_bias;

    std::vector<float> ln2_gamma, ln2_beta;
    std::vector<float> fc_weight, fc_bias;
    std::vector<float> mlp_out_weight, mlp_out_bias;
};

inline void gelu_inplace(std::vector<float>& values) {
    const float coefficient = 0.7978845608f; // sqrt(2 / pi)

    for (float& value : values) {
        float x = value;

        value = 0.5f * x * (
            1.0f + std::tanh(
                coefficient * (x + 0.044715f * x * x * x)
            )
        );
    }
}

inline std::vector<float> attention_values(
    const std::vector<float>& weights,
    const std::vector<float>& v,
    std::size_t sequence_length,
    std::size_t embedding_dim,
    std::size_t num_heads
) {
    const std::size_t head_dim = embedding_dim / num_heads;

    // Shape: [sequence_length, embedding_dim].
    std::vector<float> context(
        sequence_length * embedding_dim, 0.0f
    );

    for (std::size_t head = 0; head < num_heads; ++head) {
        for (std::size_t i = 0; i < sequence_length; ++i) {
            for (std::size_t d = 0; d < head_dim; ++d) {
                float sum = 0.0f;

                for (std::size_t j = 0; j <= i; ++j) {
                    const std::size_t weight_index =
                        (head * sequence_length + i)
                        * sequence_length + j;

                    const std::size_t value_index =
                        j * embedding_dim + head * head_dim + d;

                    sum += weights[weight_index] * v[value_index];
                }

                context[
                    i * embedding_dim + head * head_dim + d
                ] = sum;
            }
        }
    }

    return context;
}

inline std::vector<float> softmax_rows(
    const std::vector<float>& input,
    std::size_t rows,
    std::size_t width
) {
    std::vector<float> output(input.size());

    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t offset = row * width;

        float maximum = input[offset];
        for (std::size_t j = 1; j < width; ++j) {
            if (input[offset + j] > maximum) {
                maximum = input[offset + j];
            }
        }

        float sum = 0.0f;
        for (std::size_t j = 0; j < width; ++j) {
            output[offset + j] =
                std::exp(input[offset + j] - maximum);

            sum += output[offset + j];
        }

        for (std::size_t j = 0; j < width; ++j) {
            output[offset + j] /= sum;
        }
    }

    return output;
}

inline std::vector<float> attention_scores(
    const std::vector<float>& q,
    const std::vector<float>& k,
    std::size_t sequence_length,
    std::size_t embedding_dim,
    std::size_t num_heads
) {
    const std::size_t head_dim = embedding_dim / num_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Shape: [num_heads, sequence_length, sequence_length].
    // Future positions stay at -infinity.
    std::vector<float> scores(
        num_heads * sequence_length * sequence_length,
        -std::numeric_limits<float>::infinity()
    );

    for (std::size_t head = 0; head < num_heads; ++head) {
        for (std::size_t i = 0; i < sequence_length; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                float dot = 0.0f;

                for (std::size_t d = 0; d < head_dim; ++d) {
                    const std::size_t feature = head * head_dim + d;

                    dot += q[i * embedding_dim + feature]
                         * k[j * embedding_dim + feature];
                }

                const std::size_t index =
                    (head * sequence_length + i) * sequence_length + j;

                scores[index] = dot * scale;
            }
        }
    }

    return scores;
}

std::vector<float> linear(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    const std::vector<float>& bias,
    std::size_t rows,
    std::size_t input_width,
    std::size_t output_width
) {
    std::vector<float> output(rows * output_width);

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t out = 0; out < output_width; ++out) {
            double sum = static_cast<double>(bias[out]);

            for (std::size_t in = 0; in < input_width; ++in) {
                sum += static_cast<double>(input[row * input_width + in])
                * static_cast<double>(weight[in * output_width + out]);
            }

            output[row * output_width + out] = static_cast<float>(sum);
        }
    }

    return output;
}

std::vector<float> load_floats(
    const std::string& filename,
    std::size_t count
) {
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        throw std::runtime_error("Cannot open: " + filename);
    }

    std::vector<float> values(count);

    const auto bytes =
        static_cast<std::streamsize>(count * sizeof(float));

    if (!file.read(reinterpret_cast<char*>(values.data()), bytes)) {
        throw std::runtime_error("Incomplete file: " + filename);
    }

    return values;
}

inline BlockWeights load_block(
    const std::string& directory,
    std::size_t layer,
    std::size_t width
) {
    const std::string prefix =
        directory + "/block" + std::to_string(layer) + "_";

    const std::size_t mlp_width = 4 * width;

    BlockWeights w;

    w.ln1_gamma = load_floats(prefix + "ln1_weight.bin", width);
    w.ln1_beta  = load_floats(prefix + "ln1_bias.bin", width);

    w.qkv_weight = load_floats(
        prefix + "qkv_weight.bin", width * 3 * width
    );
    w.qkv_bias = load_floats(prefix + "qkv_bias.bin", 3 * width);

    w.attn_out_weight = load_floats(
        prefix + "attn_out_weight.bin", width * width
    );
    w.attn_out_bias = load_floats(
        prefix + "attn_out_bias.bin", width
    );

    w.ln2_gamma = load_floats(prefix + "ln2_weight.bin", width);
    w.ln2_beta  = load_floats(prefix + "ln2_bias.bin", width);

    w.fc_weight = load_floats(
        prefix + "fc_weight.bin", width * mlp_width
    );
    w.fc_bias = load_floats(prefix + "fc_bias.bin", mlp_width);

    w.mlp_out_weight = load_floats(
        prefix + "mlp_out_weight.bin", mlp_width * width
    );
    w.mlp_out_bias = load_floats(
        prefix + "mlp_out_bias.bin", width
    );

    return w;
}

std::vector<float> layer_norm(
    const std::vector<float>& x,
    const std::vector<float>& gamma,
    const std::vector<float>& beta,
    std::size_t rows,
    std::size_t width,
    float eps = 1e-5f
) {
    std::vector<float> output(x.size());

    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t offset = row * width;

        float mean = 0.0f;
        for (std::size_t d = 0; d < width; ++d) {
            mean += x[offset + d];
        }
        mean /= static_cast<float>(width);

        float variance = 0.0f;
        for (std::size_t d = 0; d < width; ++d) {
            float difference = x[offset + d] - mean;
            variance += difference * difference;
        }
        variance /= static_cast<float>(width);

        float inv_std = 1.0f / std::sqrt(variance + eps);

        for (std::size_t d = 0; d < width; ++d) {
            output[offset + d] =
                (x[offset + d] - mean) * inv_std * gamma[d]
                + beta[d];
        }
    }

    return output;
}

inline std::vector<float> transformer_block(
    const std::vector<float>& x,
    const BlockWeights& w,
    std::size_t sequence_length,
    std::size_t embedding_dim,
    std::size_t num_heads
) {
    const std::size_t qkv_width = 3 * embedding_dim;
    const std::size_t mlp_width = 4 * embedding_dim;

    // First LayerNorm and Q/K/V projection.
    auto normalized = layer_norm(
        x, w.ln1_gamma, w.ln1_beta,
        sequence_length, embedding_dim
    );

    auto qkv = linear(
        normalized, w.qkv_weight, w.qkv_bias,
        sequence_length, embedding_dim, qkv_width
    );

    // Split the packed Q/K/V rows.
    std::vector<float> q(x.size());
    std::vector<float> k(x.size());
    std::vector<float> v(x.size());

    for (std::size_t pos = 0; pos < sequence_length; ++pos) {
        for (std::size_t d = 0; d < embedding_dim; ++d) {
            const std::size_t source = pos * qkv_width;
            const std::size_t destination = pos * embedding_dim + d;

            q[destination] = qkv[source + d];
            k[destination] = qkv[source + embedding_dim + d];
            v[destination] = qkv[source + 2 * embedding_dim + d];
        }
    }

    // Causal attention.
    auto scores = attention_scores(
        q, k, sequence_length, embedding_dim, num_heads
    );

    auto weights = softmax_rows(
        scores, num_heads * sequence_length, sequence_length
    );

    auto context = attention_values(
        weights, v, sequence_length, embedding_dim, num_heads
    );

    auto attention_output = linear(
        context, w.attn_out_weight, w.attn_out_bias,
        sequence_length, embedding_dim, embedding_dim
    );

    // First residual.
    std::vector<float> after_attention(x.size());

    for (std::size_t i = 0; i < x.size(); ++i) {
        after_attention[i] = x[i] + attention_output[i];
    }

    // Second LayerNorm and MLP.
    auto mlp_input = layer_norm(
        after_attention, w.ln2_gamma, w.ln2_beta,
        sequence_length, embedding_dim
    );

    auto hidden = linear(
        mlp_input, w.fc_weight, w.fc_bias,
        sequence_length, embedding_dim, mlp_width
    );

    gelu_inplace(hidden);

    auto mlp_output = linear(
        hidden, w.mlp_out_weight, w.mlp_out_bias,
        sequence_length, mlp_width, embedding_dim
    );

    // Second residual.
    std::vector<float> output(x.size());

    for (std::size_t i = 0; i < x.size(); ++i) {
        output[i] = after_attention[i] + mlp_output[i];
    }

    return output;
}

inline std::vector<float> forward(
    const std::vector<int>& tokens,
    const std::vector<float>& token_embeddings,
    const std::vector<float>& position_embeddings,
    const std::vector<BlockWeights>& blocks,
    const std::vector<float>& final_gamma,
    const std::vector<float>& final_beta,
    std::size_t embedding_dim,
    std::size_t num_heads
) {
    const std::size_t sequence_length = tokens.size();
    const std::size_t vocab_size =
        token_embeddings.size() / embedding_dim;
    const std::size_t max_positions =
        position_embeddings.size() / embedding_dim;

    if (tokens.empty() || sequence_length > max_positions) {
        throw std::runtime_error("Invalid input sequence length");
    }

    // Token + position embeddings.
    std::vector<float> hidden(sequence_length * embedding_dim);

    for (std::size_t pos = 0; pos < sequence_length; ++pos) {
        const int id = tokens[pos];

        if (id < 0 || static_cast<std::size_t>(id) >= vocab_size) {
            throw std::runtime_error("Invalid token ID");
        }

        for (std::size_t d = 0; d < embedding_dim; ++d) {
            hidden[pos * embedding_dim + d] =
                token_embeddings[
                    static_cast<std::size_t>(id) * embedding_dim + d
                ]
                + position_embeddings[pos * embedding_dim + d];
        }
    }

    // All transformer blocks.
    for (const auto& block : blocks) {
        hidden = transformer_block(
            hidden, block, sequence_length,
            embedding_dim, num_heads
        );
    }

    // Final LayerNorm.
    hidden = layer_norm(
        hidden, final_gamma, final_beta,
        sequence_length, embedding_dim
    );

    // Vocabulary scores for the last position.
    const std::size_t last_offset =
        (sequence_length - 1) * embedding_dim;

    std::vector<float> logits(vocab_size);

    for (std::size_t token = 0; token < vocab_size; ++token) {
        double sum = 0.0;

        for (std::size_t d = 0; d < embedding_dim; ++d) {
            sum += static_cast<double>(hidden[last_offset + d])
                 * static_cast<double>(
                       token_embeddings[token * embedding_dim + d]
                   );
        }

        logits[token] = static_cast<float>(sum);
    }

    return logits;
}
