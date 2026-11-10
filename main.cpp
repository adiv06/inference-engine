#include "helpers.h"
#include <chrono>

int main() {
    // Token IDs for "The capital of France is"
    std::vector<int> tokens = {464, 3139, 286, 4881, 318};

    auto token_embeddings = load_floats(
        "exports/token_embeddings.bin",
        50257ULL * 768
    );

    auto position_embeddings = load_floats(
        "exports/position_embeddings.bin",
        1024ULL * 768
    );

    std::cout << "Loaded both embedding tables\n";

    const std::size_t embedding_dim = 768;
    const std::size_t sequence_length = tokens.size();

    // Flat storage for a [sequence_length, embedding_dim] matrix.
    std::vector<float> x(sequence_length * embedding_dim);

    for (std::size_t pos = 0; pos < sequence_length; ++pos) { //the token
        int token_id = tokens[pos];

        for (std::size_t d = 0; d < embedding_dim; ++d) { //doing a sequential computation for all 768
            x[pos * embedding_dim + d] =
                token_embeddings[token_id * embedding_dim + d]
                + position_embeddings[pos * embedding_dim + d];
        }
    }

    const std::size_t num_heads = 12;
    const std::size_t num_layers = 12;

    // Load once, so we can reuse the weights for future generation.
    std::vector<BlockWeights> blocks;
    blocks.reserve(num_layers);

    for (std::size_t layer = 0; layer < num_layers; ++layer) {
        blocks.push_back(load_block("exports", layer, embedding_dim));
    }

    auto final_gamma = load_floats(
        "exports/final_ln_weight.bin",
        embedding_dim
    );

    auto final_beta = load_floats(
        "exports/final_ln_bias.bin",
        embedding_dim
    );

    const std::size_t max_new_tokens = 20;
    const int eos_token_id = 50256;

    const std::size_t max_positions =
        position_embeddings.size() / embedding_dim;

    std::cout << "Generated token IDs:\n";

    const std::size_t initial_length = tokens.size();
    const auto start = std::chrono::steady_clock::now();

    for (std::size_t step = 0; step < max_new_tokens; ++step) {
        if (tokens.size() >= max_positions) {
            break;
        }

        auto logits = forward(
            tokens,
            token_embeddings,
            position_embeddings,
            blocks,
            final_gamma,
            final_beta,
            embedding_dim,
            num_heads
        );

        // Greedy selection.
        std::size_t next_id = 0;

        for (std::size_t id = 1; id < logits.size(); ++id) {
            if (logits[id] > logits[next_id]) {
                next_id = id;
            }
        }

        tokens.push_back(static_cast<int>(next_id));

        // std::cout << next_id << ' ' << std::flush;

        if (static_cast<int>(next_id) == eos_token_id) {
            break;
        }
    }

    const auto end = std::chrono::steady_clock::now();

    const double seconds =
        std::chrono::duration<double>(end - start).count();

    const std::size_t generated_count = tokens.size() - initial_length;

    std::cout << "\nGeneration time: " << seconds << " seconds\n";
    std::cout << "Throughput: "
            << generated_count / seconds << " tokens/sec\n";

    std::cout << "\nFull sequence:\n";

    for (int id : tokens) {
        std::cout << id << ' ';
    }

    std::cout << '\n';

}