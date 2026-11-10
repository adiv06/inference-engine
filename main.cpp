#include "helpers.h"
#include <chrono>
#include <algorithm>
#include <array>

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

    KVCache cache(
        blocks.size(),
        max_positions,
        embedding_dim
    );

        // Five-run benchmark: normal forward(), no KV caching.
    const std::vector<int> prompt = {464, 3139, 286, 4881, 318};

    std::array<double, 5> run_times{};
    std::array<std::size_t, 5> generated_counts{};

    std::vector<int> first_sequence;
    bool sequences_match = true;

    for (std::size_t run = 0; run < run_times.size(); ++run) {
        // Start every run with the same original prompt.
        auto run_tokens = prompt;

        const auto start = std::chrono::steady_clock::now();

        for (std::size_t step = 0; step < max_new_tokens; ++step) {
            if (run_tokens.size() >= max_positions) {
                break;
            }

            auto logits = forward(
                run_tokens,
                token_embeddings,
                position_embeddings,
                blocks,
                final_gamma,
                final_beta,
                embedding_dim,
                num_heads
            );

            // Select the highest-scoring next token.
            std::size_t next_id = 0;

            for (std::size_t id = 1; id < logits.size(); ++id) {
                if (logits[id] > logits[next_id]) {
                    next_id = id;
                }
            }

            run_tokens.push_back(static_cast<int>(next_id));

            if (static_cast<int>(next_id) == eos_token_id) {
                break;
            }
        }

        const auto end = std::chrono::steady_clock::now();

        run_times[run] =
            std::chrono::duration<double>(end - start).count();

        generated_counts[run] = run_tokens.size() - prompt.size();

        if (run == 0) {
            first_sequence = run_tokens;
        } else if (run_tokens != first_sequence) {
            sequences_match = false;
        }
    }

    // Report results outside the timed sections.
    double total_seconds = 0.0;
    std::size_t total_generated = 0;

    for (std::size_t run = 0; run < run_times.size(); ++run) {
        total_seconds += run_times[run];
        total_generated += generated_counts[run];

        std::cout << "Run " << run + 1
                << ": " << run_times[run] << " seconds"
                << ", " << generated_counts[run] << " tokens\n";
    }

    auto sorted_times = run_times;
    std::sort(sorted_times.begin(), sorted_times.end());

    const double average_seconds =
        total_seconds / static_cast<double>(run_times.size());

    const double median_seconds =
        sorted_times[sorted_times.size() / 2];

    std::cout << "\nAverage time: "
            << average_seconds << " seconds\n";

    std::cout << "Median time: "
            << median_seconds << " seconds\n";

    std::cout << "Overall throughput: "
            << static_cast<double>(total_generated) / total_seconds
            << " tokens/sec\n";

    std::cout << "All five sequences match: "
            << (sequences_match ? "yes" : "NO") << '\n';

    std::cout << "\nFull sequence:\n";

    for (int id : first_sequence) {
        std::cout << id << ' ';
    }

    std::cout << '\n';   

}