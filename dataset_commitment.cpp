#include "dataset_commitment.h"
#include "Summer code/my_hhash.h"
#include "Summer code/mimc.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>

// Hash a single sample (image + label)
__hhash_digest hash_sample(const torch::Tensor& image, int64_t label) {
    // Ensure image is contiguous and on CPU
    torch::Tensor img_contig = image.contiguous().cpu();
    
    // Convert to float if needed (raw MNIST is uint8, transformed is float)
    if (img_contig.dtype() != torch::kFloat32) {
        img_contig = img_contig.to(torch::kFloat32);
    }
    
    // Flatten image to 1D
    auto img_flat = img_contig.flatten();
    size_t img_size = img_flat.size(0);
    
    // Convert image pixels to field elements (quantize and hash)
    // For efficiency, we hash chunks of the image
    // Each chunk will be 4 field elements (hashed together)
    
    std::vector<F> image_elements;
    image_elements.reserve(img_size);
    
    // Quantize image pixels to field elements
    for (int64_t i = 0; i < img_size; i++) {
        float pixel = img_flat[i].item<float>();
        F pixel_field = quantize(pixel);
        image_elements.push_back(pixel_field);
    }
    
    // Add label as a field element
    F label_field = quantize(static_cast<float>(label));
    image_elements.push_back(label_field);
    
    // Hash all elements using Merkle-Damgard construction
    // Process in chunks of 4 field elements
    __hhash_digest current_hash;
    memset(&current_hash, 0, sizeof(current_hash));
    
    size_t num_chunks = (image_elements.size() + 3) / 4;  // Round up
    for (size_t chunk = 0; chunk < num_chunks; chunk++) {
        F elements[4] = {F_ZERO, F_ZERO, F_ZERO, F_ZERO};
        size_t offset = chunk * 4;
        
        for (size_t i = 0; i < 4 && (offset + i) < image_elements.size(); i++) {
            elements[i] = image_elements[offset + i];
        }
        
        // Hash 4 field elements together
        current_hash = merkle_tree::hash_double_field_element_merkle_damgard(
            elements[0], elements[1], elements[2], elements[3], current_hash);
    }
    
    return current_hash;
}

// Hash a single sample (image + float label)
__hhash_digest hash_sample(const torch::Tensor& image, float label) {
    // Ensure image is contiguous and on CPU
    torch::Tensor img_contig = image.contiguous().cpu();
    
    // Convert to float if needed
    if (img_contig.dtype() != torch::kFloat32) {
        img_contig = img_contig.to(torch::kFloat32);
    }
    
    auto img_flat = img_contig.flatten();
    size_t img_size = img_flat.size(0);
    
    std::vector<F> image_elements;
    image_elements.reserve(img_size + 1);
    
    for (int64_t i = 0; i < img_size; i++) {
        float pixel = img_flat[i].item<float>();
        F pixel_field = quantize(pixel);
        image_elements.push_back(pixel_field);
    }
    
    F label_field = quantize(label);
    image_elements.push_back(label_field);
    
    __hhash_digest current_hash;
    memset(&current_hash, 0, sizeof(current_hash));
    
    size_t num_chunks = (image_elements.size() + 3) / 4;
    for (size_t chunk = 0; chunk < num_chunks; chunk++) {
        F elements[4] = {F_ZERO, F_ZERO, F_ZERO, F_ZERO};
        size_t offset = chunk * 4;
        
        for (size_t i = 0; i < 4 && (offset + i) < image_elements.size(); i++) {
            elements[i] = image_elements[offset + i];
        }
        
        current_hash = merkle_tree::hash_double_field_element_merkle_damgard(
            elements[0], elements[1], elements[2], elements[3], current_hash);
    }
    
    return current_hash;
}

template <typename Dataset>
static DatasetCommitment commit_dataset_generic(Dataset& dataset, int num_samples) {
    DatasetCommitment commitment;
    
    // Determine number of samples to commit
    auto dataset_size = dataset.size();
    if (!dataset_size.has_value()) {
        throw std::runtime_error("Dataset size unknown");
    }
    
    size_t total_samples = dataset_size.value();
    if (num_samples < 0 || num_samples > static_cast<int>(total_samples)) {
        num_samples = total_samples;
    }
    
    commitment.num_samples = num_samples;
    
    // Hash all samples
    std::vector<__hhash_digest> leaf_hashes;
    leaf_hashes.reserve(num_samples);
    
    std::cout << "[DatasetCommitment] Hashing " << num_samples << " samples..." << std::endl;
    
    for (size_t i = 0; i < static_cast<size_t>(num_samples); i++) {
        auto example = dataset.get(i);
        __hhash_digest leaf_hash = hash_sample(example.data, example.target.template item<int64_t>());
        leaf_hashes.push_back(leaf_hash);
        commitment.sample_hashes[i] = leaf_hash;
        
        if ((i + 1) % 1000 == 0) {
            std::cout << "[DatasetCommitment] Hashed " << (i + 1) << " samples..." << std::endl;
        }
    }
    
    // Pad to power of 2 for Merkle tree construction
    size_t tree_size = 1;
    int depth = 0;
    while (tree_size < leaf_hashes.size()) {
        tree_size *= 2;
        depth++;
    }
    
    commitment.tree_depth = depth;
    
    // Pad with zero hashes
    __hhash_digest zero_hash;
    memset(&zero_hash, 0, sizeof(zero_hash));
    while (leaf_hashes.size() < tree_size) {
        leaf_hashes.push_back(zero_hash);
    }
    
    // Build Merkle tree
    commitment.tree.resize(depth + 1);
    commitment.tree[0] = leaf_hashes;  // Level 0: leaves
    
    std::cout << "[DatasetCommitment] Building Merkle tree (depth=" << depth << ")..." << std::endl;
    
    for (int level = 1; level <= depth; level++) {
        size_t prev_level_size = commitment.tree[level - 1].size();
        size_t curr_level_size = prev_level_size / 2;
        commitment.tree[level].resize(curr_level_size);
        
        for (size_t i = 0; i < curr_level_size; i++) {
            __hhash_digest data[2];
            data[0] = commitment.tree[level - 1][i * 2];
            data[1] = commitment.tree[level - 1][i * 2 + 1];
            my_hhash(data, &commitment.tree[level][i]);
        }
    }
    
    // Root hash is at tree[depth][0]
    commitment.root_hash = commitment.tree[depth][0];
    
    // Print root hash
    std::cout << "[DatasetCommitment] Dataset committed. Root hash: 0x";
    uint64_t* root_words = reinterpret_cast<uint64_t*>(&commitment.root_hash);
    for (int i = 0; i < 4; i++) {
        std::cout << std::hex << std::setfill('0') << std::setw(16) << root_words[i];
    }
    std::cout << std::dec << std::endl;
    
    return commitment;
}

// Commit dataset to Merkle tree (MNIST)
DatasetCommitment commit_dataset(torch::data::datasets::MNIST& dataset, int num_samples) {
    return commit_dataset_generic(dataset, num_samples);
}

// Commit dataset to Merkle tree (CIFAR-10)
DatasetCommitment commit_dataset(cifar10_dataset::CIFAR10& dataset, int num_samples) {
    return commit_dataset_generic(dataset, num_samples);
}

// Commit dataset to Merkle tree (Synthetic linear)
DatasetCommitment commit_dataset(SyntheticLinearDataset& dataset, int num_samples) {
    DatasetCommitment commitment;
    
    auto dataset_size = dataset.size();
    if (!dataset_size.has_value()) {
        throw std::runtime_error("Dataset size unknown");
    }
    
    size_t total_samples = dataset_size.value();
    if (num_samples < 0 || num_samples > static_cast<int>(total_samples)) {
        num_samples = total_samples;
    }
    
    commitment.num_samples = num_samples;
    
    std::vector<__hhash_digest> leaf_hashes;
    leaf_hashes.reserve(num_samples);
    
    std::cout << "[DatasetCommitment] Hashing " << num_samples << " samples..." << std::endl;
    
    for (size_t i = 0; i < static_cast<size_t>(num_samples); i++) {
        auto example = dataset.get(i);
        float label = example.target.template item<float>();
        __hhash_digest leaf_hash = hash_sample(example.data, label);
        leaf_hashes.push_back(leaf_hash);
        commitment.sample_hashes[i] = leaf_hash;
        
        if ((i + 1) % 1000 == 0) {
            std::cout << "[DatasetCommitment] Hashed " << (i + 1) << " samples..." << std::endl;
        }
    }
    
    size_t tree_size = 1;
    int depth = 0;
    while (tree_size < leaf_hashes.size()) {
        tree_size *= 2;
        depth++;
    }
    
    commitment.tree_depth = depth;
    
    __hhash_digest zero_hash;
    memset(&zero_hash, 0, sizeof(zero_hash));
    while (leaf_hashes.size() < tree_size) {
        leaf_hashes.push_back(zero_hash);
    }
    
    commitment.tree.resize(depth + 1);
    commitment.tree[0] = leaf_hashes;
    
    std::cout << "[DatasetCommitment] Building Merkle tree (depth=" << depth << ")..." << std::endl;
    
    for (int level = 1; level <= depth; level++) {
        size_t prev_level_size = commitment.tree[level - 1].size();
        size_t curr_level_size = prev_level_size / 2;
        commitment.tree[level].resize(curr_level_size);
        
        for (size_t i = 0; i < curr_level_size; i++) {
            __hhash_digest data[2];
            data[0] = commitment.tree[level - 1][i * 2];
            data[1] = commitment.tree[level - 1][i * 2 + 1];
            my_hhash(data, &commitment.tree[level][i]);
        }
    }
    
    commitment.root_hash = commitment.tree[depth][0];
    
    std::cout << "[DatasetCommitment] Dataset committed. Root hash: 0x";
    uint64_t* root_words = reinterpret_cast<uint64_t*>(&commitment.root_hash);
    for (int i = 0; i < 4; i++) {
        std::cout << std::hex << std::setfill('0') << std::setw(16) << root_words[i];
    }
    std::cout << std::dec << std::endl;
    
    return commitment;
}

// Commit dataset to Merkle tree (Synthetic logistic)
DatasetCommitment commit_dataset(SyntheticLogisticDataset& dataset, int num_samples) {
    DatasetCommitment commitment;
    
    auto dataset_size = dataset.size();
    if (!dataset_size.has_value()) {
        throw std::runtime_error("Dataset size unknown");
    }
    
    size_t total_samples = dataset_size.value();
    if (num_samples < 0 || num_samples > static_cast<int>(total_samples)) {
        num_samples = total_samples;
    }
    
    commitment.num_samples = num_samples;
    
    std::vector<__hhash_digest> leaf_hashes;
    leaf_hashes.reserve(num_samples);
    
    std::cout << "[DatasetCommitment] Hashing " << num_samples << " samples..." << std::endl;
    
    for (size_t i = 0; i < static_cast<size_t>(num_samples); i++) {
        auto example = dataset.get(i);
        float label = example.target.template item<float>();
        __hhash_digest leaf_hash = hash_sample(example.data, label);
        leaf_hashes.push_back(leaf_hash);
        commitment.sample_hashes[i] = leaf_hash;
        
        if ((i + 1) % 1000 == 0) {
            std::cout << "[DatasetCommitment] Hashed " << (i + 1) << " samples..." << std::endl;
        }
    }
    
    size_t tree_size = 1;
    int depth = 0;
    while (tree_size < leaf_hashes.size()) {
        tree_size *= 2;
        depth++;
    }
    
    commitment.tree_depth = depth;
    
    __hhash_digest zero_hash;
    memset(&zero_hash, 0, sizeof(zero_hash));
    while (leaf_hashes.size() < tree_size) {
        leaf_hashes.push_back(zero_hash);
    }
    
    commitment.tree.resize(depth + 1);
    commitment.tree[0] = leaf_hashes;
    
    std::cout << "[DatasetCommitment] Building Merkle tree (depth=" << depth << ")..." << std::endl;
    
    for (int level = 1; level <= depth; level++) {
        size_t prev_level_size = commitment.tree[level - 1].size();
        size_t curr_level_size = prev_level_size / 2;
        commitment.tree[level].resize(curr_level_size);
        
        for (size_t i = 0; i < curr_level_size; i++) {
            __hhash_digest data[2];
            data[0] = commitment.tree[level - 1][i * 2];
            data[1] = commitment.tree[level - 1][i * 2 + 1];
            my_hhash(data, &commitment.tree[level][i]);
        }
    }
    
    commitment.root_hash = commitment.tree[depth][0];
    
    std::cout << "[DatasetCommitment] Dataset committed. Root hash: 0x";
    uint64_t* root_words = reinterpret_cast<uint64_t*>(&commitment.root_hash);
    for (int i = 0; i < 4; i++) {
        std::cout << std::hex << std::setfill('0') << std::setw(16) << root_words[i];
    }
    std::cout << std::dec << std::endl;
    
    return commitment;
}

// Generate membership proof for a single sample
MembershipProof prove_membership(const DatasetCommitment& commitment, size_t sample_index) {
    MembershipProof proof;
    
    if (commitment.sample_hashes.find(sample_index) == commitment.sample_hashes.end()) {
        throw std::runtime_error("Sample index out of range");
    }
    
    proof.leaf_hash = commitment.sample_hashes.at(sample_index);
    
    // Calculate position in padded tree
    size_t tree_size = commitment.tree[0].size();
    size_t position = tree_size + sample_index;  // Position in full tree (1-indexed from root)
    proof.position = static_cast<int>(position);
    
    // Build path from leaf to root
    size_t current_pos = sample_index;
    proof.path.clear();
    proof.directions.clear();
    
    for (int level = 0; level < commitment.tree_depth; level++) {
        size_t sibling_pos = current_pos ^ 1;  // XOR to get sibling
        proof.path.push_back(commitment.tree[level][sibling_pos]);
        proof.directions.push_back((current_pos & 1) == 0);  // false = left, true = right
        
        current_pos = current_pos / 2;  // Move to parent
    }
    
    return proof;
}

// Generate membership proofs for a batch
BatchMembershipProofs prove_batch_membership(
    const DatasetCommitment& commitment,
    const std::vector<size_t>& batch_indices) {
    
    BatchMembershipProofs batch_proofs;
    batch_proofs.commitment = commitment;
    batch_proofs.proofs.reserve(batch_indices.size());
    
    for (size_t idx : batch_indices) {
        batch_proofs.proofs.push_back(prove_membership(commitment, idx));
    }
    
    return batch_proofs;
}

// Verify membership proof
bool verify_membership(const DatasetCommitment& commitment, const MembershipProof& proof) {
    // Start with leaf hash
    __hhash_digest current_hash = proof.leaf_hash;
    
    // Verify path from leaf to root
    for (size_t i = 0; i < proof.path.size(); i++) {
        __hhash_digest data[2];
        bool is_right = proof.directions[i];
        
        if (is_right) {
            data[0] = current_hash;
            data[1] = proof.path[i];
        } else {
            data[0] = proof.path[i];
            data[1] = current_hash;
        }
        
        my_hhash(data, &current_hash);
    }
    
    // Check if final hash matches root
    // Use the equals function from my_hhash.h for proper hash comparison
    return equals(commitment.root_hash, current_hash);
}

// Save commitment to file (includes sample_hashes for membership proofs)
void save_commitment(const DatasetCommitment& commitment, const std::string& filepath) {
    std::ofstream out(filepath, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }
    
    // Write metadata
    out.write(reinterpret_cast<const char*>(&commitment.num_samples), sizeof(commitment.num_samples));
    out.write(reinterpret_cast<const char*>(&commitment.tree_depth), sizeof(commitment.tree_depth));
    
    // Write root hash
    out.write(reinterpret_cast<const char*>(&commitment.root_hash), sizeof(commitment.root_hash));
    
    // Write sample_hashes map (needed for membership proofs)
    size_t map_size = commitment.sample_hashes.size();
    out.write(reinterpret_cast<const char*>(&map_size), sizeof(map_size));
    for (const auto& [idx, hash] : commitment.sample_hashes) {
        out.write(reinterpret_cast<const char*>(&idx), sizeof(idx));
        out.write(reinterpret_cast<const char*>(&hash), sizeof(hash));
    }
    
    // Write tree structure (needed for membership proofs)
    size_t tree_levels = commitment.tree.size();
    out.write(reinterpret_cast<const char*>(&tree_levels), sizeof(tree_levels));
    for (const auto& level : commitment.tree) {
        size_t level_size = level.size();
        out.write(reinterpret_cast<const char*>(&level_size), sizeof(level_size));
        for (const auto& hash : level) {
            out.write(reinterpret_cast<const char*>(&hash), sizeof(hash));
        }
    }
    
    std::cout << "[DatasetCommitment] Saved commitment to " << filepath << std::endl;
}

// Load commitment from file
DatasetCommitment load_commitment(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open file for reading: " + filepath);
    }
    
    DatasetCommitment commitment;
    
    // Read metadata
    in.read(reinterpret_cast<char*>(&commitment.num_samples), sizeof(commitment.num_samples));
    in.read(reinterpret_cast<char*>(&commitment.tree_depth), sizeof(commitment.tree_depth));
    
    // Read root hash
    in.read(reinterpret_cast<char*>(&commitment.root_hash), sizeof(commitment.root_hash));
    
    // Try to read sample_hashes map (may not exist in old format files)
    size_t map_size = 0;
    if (in.read(reinterpret_cast<char*>(&map_size), sizeof(map_size))) {
        for (size_t i = 0; i < map_size; i++) {
            size_t idx;
            __hhash_digest hash;
            in.read(reinterpret_cast<char*>(&idx), sizeof(idx));
            in.read(reinterpret_cast<char*>(&hash), sizeof(hash));
            commitment.sample_hashes[idx] = hash;
        }
        
        // Try to read tree structure
        size_t tree_levels = 0;
        if (in.read(reinterpret_cast<char*>(&tree_levels), sizeof(tree_levels))) {
            commitment.tree.resize(tree_levels);
            for (size_t level = 0; level < tree_levels; level++) {
                size_t level_size = 0;
                in.read(reinterpret_cast<char*>(&level_size), sizeof(level_size));
                commitment.tree[level].resize(level_size);
                for (size_t j = 0; j < level_size; j++) {
                    in.read(reinterpret_cast<char*>(&commitment.tree[level][j]), sizeof(__hhash_digest));
                }
            }
        }
    }
    
    std::cout << "[DatasetCommitment] Loaded commitment from " << filepath << std::endl;
    std::cout << "[DatasetCommitment] Root hash: 0x";
    const uint64_t* root_words = reinterpret_cast<const uint64_t*>(&commitment.root_hash);
    for (int i = 0; i < 4; i++) {
        std::cout << std::hex << std::setfill('0') << std::setw(16) << root_words[i];
    }
    std::cout << std::dec << std::endl;
    
    // Report if membership proofs are available
    if (commitment.sample_hashes.empty()) {
        std::cout << "[DatasetCommitment] Warning: No sample hashes loaded (old format file)" << std::endl;
        std::cout << "[DatasetCommitment] Delete the file and rerun to enable membership proofs" << std::endl;
    } else {
        std::cout << "[DatasetCommitment] Loaded " << commitment.sample_hashes.size() 
                  << " sample hashes for membership proofs" << std::endl;
    }
    
    return commitment;
}

