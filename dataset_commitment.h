#pragma once

#include <torch/torch.h>
#include "cifar10_dataset.h"
#include "synthetic_linear_dataset.h"
#include "synthetic_logistic_dataset.h"
#include "Summer code/merkle_tree.h"
#include "Summer code/config_pc.hpp"
#include "veridp_utils.h"
#include <vector>
#include <string>
#include <map>

struct DatasetCommitment {
    __hhash_digest root_hash;
    std::vector<std::vector<__hhash_digest>> tree;
    std::map<size_t, __hhash_digest> sample_hashes;
    size_t num_samples;
    int tree_depth;
};

struct MembershipProof {
    __hhash_digest leaf_hash;
    int position;
    std::vector<__hhash_digest> path;
    std::vector<bool> directions;
};

struct BatchMembershipProofs {
    std::vector<MembershipProof> proofs;
    DatasetCommitment commitment;
};

DatasetCommitment commit_dataset(torch::data::datasets::MNIST& dataset, int num_samples = -1);
DatasetCommitment commit_dataset(cifar10_dataset::CIFAR10& dataset, int num_samples = -1);
DatasetCommitment commit_dataset(SyntheticLinearDataset& dataset, int num_samples = -1);
DatasetCommitment commit_dataset(SyntheticLogisticDataset& dataset, int num_samples = -1);

__hhash_digest hash_sample(const torch::Tensor& image, int64_t label);
__hhash_digest hash_sample(const torch::Tensor& image, float label);

MembershipProof prove_membership(const DatasetCommitment& commitment, size_t sample_index);

BatchMembershipProofs prove_batch_membership(
    const DatasetCommitment& commitment,
    const std::vector<size_t>& batch_indices
);

bool verify_membership(const DatasetCommitment& commitment, const MembershipProof& proof);

void save_commitment(const DatasetCommitment& commitment, const std::string& filepath);

DatasetCommitment load_commitment(const std::string& filepath);
