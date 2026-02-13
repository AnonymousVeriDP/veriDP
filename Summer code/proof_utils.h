#pragma once
#include "config_pc.hpp"
#include "GKR.h"
#include <vector>

using namespace std;

// Global variable declaration
extern F current_randomness;
extern vector<F> x_transcript;
extern vector<F> y_transcript;

// Function declarations
struct proof _prove_matrix2matrix(vector<vector<F>> M1, vector<vector<F>> M2, 
                                 vector<F> r_eval, F previous_sum);
struct proof generate_3product_sumcheck_proof(vector<F> &v1, vector<F> &v2, vector<F> &v3, F previous_r);
struct proof _prove_bit_decomposition(vector<F> bits, vector<F> r1, F previous_sum, int domain);
void init_SHA();  // Initialize SHA constants

