// Minimal utilities for proof generation without main()
#include "proof_utils.h"
#include "utils.hpp"
#include "mimc.h"
#include "polynomial.h"
#include "quantization.h"
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace std;

// Global variables
F current_randomness;
unsigned long int mul_counter = 0;
vector<F> x_transcript;
vector<F> y_transcript;
vector<F> SHA;
vector<F> H;

// Implementation of init_SHA extracted from main.cpp
void init_SHA(){
	for(int i = 0; i < 64; i++){
		SHA.push_back(F(random()%(1ULL<<32)));
	}
	for(int i = 0; i < 8; i++){
		H.push_back(F(random()%(1ULL<<32)));
	}
}

// Forward declarations
vector<vector<F>> prepare_matrixes(vector<vector<F>> &M1, vector<vector<F>> &M2, vector<F> r1, vector<F> r2);
struct proof generate_2product_sumcheck_proof(vector<F> v1, vector<F> v2, F previous_r);
vector<vector<F>> generate_bit_matrix(vector<F> bits, int domain);
void precompute_beta(vector<F> r, vector<F> &B);

// Implementation of prepare_matrixes extracted from main.cpp
vector<vector<F>> prepare_matrixes(vector<vector<F>> &M1, vector<vector<F>> &M2, vector<F> r1, vector<F> r2){
	vector<vector<F>> V;
	
	V.push_back(prepare_matrix(transpose(M1),r1));
	V.push_back(prepare_matrix(transpose(M2),r2));
	
	return V;
}

// Implementation of generate_2product_sumcheck_proof extracted from main.cpp
struct proof generate_2product_sumcheck_proof(vector<F> v1, vector<F> v2, F previous_r){
	struct proof Pr;
	vector<F> r_sc;
	F rand = mimc_hash(previous_r,F(0));
	vector<quadratic_poly> p;
	int rounds = int(log2(v1.size()));
	for(int i = 0; i < rounds; i++){
		quadratic_poly poly = quadratic_poly(F_ZERO,F_ZERO,F_ZERO);
		quadratic_poly temp_poly = quadratic_poly(F_ZERO,F_ZERO,F_ZERO);
		linear_poly l1,l2;
		int L = 1 << (rounds - 1-i);
		for(int j = 0; j < L; j++){
			l1 = linear_poly(v1[2*j+1] - v1[2*j],v1[2*j]);
			l2 = linear_poly(v2[2*j+1] - v2[2*j],v2[2*j]);
			temp_poly = l1*l2;
			poly = poly + temp_poly;
			v1[j] = v1[2*j] + rand*(v1[2*j+1]-v1[2*j]);
			v2[j] = v2[2*j] + rand*(v2[2*j+1]-v2[2*j]);
		
		}
		r_sc.push_back(rand);
		vector<F> input;
		mimc_hash(poly.a,current_randomness);
		mimc_hash(poly.b,current_randomness);
		mimc_hash(poly.c,current_randomness);
		
		rand = current_randomness;
		p.push_back(poly);
	}
	mimc_hash(v1[0],current_randomness);
	mimc_hash(v2[0],current_randomness);
	
	Pr.vr.push_back(v1[0]);
	Pr.vr.push_back(v2[0]);
	Pr.q_poly = p;
	Pr.randomness.push_back(r_sc);
	Pr.sc_challenges.push_back(r_sc);
	return Pr;
}

// Implementation of _prove_matrix2matrix extracted from main.cpp
struct proof _prove_matrix2matrix(vector<vector<F>> M1, vector<vector<F>> M2, vector<F> r_eval, F previous_sum){
	struct proof Pr;
	int r1_len = (int)log2(M1.size());
	int r2_len = (int)log2(M2.size());
	assert((int)r_eval.size() == r1_len + r2_len);

	vector<F> r2(r_eval.begin(), r_eval.begin()+ r2_len); 
	vector<F> r1(r_eval.begin()+r2_len, r_eval.end());

	clock_t start = clock();
	vector<vector<F>> V = prepare_matrixes(M1,M2,r1,r2);
	// Note: total_prove timing removed for standalone use

	start = clock();
	if(V[0].size() != 1){
		Pr = generate_2product_sumcheck_proof(V[0],V[1],r_eval.back());
		Pr.randomness.push_back(r1);
		Pr.randomness.push_back(r2);

		F expected_sum = previous_sum * F(1ULL<<Q);
		F actual_sum = Pr.q_poly[0].eval(0) + Pr.q_poly[0].eval(1);
		if(expected_sum != actual_sum){
			printf("Error in Matrix2Matrix multiplication\n");
			printf("  previous_sum = %lld\n", previous_sum.toint128());
			printf("  Q = %d\n", Q);
			printf("  expected_sum = %lld\n", expected_sum.toint128());
			printf("  actual_sum = %lld\n", actual_sum.toint128());
			printf("  difference = %lld\n", (expected_sum - actual_sum).toint128());
			exit(-1);
		}
		Pr.type = MATMUL_PROOF;
    }else{
        
        F product = V[0][0]*V[1][0];
        Pr.q_poly.push_back(quadratic_poly(F_ZERO, product, F_ZERO));

        Pr.randomness.push_back(vector<F>()); // no sumcheck randomness
        if (r1.empty()) {
            r1.push_back(F_ZERO);
        }
        if (r2.empty()) {
            r2.push_back(F_ZERO);
        }
        Pr.randomness.push_back(r1);
        Pr.randomness.push_back(r2);
        if(previous_sum*F(1ULL<<Q) != product){
            printf("Error in Matrix2Matrix multiplication\n");
            exit(-1);
        }
        Pr.type = MATMUL_PROOF;
	}
	// Note: total_prove timing removed for standalone use
	return Pr;
}

// Implementation of generate_3product_sumcheck_proof extracted from main.cpp
struct proof generate_3product_sumcheck_proof(vector<F> &v1, vector<F> &v2, vector<F> &v3, F previous_r){
	struct proof Pr;
	vector<F> r_sc;
	int rounds = int(log2(v1.size()));
	vector<cubic_poly> p;
	F rand = previous_r;

	for(int i = 0; i < rounds; i++){
		cubic_poly poly = cubic_poly(F_ZERO,F_ZERO,F_ZERO,F_ZERO);
		cubic_poly temp_poly = cubic_poly(F_ZERO,F_ZERO,F_ZERO,F_ZERO);
		linear_poly l1,l2,l3;
		
		int L = 1 << (rounds -1- i);
		for(int j = 0; j < L; j++){
			if(!(v2[2*j] == F(0) && v2[2*j+1] == F(0))){
				l1 = linear_poly(v1[2*j+1] - v1[2*j],v1[2*j]);
				l2 = linear_poly(v2[2*j+1] - v2[2*j],v2[2*j]);
				l3 = linear_poly(v3[2*j+1] - v3[2*j],v3[2*j]);
				temp_poly = l1*l2*l3;
				poly = poly + temp_poly;
			}
			v1[j] = v1[2*j] + rand*(v1[2*j+1]-v1[2*j]);
			if(v2[2*j] == F(0) && v2[2*j+1] == F(0)){
				v2[j] = F(0);
				v3[j] = F(1);
			}else{
				v2[j] = v2[2*j] + rand*(v2[2*j+1]-v2[2*j]);
				v3[j] = F(1)-v2[j];
			}
		}
		r_sc.push_back(rand);
		mimc_hash(poly.a,current_randomness);
		mimc_hash(poly.b,current_randomness);
		mimc_hash(poly.c,current_randomness);
		mimc_hash(poly.d,current_randomness);
		rand = current_randomness;
		p.push_back(poly);
	}
	Pr.c_poly = p;
	mimc_hash(v2[0],current_randomness);
	mimc_hash(v1[0],current_randomness);
	
	Pr.vr.push_back(v2[0]);
	Pr.vr.push_back(v1[0]);
	Pr.randomness.push_back(r_sc);
	Pr.sc_challenges.push_back(r_sc);
	
	return Pr;
}

// Implementation of generate_bit_matrix extracted from main.cpp
vector<vector<F>> generate_bit_matrix(vector<F> bits, int domain){
	vector<vector<F>> M;
	int elements = bits.size()/domain; 
	M.resize(domain);
	for(int i = 0; i < M.size(); i++){
		for(int j = 0; j < elements; j++){
			M[i].push_back(bits[j*domain+i]);
		}
	}
	return M;
}

// Implementation of _prove_bit_decomposition extracted from main.cpp
struct proof _prove_bit_decomposition(vector<F> bits, vector<F> r1, F previous_sum, int domain){
	vector<F> powers;
	powers.push_back(F(1));
	for(int i = 1; i < domain; i++){
		powers.push_back(F(2)*powers[i-1]);
	}

	clock_t start = clock(); 
	vector<vector<F>> M = generate_bit_matrix(bits,domain);
	vector<F> v1 = prepare_matrix(M,r1);
	// Note: total_prove timing removed for standalone use

	start = clock(); 
	struct proof Pr1 = generate_2product_sumcheck_proof(v1,powers,r1[r1.size()-1]);
	// Note: total_prove timing removed for standalone use

	if(previous_sum != Pr1.q_poly[0].eval(0) + Pr1.q_poly[0].eval(1)){
		printf("Error in bit_decomposition\n");
		exit(-1);
	}
	vector<F> r2 = generate_randomness(int(log2(bits.size())),r1[r1.size()-1]);
	vector<F> beta;
	
	start = clock(); 
	precompute_beta(r2,beta);
	// Note: total_prove timing removed for standalone use

	vector<F> inv_bits;
	for(int i  = 0; i < bits.size(); i++){
		inv_bits.push_back(F(1) - bits[i]);
	}

	start = clock(); 
	struct proof Pr2 = generate_3product_sumcheck_proof(beta,bits,inv_bits,r2[r2.size()-1]);
	// Note: total_prove timing removed for standalone use

	struct proof Pr;
	Pr.randomness.push_back(r1);
	Pr.randomness.push_back(r2);
	Pr.randomness.push_back(Pr1.randomness[0]);
	Pr.randomness.push_back(Pr2.randomness[0]);
	Pr.vr.insert(Pr.vr.end(),Pr1.vr.begin(),Pr1.vr.end());
	Pr.vr.insert(Pr.vr.end(),Pr2.vr.begin(),Pr2.vr.end());
	Pr.q_poly = Pr1.q_poly;
	Pr.c_poly = Pr2.c_poly;
	Pr.type = RANGE_PROOF;
	Pr.w_hashes.insert(Pr.w_hashes.end(),Pr1.w_hashes.begin(),Pr1.w_hashes.end());
	Pr.w_hashes.insert(Pr.w_hashes.end(),Pr2.w_hashes.begin(),Pr2.w_hashes.end());
	
	return Pr;
}

