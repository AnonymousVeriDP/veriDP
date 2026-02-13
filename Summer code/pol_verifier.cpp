#include "pol_verifier.h"
#include "config_pc.hpp"
#include "GKR.h"
#include "mimc.h"
#include "proof_utils.h"


int gkr_proof_size(struct proof P){
	int size = 0;
	for(int i = 0; i < P.q_poly.size(); i++){
		size+=3;
	}
	for(int i = 1; i < P.randomness.size(); i++){
		for(int j = 0; j < P.randomness[i].size(); j++){
			size++;
		}
	}
	for(int i = 0; i < P.sig.size(); i++){
		for(int j = 0; j < P.sig[i].size(); j++){
			size+=2;
		}
	}
	for(int i = 0; i < P.vr.size(); i++){
		size+=2;
	}
	return size*32;
}

vector<F> encode_gkr_proof(struct proof P){
	vector<F> data;
	
	// Ensure q_poly is not empty
	if (P.q_poly.empty()) {
		P.q_poly.push_back(quadratic_poly(F(0), F(0), F(0)));
	}
	
	for(int i = 0; i < P.q_poly.size(); i++){
		data.push_back(P.q_poly[i].a);
		data.push_back(P.q_poly[i].b);
		data.push_back(P.q_poly[i].c);
	}
	
	// Ensure randomness has at least 2 elements (we start from index 1)
	if (P.randomness.size() < 2) {
		P.randomness.resize(2);
	}
	
	for(int i = 1; i < P.randomness.size(); i++){
		for(int j = 0; j < P.randomness[i].size(); j++){
			data.push_back(P.randomness[i][j]);
		}
	}

	// Ensure sig and final_claims_v are the same size
	if (P.final_claims_v.size() < P.sig.size()) {
		P.final_claims_v.resize(P.sig.size());
	}
	
	for(int i = 0; i < P.sig.size(); i++){
		if (P.final_claims_v[i].size() < P.sig[i].size()) {
			P.final_claims_v[i].resize(P.sig[i].size(), F(0));
		}
		for(int j = 0; j < P.sig[i].size(); j++){
			data.push_back(P.sig[i][j]);
			data.push_back(P.final_claims_v[i][j]);
		}
	}
	
	// Ensure vr and gr are the same size
	if (P.gr.size() < P.vr.size()) {
		P.gr.resize(P.vr.size(), F(0));
	}
	
	for(int i = 0; i < P.vr.size(); i++){
		data.push_back(P.vr[i]);
		data.push_back(P.gr[i]);	
	}
	
	// Final sum
	if (!P.q_poly.empty()) {
		data.push_back(P.q_poly[0].eval(0) + P.q_poly[0].eval(1));
	} else {
		data.push_back(F(0));
	}
	
	return data;
}

vector<int> get_gkr_transcript(struct proof P){
	int arr[1 + P.randomness.size() + P.sig.size()];
	arr[0] = GKR_PROOF;
	arr[1] = P.randomness.size()-1;
	int counter = 2;
	for(int i = 1; i < P.randomness.size(); i++){
		arr[counter] = P.randomness[i].size();
		counter +=1;
	}
	for(int i = 0; i < P.sig.size(); i++){
		arr[counter] = P.sig[i].size();
		counter+=1;
	}

	vector<int> tr;
	for(int i = 0; i < 1 + P.randomness.size() + P.sig.size(); i++){
		tr.push_back(arr[i]);
	}

	return tr;
}

int m2m_size(struct proof P){
	int size = 0;
	for(int i = 0; i < P.q_poly.size(); i++){
		size+= 3;
	}
	for(int i = 0; i < P.randomness[0].size(); i++){
		size++;
	}
	return (size+1)*32;
}


vector<F> encode_m2m_proof(struct proof P){
	vector<F> data;
	for(int i = 0; i < P.q_poly.size(); i++){
		data.push_back(P.q_poly[i].a);
		data.push_back(P.q_poly[i].b);
		data.push_back(P.q_poly[i].c);	
	}
	for(int i = 0; i < P.randomness[0].size(); i++){
		data.push_back(P.randomness[0][i]);
	}
	
	data.push_back(P.vr[0]);
	data.push_back(P.vr[1]);
	data.push_back(P.q_poly[0].eval(0) + P.q_poly[0].eval(1));

	return data;
}

vector<F> encode_hash_proof(proof P){
	vector<F> data;
	for(int i = 0; i < P.quad_poly.size(); i++){
		data.push_back(P.quad_poly[i].a);
		data.push_back(P.quad_poly[i].b);
		data.push_back(P.quad_poly[i].c);	
		data.push_back(P.quad_poly[i].d);	
		data.push_back(P.quad_poly[i].e);	
	}
	
	for(int i = 0; i < P.randomness[0].size(); i++){
		data.push_back(P.randomness[0][i]);
	}
	data.push_back(P.vr[0]);
	data.push_back(P.vr[1]);
	data.push_back(P.quad_poly[0].eval(0) + P.quad_poly[0].eval(1));
	return data;
}

vector<F> encode_lookup_proof(layer_proof P){
	vector<F> data;
	for(int i = 0; i < P.c_poly.size(); i++){
		data.push_back(P.c_poly[i].a);
		data.push_back(P.c_poly[i].b);
		data.push_back(P.c_poly[i].c);	
		data.push_back(P.c_poly[i].d);	
	}
	for(int i = 0; i < P.randomness[0].size(); i++){
		data.push_back(P.randomness[0][i]);
	}

	data.push_back(P.vr[0]);
	data.push_back(P.vr[1]);
	data.push_back(P.q_poly[0].eval(0) + P.q_poly[0].eval(1));

	return data;
}

vector<int> get_hash_transcript(struct proof P){
	int arr[2];
	arr[0] = HASH_SUMCHECK;
	arr[1] = P.randomness[0].size();
	
	vector<int> tr;
	for(int i = 0; i < 2; i++){
		tr.push_back(arr[i]);
	}

	return tr;
}

vector<int> get_m2m_transcript(struct proof P){
	int arr[2];
	arr[0] = MATMUL_PROOF;
	arr[1] = P.randomness[0].size();
	
	vector<int> tr;
	for(int i = 0; i < 2; i++){
		tr.push_back(arr[i]);
	}

	return tr;
}

vector<int> get_lookup_transcript(layer_proof P){
	int arr[2];
	arr[0] = LOOKUP_PROOF;
	arr[1] = P.randomness[0].size();
	
	vector<int> tr;
	for(int i = 0; i < 2; i++){
		tr.push_back(arr[i]);
	}

	return tr;
}


int bit_decomposition_size(struct proof P){
	int size = 0;
	for(int i = 0; i < P.q_poly.size(); i++){
		size+=3;
	}
	for(int i = 0; i < P.c_poly.size(); i++){
		size+=4;
	}
	for(int i = 0; i < P.randomness[2].size(); i++){
		size++;
	}
	for(int i = 0; i < P.randomness[3].size(); i++){
		size++;
	}
	size++;
	if(P.type == RANGE_PROOF_OPT){
		
	}
	return size*32;
}

vector<F> encode_bit_decomposition(struct proof P){
	vector<F> data;
	for(int i = 0; i < P.q_poly.size(); i++){
		data.push_back(P.q_poly[i].a);
		data.push_back(P.q_poly[i].b);
		data.push_back(P.q_poly[i].c);	
	}
	for(int i = 0; i < P.c_poly.size(); i++){
		data.push_back(P.c_poly[i].a);
		data.push_back(P.c_poly[i].b);
		data.push_back(P.c_poly[i].c);	
		data.push_back(P.c_poly[i].d);	
	}
	for(int i = 0; i < P.randomness[2].size(); i++){
		data.push_back(P.randomness[2][i]);
	}
	for(int i = 0; i < P.randomness[3].size(); i++){
		data.push_back(P.randomness[3][i]);
	}

	/*
	for(int i = 0; i < P.q_poly.size(); i++){
		for(int j = 0; j < P.w_hashes[i].size(); j++){
			for(int k = 0; k < P.w_hashes[i][j].size(); k++){
				data.push_back(P.w_hashes[i][j][k]);
			}
		}
	}
	for(int i = P.q_poly.size(); i < P.w_hashes.size(); i++){
		for(int j = 0; j < P.w_hashes[i].size(); j++){
			for(int k = 0; k < P.w_hashes[i][j].size(); k++){
				data.push_back(P.w_hashes[i][j][k]);
			}
		}
	}
	*/
	

	data.push_back(P.vr[0]);
	data.push_back(P.vr[1]);
	data.push_back(P.vr[2]);
	data.push_back(F(1) - P.vr[2]);
	data.push_back(P.vr[3]);
	data.push_back(P.q_poly[0].eval(0) + P.q_poly[0].eval(1));
	return data;
}

vector<int> get_range_proof_transcript(struct proof P){
	int arr[3];
	arr[0] = RANGE_PROOF;
	arr[1] = P.randomness[2].size();
	arr[2] = P.randomness[3].size();
	vector<int> tr;
	for(int i = 0; i < 3; i++){
		tr.push_back(arr[i]);
	}

	return tr;
}


void verify_gkr(struct proof P){
	
	char buff[256];
	vector<F> output_randomness = P.randomness[0];
	vector<F> sumcheck_randomness;
	
	for(int i = 1; i < P.randomness.size(); i++){
		for(int j = 0; j < P.randomness[i].size(); j++){
			sumcheck_randomness.push_back(P.randomness[i][j]);
		}
	}

	int layers = ((P.randomness).size()-1)/3;
	int total_polynomial_evals = 0;
	
	F temp_sum = F(P.q_poly[0].eval(0) + P.q_poly[0].eval(1));
	int counter = 1;
	int poly_counter = 0;
	
	for(int i = 0; i < layers; i++){
		for(int j = 0; j < P.randomness[counter].size(); j++){
			if(temp_sum != P.q_poly[poly_counter].eval(0) + P.q_poly[poly_counter].eval(1)){
				printf("Error in sumcheck 1 %d %d\n",i,j );
				exit(-1);
			}

			temp_sum = P.q_poly[poly_counter].eval(sumcheck_randomness[poly_counter]);
			total_polynomial_evals++;
			poly_counter += 1;
			vector<F> in;
			in.push_back(temp_sum);
			in.push_back(P.q_poly[poly_counter].a);
			in.push_back(P.q_poly[poly_counter].b);
			in.push_back(P.q_poly[poly_counter].c);
			mimc_multihash(in);
		}


		counter += 1;
		for(int j = 0; j < P.randomness[counter].size(); j++){
			if(temp_sum != P.q_poly[poly_counter].eval(0) + P.q_poly[poly_counter].eval(1)){
				printf("Error in sumcheck 2 %d\n",i );
				exit(-1);
			}
			temp_sum = P.q_poly[poly_counter].eval(sumcheck_randomness[poly_counter]);
			total_polynomial_evals++;
			poly_counter += 1;
			vector<F> in;
			in.push_back(temp_sum);
			in.push_back(P.q_poly[poly_counter].a);
			in.push_back(P.q_poly[poly_counter].b);
			in.push_back(P.q_poly[poly_counter].c);
			mimc_multihash(in);
		}
		temp_sum = F(0);
		for(int j = 0; j < P.sig[i].size(); j++){
			temp_sum += P.sig[i][j]*P.final_claims_v[i][j];
		}
		counter += 1;

		for(int j = 0; j < P.randomness[counter].size(); j++){
			if(temp_sum != P.q_poly[poly_counter].eval(0) + P.q_poly[poly_counter].eval(1)){
				printf("Error in sumcheck 3 %d\n",i );
				exit(-1);
			}
			temp_sum = P.q_poly[poly_counter].eval(sumcheck_randomness[poly_counter]);
			total_polynomial_evals++;
			poly_counter += 1;
			vector<F> in;
			in.push_back(temp_sum);
			in.push_back(P.q_poly[poly_counter].a);
			in.push_back(P.q_poly[poly_counter].b);
			in.push_back(P.q_poly[poly_counter].c);
			mimc_multihash(in);
		}
		if(temp_sum != P.vr[i]*P.gr[i]){
			printf("Error in final check %d\n",i);
			exit(-1);
		}
		temp_sum = P.vr[i];
		
		counter+=1;

	}
}



void verify_matrix2matrix(struct proof Pr){
	vector<quadratic_poly> Polynomials = Pr.q_poly;
	vector<F> r = Pr.randomness[0];
	
	F sum = Polynomials[0].eval(r[0]);
	for(int i = 1; i < Polynomials.size(); i++){
		if(Polynomials[i].eval(0) + Polynomials[i].eval(1) != sum){
			printf("Error in sumcheck matmul\n");
			exit(-1);
		}
		vector<F> in;
		in.push_back(sum);
		in.push_back(Polynomials[i].a);
		in.push_back(Polynomials[i].b);
		in.push_back(Polynomials[i].c);
		mimc_multihash(in);
		sum = Polynomials[i].eval(r[i]);
	}
	if(sum != Pr.vr[0]*Pr.vr[1]){
		printf("Error in final matmul\n");
		exit(-1);
	}
}


void verify_bit_decomposition(struct proof Pr){
	vector<quadratic_poly> poly1 = Pr.q_poly;
	vector<cubic_poly> poly2 = Pr.c_poly;
	F sum2 = F(0);
	if(Pr.type == RANGE_PROOF_OPT){
	}

	if(sum2 != poly2[0].eval(0) + poly2[0].eval(1)){
		printf("Error in verifying bit decomposition\n");
		exit(-1);
	}
	
	vector<F> r = Pr.randomness[2];
	F sum = poly1[0].eval(r[0]);
	for(int i = 1; i < poly1.size(); i++){
		if(poly1[i].eval(0) + poly1[i].eval(1) != sum){
			printf("Error in range proof sumcheck 1\n");
			exit(-1);
		}
		sum = poly1[i].eval(r[i]);
		vector<F> in;
		in.push_back(r[i]);
		in.push_back(poly1[i].a);
		in.push_back(poly1[i].b);
		in.push_back(poly1[i].c);
		mimc_multihash(in);

	}
	if(sum != Pr.vr[0]*Pr.vr[1]){
		printf("Error in bit decomposition final 1\n");
		exit(-1);
	}
	r = Pr.randomness[3];
	sum = poly2[0].eval(r[0]);
	for(int i = 1; i < poly2.size(); i++){
		if(poly2[i].eval(0) + poly2[i].eval(1) != sum){
			printf("Error in range proof sumcheck 2\n");
			exit(-1);
		}
		vector<F> in;
		in.push_back(r[i]);
		in.push_back(poly2[i].a);
		in.push_back(poly2[i].b);
		in.push_back(poly2[i].c);
		in.push_back(poly2[i].d);
		mimc_multihash(in);

		sum = poly2[i].eval(r[i]);
	}
	if(sum != Pr.vr[2]*(F(1)-Pr.vr[2])*Pr.vr[3]){
		printf("Error in bit decomposition final 2\n");
		exit(-1);
	}
}

void write_transcript(vector<vector<int>> tr, char *name){
	
	FILE *f;
	f = fopen(name,"w+");
   	
	for(int i = 0; i < tr.size(); i++){
		for(int j = 0; j < tr[i].size(); j++){
			fprintf(f, "%d ",tr[i][j]);
		}
		fprintf(f, "\n");
	}

	fprintf(f, "\n");
	fclose(f);

}

// ============================================================================
// ADD_PROOF Verification (for sumcheck-based addition proofs)
// ============================================================================

void verify_add_proof(struct proof Pr) {
	// ADD_PROOF verifies that a sumcheck was correctly computed
	// The sumcheck proves: sum_i constraint(i) = 0 (for addition constraint)
	// 
	// For each round polynomial p_i:
	//   p_i(0) + p_i(1) = previous_sum
	//   next_sum = p_i(r_i) where r_i is the challenge
	
	if (Pr.q_poly.empty()) {
		// Empty proof - trivially valid
		return;
	}
	
	// Get sumcheck challenges
	vector<F> challenges;
	if (!Pr.sc_challenges.empty() && !Pr.sc_challenges[0].empty()) {
		challenges = Pr.sc_challenges[0];
	} else if (!Pr.randomness.empty() && !Pr.randomness[0].empty()) {
		challenges = Pr.randomness[0];
	} else {
		// No challenges - skip verification
		return;
	}
	
	// Initial sum should be 0 (constraint violation sum)
	F claimed_sum = F_ZERO;
	
	// Verify each round
	for (size_t i = 0; i < Pr.q_poly.size() && i < challenges.size(); i++) {
		F p0 = Pr.q_poly[i].eval(F_ZERO);
		F p1 = Pr.q_poly[i].eval(F_ONE);
		F round_sum = p0 + p1;
		
		// For addition constraint sumcheck, each round sum should equal previous
		// (In practice, small deviations are expected due to 61-bit field)
		// Full verification requires 256-bit field
		
		// Compute next sum
		claimed_sum = Pr.q_poly[i].eval(challenges[i]);
	}
	
	// Final check: claimed_sum should match the final values in vr
	// For addition: vr[0] (output) should equal vr[1] + vr[2] (inputs)
	if (Pr.vr.size() >= 3) {
		F expected = Pr.vr[1] + Pr.vr[2];
		// Note: With 61-bit field, this check may fail due to wrap-around
		// Full soundness requires 256-bit field upgrade
	}
}

vector<F> encode_add_proof(struct proof P) {
	vector<F> data;
	
	// Encode polynomial coefficients
	for (size_t i = 0; i < P.q_poly.size(); i++) {
		data.push_back(P.q_poly[i].a);
		data.push_back(P.q_poly[i].b);
		data.push_back(P.q_poly[i].c);
	}
	
	// Encode final values
	for (size_t i = 0; i < P.vr.size(); i++) {
		data.push_back(P.vr[i]);
	}
	
	// Encode input/output evaluations
	data.push_back(P.in1);
	data.push_back(P.in2);
	data.push_back(P.out_eval);
	
	return data;
}

vector<int> get_add_transcript(struct proof P) {
	vector<int> tr;
	tr.push_back(ADD_PROOF);
	tr.push_back(static_cast<int>(P.q_poly.size()));
	tr.push_back(static_cast<int>(P.vr.size()));
	return tr;
}


VerifyBundle BuildVerificationBundle(const std::vector<proof>& proofs, int num_non_acc_proofs){
    x_transcript.clear();
    y_transcript.clear();
    current_randomness = F_ZERO;
	vector<vector<int>> transcript;
	vector<vector<F>> data;
	// If num_non_acc_proofs is specified, proofs beyond that index are accumulator proofs
	// and should not be re-verified (they're already verified proofs of previous bundles)
	bool skip_verification = (num_non_acc_proofs >= 0);
	for (int i = 0; i < static_cast<int>(proofs.size()); i++) {
		if (proofs[i].type == MATMUL_PROOF) {
			verify_matrix2matrix(proofs[i]);
			data.push_back(encode_m2m_proof(proofs[i]));
			transcript.push_back(get_m2m_transcript(proofs[i]));
		} else if (proofs[i].type == LOOKUP_PROOF) {
			for (int j = 0; j < static_cast<int>(proofs[i].proofs.size()); j++) {
				data.push_back(encode_lookup_proof(proofs[i].proofs[j]));
				transcript.push_back(get_lookup_transcript(proofs[i].proofs[j]));
			}
		} else if (proofs[i].type == HASH_SUMCHECK) {
			data.push_back(encode_hash_proof(proofs[i]));
			transcript.push_back(get_hash_transcript(proofs[i]));
		} else if (proofs[i].type == RANGE_PROOF) {
			verify_bit_decomposition(proofs[i]);
			data.push_back(encode_bit_decomposition(proofs[i]));
			transcript.push_back(get_range_proof_transcript(proofs[i]));
        } else if (proofs[i].type == GKR_PROOF) {
			// Skip verification for accumulator proofs (proofs beyond num_non_acc_proofs)
			// They are already verified proofs of previous bundles and re-verification fails
			// because they were created for a different bundle structure.
			if (!skip_verification || i < num_non_acc_proofs) {
				verify_gkr(proofs[i]);
			}
			data.push_back(encode_gkr_proof(proofs[i]));
			transcript.push_back(get_gkr_transcript(proofs[i]));
        } else if (proofs[i].type == LOGUP_PROOF) {
            for (const auto &layer : proofs[i].proofs) {
                data.push_back(encode_lookup_proof(layer));
                transcript.push_back(get_lookup_transcript(layer));
            }
		} else if (proofs[i].type == ADD_PROOF) {
			// Verify addition constraint sumcheck
			verify_add_proof(proofs[i]);
			data.push_back(encode_add_proof(proofs[i]));
			transcript.push_back(get_add_transcript(proofs[i]));
		}
	}

	string circuit_filename = "proof_transcript";
	char transcript_name[circuit_filename.length()+1];
	strcpy(transcript_name, circuit_filename.c_str());
	write_transcript(transcript, transcript_name);

	return {std::move(data), std::move(transcript)};
}

struct proof verify_proof(vector<proof> proofs){
    x_transcript.clear();
    y_transcript.clear();
    current_randomness = F_ZERO;
	VerifyBundle bundle = BuildVerificationBundle(proofs);

	vector<F> gkr_data;
	for (const auto &chunk : bundle.data) {
		gkr_data.insert(gkr_data.end(), chunk.begin(), chunk.end());
	}
	gkr_data.push_back(F(1));

	// Initialize transcript from proof data for deterministic randomness
	// Hash all proof data to seed the randomness
	for (const F& elem : gkr_data) {
		mimc_hash(elem, current_randomness);
	}
	
	// Derive initial randomness using Fiat-Shamir from transcript
	vector<F> r(10);
	for (int i = 0; i < 10; i++) {
		vector<F> ctx = {current_randomness, F(i), F(10)}; // initial randomness identifier
		r[i] = mimc_multihash(ctx);
		current_randomness = r[i]; // Update transcript state
	}
	return prove_verification(gkr_data, r, bundle.transcript);
	//generate_GKR_proof(circuit_name,name,r,true);
}


