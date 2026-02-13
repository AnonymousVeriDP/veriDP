//#include <mcl/bls12_381.hpp>
#include "mimc.h"
#include "GKR.h"
#include "config_pc.hpp"
#define ROUNDS 80
//#define F Fr


//using namespace mcl::bn;
using namespace std;
extern vector<F> x_transcript,y_transcript;
extern F current_randomness;
int partitions;


vector<F> Common;
F K_MIMC;

void init_hash(){
	partitions = 4;
	
	Common.resize(ROUNDS);
	for(int i = 0; i < ROUNDS; i++){
		Common[i] = random();
	}
	K_MIMC = random();
}


vector<F> mimc_hash_segments(F input, F k){
	F t,hash_val;
	vector<F> segments(partitions);
	hash_val = input;
	segments[0] = input + k;
	for(int j = 0; j < partitions-1; j++){
		for(int i = (80/partitions)*j; i < (80/partitions)*(j+1); i++){
			if(i == 0){
				t = input + k+  Common[i];
			}
			else{
				t = hash_val +  Common[i];
			}
			hash_val = t*t*t;
		}
		segments[j+1] = hash_val;
	}
	return segments;
}

F my_mimc_hash(F x1,F x2, vector<F> &aux){
	F t,hash_val;
	F k = F(1245);
	aux[0] = x1;
	aux[1] = k;
	
	for(int i = 0; i < ROUNDS; i++){
		if(i == 0){
			t = x1 + k;
		}
		else{
			t = hash_val + Common[i-1];
		}
		hash_val = t*t*t;
	}
	aux[2] = x2;
	aux[3] = hash_val;
	
	for(int i = 0; i < ROUNDS; i++){
		if(i == 0){
			t = x2 + hash_val;
		}
		else{
			t = hash_val + Common[i-1];
		}
		hash_val = t*t*t;
	}
	return hash_val;
}



F mimc_hash(F input,F k){
	F t,hash_val;
	k = current_randomness;
	x_transcript.push_back(input);
	x_transcript.push_back(k);

	for(int i = 0; i < ROUNDS; i++){
		if(i == 0){
			t = input + k;
		}
		else{
			t = hash_val + Common[i-1];
		}
		hash_val = t*t*t;
	}
	y_transcript.push_back(hash_val);
	current_randomness = hash_val;
	return hash_val;
}


vector<vector<F>> mimc_multihash3(vector<F> input){
	vector<vector<F>> hashes;
	F hash = F(0);
	//hashes.push_back(hash);
	for(int i = 0; i < input.size(); i++){
		vector<F> segments = mimc_hash_segments(input[i],hash);
		hash = (hash + input[i] + segments[segments.size()-1]);
		segments.push_back(hash);
		hashes.push_back(segments);
	}
	return hashes;
}


vector<F> mimc_multihash2(vector<F> input){
	vector<F> hashes;
	F hash = F(0);
	hashes.push_back(hash);
	for(int i = 0; i < input.size(); i++){
		hash = (hash + input[i] + mimc_hash(input[i],hash));
		hashes.push_back(hash);
	}
	return hashes;
}


F mimc_multihash(vector<F> input){
	F hash = F(0);
	for(int i = 0; i < input.size(); i++){
		hash = (hash + input[i] + mimc_hash(input[i],hash));
	}
	return hash;
}

vector<F> get_parameters(){
	vector<F> r;
	for(int i = 0; i < Common.size(); i++){
		r.push_back(Common[i]);
	}
	//r.push_back(K_MIMC);
	return r;
}