#include "expanders.h"
#include "quantization.h"
#include "linear_code_encode.h"
#include <math.h>
#include "poly_commit.h"
#include "mimc.h"
#ifdef DEBUG_LOGUP
#include <iostream>
#endif

// Define constants before using them
#define MIMC 1
#define SHA 2

// Global variables - define them here if not defined elsewhere
// These are used by poly_commit functions
int PC_scheme = 1;  // Default polynomial commitment scheme
int Commitment_hash = SHA;  // Default to SHA (2)

// Global variables for tracking commitment times by source type
double g_commit_general = 0.0;
double g_commit_aggr = 0.0;
double g_commit_logup = 0.0;

// Function to accumulate commitment time by source type
void commit_accumulate(CommitSrc src, double time) {
    switch (src) {
        case CommitSrc::General:
            g_commit_general += time;
            break;
        case CommitSrc::Aggregation:
            g_commit_aggr += time;
            break;
        case CommitSrc::Logup:
            g_commit_logup += time;
            break;
    }
}


// Global variables - define defaults if not defined elsewhere
int arity = 2;  // Default arity for aggregation (defined here since main.cpp is excluded)
extern bool __encode_initialized;  // Defined in linear_code_encode.cpp
double aggregation_time = 0.0;  // Aggregation timing (defined here since main.cpp is excluded)
vector<F> w;

void field_to_int32(vector<unsigned int> &buff, F num1, F num2){
    unsigned long long int _num = num1.toint128();
    buff[0] = _num%(1ULL<<32);
    buff[1] = _num>>32;
    _num = num2.toint128();
    buff[2] = _num%(1ULL<<32);
    buff[3] = _num>>32;
    for(int i = 4; i < 8; i++)buff[i] = 0;
}


void query(int col, int row,  vector<vector<F>> &matrix, commitment &mt, aggregation_witness &_data){

    vector<F> aux(4);
    __hhash_digest res;
    __hhash_digest data[2];
    F data_f[2];
    vector<__hhash_digest> _proof;
    //vector<vector<unsigned int>> proof;
    vector<vector<unsigned int>> _out;
    vector<unsigned int> integers(8);
    vector<F> idx;
    int counter = 1;

    bool has_sha  = !mt.hashes_sha.empty();
    bool has_mimc = !mt.hashes_f.empty();

    if(Commitment_hash == SHA || (Commitment_hash == MIMC && has_sha)){
        int leaf_groups = (int)mt.hashes_sha[0].size();
        // 防御：col 必须 < leaf_groups
        if (col < 0 || col >= leaf_groups) {
            fprintf(stderr, "[query] col=%d out of range, leaf_groups=%d\n", col, leaf_groups);
            abort();
        }
        
        int offset = 4 * col;
        memset(&res, 0, sizeof(__hhash_digest));
        idx.push_back(F(1));

        if ((int)matrix[row].size() < offset + 4) {
            fprintf(stderr, "[query] matrix row size=%zu < offset+4=%d\n",
                    matrix[row].size(), offset + 4);
            abort();
        }

        field_to_int32(integers, matrix[row][offset],matrix[row][offset+1]);
        _data.merkle_proof.push_back(integers);
        field_to_int32(integers, matrix[row][offset+2],matrix[row][offset+3]);
        _data.merkle_proof.push_back(integers);

        res = merkle_tree::hash_double_field_element_merkle_damgard(matrix[row][offset],matrix[row][offset+1],matrix[row][offset+2], matrix[row][offset+3],res);
        merkle_tree::get_int32(integers, res);
        _out.push_back(integers);
        if(mt.hashes_sha.size() != 1){
             _data.merkle_proof.push_back(integers);
        }

        int pos = (row * leaf_groups + col) / 2;
        
        while (pos != 1){
            if (counter >= (int)mt.hashes_sha.size()) break;
            int layer_size = (int)mt.hashes_sha[counter].size();
            if (pos < 0 || pos >= layer_size) {
                fprintf(stderr, "[query] SHA layer=%d pos=%d out of range size=%d\n",
                        counter, pos, layer_size);
                abort();
            }
            idx.push_back(F(pos&1));
            data[pos&1] = res;
            data[(pos & 1) ^ 1] = mt.hashes_sha[counter][pos ^ 1];

            merkle_tree::get_int32(integers, data[(pos & 1) ^ 1]);
            _data.merkle_proof.push_back(integers);

            my_hhash(data, &res);
            merkle_tree::get_int32(integers, res);
            _out.push_back(integers);

            if(counter + 1 < mt.hashes_sha.size()){
                _data.merkle_proof.push_back(integers);
            }
            pos /= 2;
            counter++;
        }

        
        counter = 0;
        vector<F> _proof_f;
        F res_f;

        while(pos != 1){
            if(!has_mimc){
                break;
            }
            int layer_size = (int)mt.hashes_f[counter].size();
            if (pos < 0 || pos >= layer_size) {
                fprintf(stderr, "[query] MiMC layer=%d pos=%d out of range size=%d\n",
                        counter, pos, layer_size);
                abort();
            }
            _data.idx_f.push_back(F(pos&1));
            data_f[pos&1] = mt.hashes_f[counter][pos&1];
            data_f[(pos & 1) ^ 1] = mt.hashes_f[counter][pos ^ 1];
           
            res_f = my_mimc_hash(data_f[pos&1], data_f[(pos & 1) ^ 1],aux);
            _data.merkle_proof_f.insert(_data.merkle_proof_f.end(),aux.begin(),aux.end());
            pos /= 2;
            counter++;
        }
        
        _data.idx.insert(_data.idx.end(),idx.begin(),idx.end());

        vector<F> hash(8);
        for(int i = 0; i < _out.size(); i++){
            for(int j = 0; j < 8; j++){
                hash[j] = F(_out[i][j]);
            }
            _data.output.push_back(hash);
        }
    }else{
        int leaf_groups = (int)mt.hashes_f[0].size();
        if (col < 0 || col >= leaf_groups) {
            fprintf(stderr, "[query] col=%d out of range (MiMC), leaf_groups=%d\n", col, leaf_groups);
            abort();
        }
        int offset = 2*col;
        _data.idx_f.push_back(F(1));
        
        //res = merkle_tree::hash_double_field_element_merkle_damgard(matrix[row][offset],matrix[row][offset+1],matrix[row][offset+2], matrix[row][offset+3],res);
        vector<F> _proof_f;
        F res_f;
        res_f = my_mimc_hash(matrix[row][offset],matrix[row][offset+1],aux); 
        _data.merkle_proof_f.insert(_data.merkle_proof_f.end(),aux.begin(),aux.end());
    
        int pos = (row * leaf_groups + col) / 2;
        counter = 0;

        while (pos != 1) {
            int layer_size = (int)mt.hashes_f[counter].size();
            if (pos < 0 || pos >= layer_size) {
                fprintf(stderr, "[query] MiMC layer=%d pos=%d out of range size=%d\n",
                        counter, pos, layer_size);
                abort();
            }
            _data.idx_f.push_back(F(pos&1));
            data_f[pos & 1]       = mt.hashes_f[counter][pos & 1];
            data_f[(pos & 1) ^ 1] = mt.hashes_f[counter][pos ^ 1];

            res_f = my_mimc_hash(data_f[pos & 1], data_f[(pos & 1) ^ 1], aux);
            _data.merkle_proof_f.insert(_data.merkle_proof_f.end(), aux.begin(), aux.end());

            pos /= 2;
            counter++;
        }
    }
}   

void precompute_omegas(int logn){
    w.clear();
    w.resize(1ULL<<logn);

    w[0] = F_ONE;

    // Get root of unity using field's static method (works for both 61-bit and 256-bit)
    F rou = F::getRootOfUnity(logn);
    w[1] = rou;
    
    for (u32 i = 2; i < (1ULL<<logn); ++i) w[i] = w[i - 1] * w[1];
}

void my_fft(vector<F> &arr, int logn, bool flag) {
//    cerr << "fft: " << endl;
//    for (auto x: arr) cerr << x << ' ';
//    cerr << endl;
    static std::vector<u32> rev;
    static std::vector<F> w;
    static int last_logn = -1;  // Track if w needs reinitialization

    u32 len = 1ULL << logn;
    assert(arr.size() == len);

    rev.resize(len);
    w.resize(len);

    rev[0] = 0;
    for (u32 i = 1; i < len; ++i)
        rev[i] = rev[i >> 1] >> 1 | (i & 1) << (logn - 1);

    // Initialize w if not done for this size (field-agnostic)
    if (last_logn != logn) {
        w[0] = F_ONE;
        F rou = F::getRootOfUnity(logn);
        w[1] = rou;
        for (u32 i = 2; i < len; ++i) w[i] = w[i - 1] * w[1];
        last_logn = logn;
    }
    
    if (flag) w[1] = w[1].inv();
    
    
    for (u32 i = 0; i < len; ++i)
        if (rev[i] < i) std::swap(arr[i], arr[rev[i]]);

    for (u32 i = 2; i <= len; i <<= 1)
        for (u32 j = 0; j < len; j += i)
            for (u32 k = 0; k < (i >> 1); ++k) {
                auto u = arr[j + k];
                auto v = arr[j + k + (i >> 1)] * w[len / i * k];
                arr[j + k] = u + v;
                arr[j + k + (i >> 1)] = u - v;
    
            }

    if (flag) {
        F ilen;
        ilen = F(len).inv();//ilen, len);
        for (u32 i = 0; i < len; ++i){
            arr[i] = arr[i] * ilen;
    
        }
    }
}



void commit_matrix(vector<vector<F>> &encoded_matrix,int bit_row,int bit_col, commitment &com, int level, CommitSrc src){
	vector<vector<__hhash_digest>> buff;//(poly.size());
	vector<vector<F>> _buff;
	vector<F> aux(4);
#ifdef DEBUG_LOGUP
	std::cerr << "[poly_commit] commit_matrix level=" << level
	          << " bit_row=" << bit_row
	          << " bit_col=" << bit_col
	          << " rows=" << encoded_matrix.size()
	          << " row0_size=" << (encoded_matrix.empty() ? 0 : encoded_matrix[0].size())
	          << std::endl;
#endif
	
	
	int poly_size = encoded_matrix.size()*encoded_matrix[0].size()/4;
#ifdef DEBUG_LOGUP
	std::cerr << "[poly_commit] poly_size=" << poly_size << std::endl;
#endif
    if(Commitment_hash == SHA){
        buff.resize((int)log2(poly_size)+1);
        buff[0].resize(poly_size);
        for(int i = 1; i < buff.size(); i++){
            buff[i].resize(poly_size/(1ULL<<i));
        }
    }else{
        if(level == -1){
            _buff.resize((int)log2(poly_size*2) + 1);
            _buff[0].resize(2*poly_size);
#ifdef DEBUG_LOGUP
			std::cerr << "[poly_commit] level -1 sizes: _buff[0]=" << _buff[0].size()
			          << " poly_size=" << poly_size << std::endl;
#endif
            //printf("%d\n",2*poly_size);
            for(int i = 1; i < (int)log2(poly_size) + 2; i++){
                _buff[i].resize((poly_size*2)/(1ULL<<(i)));
            }
        }else{
            buff.resize(level);    
            _buff.resize((int)log2(poly_size) - level + 1);
            buff[0].resize(poly_size);
            int i = 1;
            for( i = 1; i < level; i++){
                buff[i].resize(poly_size/(1ULL<<(i)));
            }
            _buff[0].resize(poly_size/(1ULL << (i)));
#ifdef DEBUG_LOGUP
			std::cerr << "[poly_commit] level>0 buffers: "
			          << "buff[0]=" << buff[0].size()
			          << " buff[1]=" << (buff.size() > 1 ? buff[1].size() : 0)
			          << " _buff[0]=" << (_buff.empty() ? 0 : _buff[0].size())
			          << std::endl;
#endif
            
            for(i = 1; i < _buff.size(); i++){
                _buff[i].resize(_buff[0].size()/(1ULL<<(i)));
            }
        }
    }
    clock_t t1 = clock();
        
    if(Commitment_hash == SHA){
        for(int i = 0; i < 1ULL<<(bit_row+1); i++){
            for(int j = 0; j < 1ULL<<(bit_col)/2; j++){
                memset(&buff[0][i*(1ULL<<(bit_col))/2 + j], 0, sizeof(__hhash_digest));
                buff[0][i*(1ULL<<(bit_col))/2 + j] = merkle_tree::hash_double_field_element_merkle_damgard(encoded_matrix[i][4*j], encoded_matrix[i][4*j+1],encoded_matrix[i][4*j+2],encoded_matrix[i][4*j+3], buff[0][i*(1ULL<<(bit_col))/2 + j]);
            }
        }
        merkle_tree::merkle_tree_prover::create_tree( buff[0].size() , buff, sizeof(__hhash_digest), true);
    }else{
        if(level == -1){
            for(int i = 0; i < 1ULL<<(bit_row+1); i++){
                for(int j = 0; j < 1ULL<<(bit_col); j++){
                    _buff[0][i*(1ULL<<(bit_col)) + j] = my_mimc_hash(encoded_matrix[i][2*j], encoded_matrix[i][2*j+1],aux);
                }
            }    
        }else{
			for(int i = 0; i < 1ULL<<(bit_row+1); i++){
				for(int j = 0; j < (1ULL<<(bit_col))/2; j++){
#ifdef DEBUG_LOGUP
					std::size_t needed = 4ULL * ((1ULL << bit_col) / 2ULL);
					if (encoded_matrix[i].size() < needed) {
						std::cerr << "[poly_commit] warning: encoded_matrix[" << i
						          << "] size=" << encoded_matrix[i].size()
						          << " expected >= " << needed << std::endl;
					}
#endif
					
					memset(&buff[0][i*(1ULL<<(bit_col))/2 + j], 0, sizeof(__hhash_digest));
					buff[0][i*(1ULL<<(bit_col))/2 + j] = merkle_tree::hash_double_field_element_merkle_damgard(encoded_matrix[i][4*j], encoded_matrix[i][4*j+1],encoded_matrix[i][4*j+2],encoded_matrix[i][4*j+3], buff[0][i*(1ULL<<(bit_col))/2 + j]);
				
				}
			}
            merkle_tree::merkle_tree_prover::create_tree_sha( buff[0].size() , buff,level, sizeof(__hhash_digest), true);
            // initialize _buff[0]
            vector<unsigned int> arr1(8),arr2(8);
            for(int i = 0; i < _buff[0].size(); i++){
                merkle_tree::get_int32(arr1, buff[level-1][2*i]);
                // We can also apply a linear combination of the elements
                merkle_tree::get_int32(arr2, buff[level-1][2*i+1]);
                _buff[0][i] = my_mimc_hash(F(arr1[0]),F(arr2[0]),aux);
            }
            
        }
        
       
        merkle_tree::merkle_tree_prover::create_tree_mimc(_buff[0].size() , _buff, level, sizeof(__hhash_digest), true);
    }
    
    commit_accumulate(src, (double)(clock() - t1) / (double)CLOCKS_PER_SEC);

    com.hashes_sha = buff;
    com.hashes_f = _buff;
    buff.clear();
    buff = vector<vector<__hhash_digest>>();
    _buff.clear();
    _buff = vector<vector<F>>();
}

void poly_commit(vector<F> poly, vector<vector<F>> &matrix, commitment &comm, int level, CommitSrc src){
    int bit_size = (int)log2(poly.size());
    int bit_row,bit_col;
    vector<F> arr;
    vector<vector<F>> temp_matrix,encoded_matrix;
#ifdef DEBUG_LOGUP
	std::cerr << "[poly_commit] start vec_size=" << poly.size()
	          << " level=" << level << std::endl;
#endif
    if(1ULL << bit_size != poly.size()){
        exit(-1);
        bit_size++;
    }
    if(PC_scheme == 1){
        bit_row = -1;
        bit_col = bit_size+1;
        __encode_initialized = false;
        
    }else{
        bit_row = -1;
        bit_col = bit_size;
        precompute_omegas(bit_col+1);
    }
    encoded_matrix.resize(1ULL<<(bit_row+1));
    
    
    clock_t t1 = clock();
    
    if(PC_scheme == 1){
        
        vector<F> arr = poly;
        
        vector<F> temp(2*arr.size());
        encode(arr.data(),temp.data(),arr.size());
        encoded_matrix[0].insert(encoded_matrix[0].begin(),temp.begin(),temp.end());
        encode(arr.data(),temp.data(),arr.size());
        encoded_matrix[0].insert(encoded_matrix[0].begin(),temp.begin(),temp.end());
        arr = vector<F>();
        temp = vector<F>();
        
    }else{
        encoded_matrix[0].resize(1ULL<<(bit_col+1),F_ZERO);
        for(int i = 0; i < (1ULL<<bit_col); i++){
            encoded_matrix[0][i] = poly[i];
        }
        my_fft(encoded_matrix[0],bit_col,false);
    }
    
    commit_accumulate(src, (double)(clock() - t1) / (double)CLOCKS_PER_SEC);
   
   commit_matrix( encoded_matrix,bit_row,bit_col,comm, level, src);
   matrix = encoded_matrix;
   encoded_matrix.clear();
   encoded_matrix = vector<vector<F>>();
}


aggregation_witness aggregate(vector<vector<vector<F>>> &encoded_matrixes, vector<commitment> &mt_arr, vector<F> a, int level){
    vector<vector<F>> aggr_poly(encoded_matrixes[0].size());
    aggregation_witness data;
    
    int QUERIES;
    if(PC_scheme == 1){
        QUERIES = 450;
    }else{
        QUERIES = 25;
    }
    for(int i =0 ; i < aggr_poly.size(); i++){
        aggr_poly[i].resize(encoded_matrixes[0][0].size(),F_ZERO);
    }

    clock_t begin = clock();
    for(int i = 0; i <  encoded_matrixes.size(); i++){
        for(int j = 0; j < aggr_poly.size(); j++){
            for(int k = 0; k < aggr_poly[j].size(); k++){
                aggr_poly[j][k] += a[i]*encoded_matrixes[i][j][k];
            }
        }
    }
    
    int leaf_groups0 = !mt_arr[0].hashes_sha.empty() ? (int)mt_arr[0].hashes_sha[0].size()
                   : (!mt_arr[0].hashes_f.empty() ? (int)mt_arr[0].hashes_f[0].size() : 0);
    if (leaf_groups0 <= 0) {
        fprintf(stderr, "[aggregate] invalid leaf_groups0\n");
        abort();
    }
    int row_size = (int)aggr_poly.size();
    int col_size = leaf_groups0;   

    int sha_levels  = (int)mt_arr[0].hashes_sha.size();
    int mimc_levels = (int)mt_arr[0].hashes_f.size();
    data.proof_size = std::max(0, sha_levels - 1) + std::max(0, mimc_levels - 1);
   
    
    if(Commitment_hash == MIMC && level == -1){
        col_size = aggr_poly[0].size();
    }
    vector<vector<int>> pos;
    vector<int> coord(2);
    for(int i = 0; i < QUERIES; i++){
        coord[0] = (unsigned int)rand()%col_size;
        coord[1] = (unsigned int)rand()%row_size;
        pos.push_back(coord);
    }
    data.proof_size = (int)log2(row_size*col_size);

    __hhash_digest *mt;

    vector<vector<vector<unsigned int>>> proofs;

    commitment dst;
    aggregation_time += (double)(clock()-begin)/(double)CLOCKS_PER_SEC;

    
    commit_matrix( aggr_poly,(int)log2(aggr_poly.size())-1,(int)log2(aggr_poly[0].size())-1,dst,  level, CommitSrc::Aggregation);

   
    vector<F> root(8);
    
    begin = clock();
    for(int j = 0; j < encoded_matrixes.size(); j++){
        aggregation_witness _data;
        for(int i = 0; i < QUERIES; i++){
            query(pos[i][0],pos[i][1],encoded_matrixes[j],mt_arr[j],_data);
            if(i == 0){
                if(mt_arr[j].hashes_f.size()){
                    _data.root.push_back(mt_arr[j].hashes_f[mt_arr[j].hashes_f.size()-1][0]);
                }else{
                    vector<unsigned int> rt(8);
                    merkle_tree::get_int32(rt,mt_arr[j].hashes_sha[mt_arr[j].hashes_sha.size()-1][0]);
                    for(int i = 0; i < 8; i++){
                        root[i] = F(rt[i]);
                    }
                    _data.root_sha.push_back(root);
                }
            }
        } 
            if(_data.root.size() != 0){
                data.root.insert(data.root.end(),_data.root.begin(),_data.root.end());
            }
            if(_data.root_sha.size() != 0){
                data.root_sha.insert(data.root_sha.end(),_data.root_sha.begin(),_data.root_sha.end());
            }
            if(_data.merkle_proof.size() != 0){
                data.merkle_proof.insert(data.merkle_proof.end(),_data.merkle_proof.begin(),_data.merkle_proof.end());
            }
            if(_data.merkle_proof_f.size() != 0){
                data.merkle_proof_f.insert(data.merkle_proof_f.end(),_data.merkle_proof_f.begin(),_data.merkle_proof_f.end());
            }
            if(_data.idx_f.size() != 0){
                data.idx_f.insert(data.idx_f.end(),_data.idx_f.begin(),_data.idx_f.end());
            }
            if(_data.output.size()!= 0){
                data.output.insert(data.output.end(),_data.output.begin(),_data.output.end());
            }
            if(_data.idx.size()!= 0){
                data.idx.insert(data.idx.end(),_data.idx.begin(),_data.idx.end());
            }
        
    }
    
    printf("OK1\n");
        
    // /// Comment out FOR A REAL IMPLEMENTATION
    for(int i = 0; i < QUERIES; i++){
        query(pos[i][0],pos[i][1],aggr_poly,dst,data);
        if(i == 0){
            if(dst.hashes_f.size()){
                data.root.push_back(dst.hashes_f[dst.hashes_f.size()-1][0]);
            }else{
                vector<unsigned int> rt(8);
                merkle_tree::get_int32(rt,dst.hashes_sha[dst.hashes_sha.size()-1][0]);
                for(int i = 0; i < 8; i++){
                    root[i] = F(rt[i]);
                }
               data.root_sha.push_back(root);
                
            }
        }
    }
    aggregation_time += (double)(clock()-begin)/(double)CLOCKS_PER_SEC;

    
    printf("aggr %lf\n",aggregation_time);
    
    data.trees = encoded_matrixes.size() +1;
    data.a = a;
    data.a.push_back(-F(1));
    return data;
}
