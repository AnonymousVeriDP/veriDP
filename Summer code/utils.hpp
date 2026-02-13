//
// Created by 69029 on 6/23/2021.
//
#pragma once


#include "config_pc.hpp"

F chain_hash(F previous_r, vector<F> values);
vector<F> generate_randomness(int size, F seed);
void precompute_beta(vector<F> r,vector<F> &B);
void pad_vector(vector<F> &v);
float proof_size(const vector<struct proof>& P, size_t BYTES_PER_F = 32);
vector<vector<F>> generate_tables();
vector<F> lookup_prod(vector<vector<F>> tables, F num);
vector<vector<F>> get_poly(int size);
vector<F> get_predicates(int size);
F lookup(vector<vector<F>> tables, F num);
F evaluate_matrix(vector<vector<F>> M, vector<F> r1, vector<F> r2);
F evaluate_vector(vector<F> v,vector<F> r);
vector<F> prepare_matrix(vector<vector<F>> M, vector<F> r);
vector<F> convert2vector(vector<vector<F>> M);
void write_data(vector<F> data,char *filename);
vector<F> tensor2vector(vector<vector<vector<vector<F>>>> M);
vector<vector<vector<vector<F>>>> vector2tensor(vector<F> v,vector<vector<vector<vector<F>>>> M,int w);
vector<vector<F>> transpose(vector<vector<F>> M);
vector<vector<F>> convert2matrix(vector<F> arr, int d1, int d2);
vector<F> prepare_bit_vector(vector<F> num, int domain);
vector<vector<F>> rotate(vector<vector<F>> M);
void initBetaTable(vector<F> &beta_g, u8 gLength, const vector<F>::const_iterator &r, const F &init);
F getRootofUnity(int logn);
void
initBetaTable(vector<F> &beta_g, int gLength, const vector<F>::const_iterator &r_0,
              const vector<F>::const_iterator &r_1, const F &alpha, const F &beta);

void fft(vector<F> &arr, int logn, bool flag);
//F getRootOfUnit(int n);
void phiGInit(vector<F> &phi_g, const vector<F>::const_iterator &rx, const F &scale, int n, bool isIFFT);
template <class T>
void myResize(vector<T> &vec, u64 sz) {
    if (vec.size() < sz) vec.resize(sz);
}

F _inner_product(vector<F> v1, vector<F> v2);
vector<F> _mat_vec_mul(vector<vector<F>> M, vector<F> v);



void clear_arr(vector<F> &arr);
void clear_matrix(vector<vector<F>> &arr);
void clear_tensor1(vector<vector<vector<F>>> &arr);
void clear_tensor2(vector<vector<vector<vector<F>>>> &arr);
void clear_tensor3(vector<vector<vector<vector<vector<F>>>>> &arr);
vector<F> compute_lagrange_coeff(F omega, F r, int degree);