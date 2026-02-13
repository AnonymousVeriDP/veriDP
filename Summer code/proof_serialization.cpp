#include "proof_serialization.h"
#include "pol_verifier.h"
#include "polynomial.h"
#include <sstream>
#include <iomanip>
#include <cassert>
#include <iostream>
#include <algorithm>

using namespace std;

// Serialize a proof to field elements + transcript
SerializedProof SerializeProof(const struct proof &P) {
  SerializedProof result;
  result.proof_type = P.type;
  
  // Use existing encoding functions based on proof type
  if (P.type == GKR_PROOF) {
    result.data = encode_gkr_proof(P);
    result.transcript = get_gkr_transcript(P);
  } else if (P.type == MATMUL_PROOF) {
    result.data = encode_m2m_proof(P);
    result.transcript = get_m2m_transcript(P);
  } else if (P.type == RANGE_PROOF || P.type == RANGE_PROOF_OPT || P.type == RANGE_PROOF_LOOKUP) {
    result.data = encode_bit_decomposition(P);
    result.transcript = get_range_proof_transcript(P);
  } else if (P.type == LOGUP_PROOF) {
    result.transcript.push_back(LOGUP_PROOF);

    result.transcript.push_back(static_cast<int>(P.output.size()));
    result.transcript.push_back(static_cast<int>(P.eval_point.size()));
    result.transcript.push_back(static_cast<int>(P.final_r.size()));

    result.transcript.push_back(static_cast<int>(P.sc_challenges.size()));
    for (const auto &vec : P.sc_challenges) {
      result.transcript.push_back(static_cast<int>(vec.size()));
    }

    result.transcript.push_back(static_cast<int>(P.sig.size()));
    for (const auto &vec : P.sig) {
      result.transcript.push_back(static_cast<int>(vec.size()));
    }

    result.transcript.push_back(static_cast<int>(P.final_claims_v.size()));
    for (const auto &vec : P.final_claims_v) {
      result.transcript.push_back(static_cast<int>(vec.size()));
    }

    result.data.insert(result.data.end(), P.output.begin(), P.output.end());
    result.data.insert(result.data.end(), P.eval_point.begin(), P.eval_point.end());
    result.data.insert(result.data.end(), P.final_r.begin(), P.final_r.end());

    for (const auto &vec : P.sc_challenges) {
      result.data.insert(result.data.end(), vec.begin(), vec.end());
    }
    for (const auto &vec : P.sig) {
      result.data.insert(result.data.end(), vec.begin(), vec.end());
    }
    for (const auto &vec : P.final_claims_v) {
      result.data.insert(result.data.end(), vec.begin(), vec.end());
    }
    result.data.push_back(P.final_eval);
  } else {
    // For other proof types, create a minimal encoding
    // This is a placeholder - in a full implementation, we'd handle all types
    result.data = vector<F>();
    result.transcript = vector<int>(1, P.type);
    cerr << "[SerializeProof] Warning: Unsupported proof type " << P.type << endl;
  }
  
  return result;
}

// Deserialize a proof from field elements + transcript
// Note: This reconstructs the proof structure, but some fields may be incomplete
// as the encoding functions only serialize essential verification data
struct proof DeserializeProof(const SerializedProof &serialized) {
  struct proof P;
  P.type = serialized.proof_type;
  
  // Initialize empty vectors
  P.q_poly = vector<quadratic_poly>();
  P.c_poly = vector<cubic_poly>();
  P.randomness = vector<vector<F>>();
  P.vr = vector<F>();
  P.gr = vector<F>();
  P.sig = vector<vector<F>>();
  P.final_claims_v = vector<vector<F>>();
  P.output = vector<F>();
  
  if (serialized.proof_type == MATMUL_PROOF) {
    // Deserialize MATMUL_PROOF
    // Format: [q_poly coefficients (3 per poly), randomness[0], vr[0], vr[1], sum]
    const vector<F> &data = serialized.data;
    const vector<int> &transcript = serialized.transcript;
    
    if (transcript.size() >= 2 && data.size() > 0) {
      int num_q_poly = (data.size() - transcript[1] - 3) / 3; // Subtract randomness size and 3 for vr[0], vr[1], sum
      if (num_q_poly > 0) {
        int idx = 0;
        // Deserialize q_poly
        for (int i = 0; i < num_q_poly; ++i) {
          if (idx + 2 < (int)data.size()) {
            P.q_poly.push_back(quadratic_poly(data[idx], data[idx+1], data[idx+2]));
            idx += 3;
          }
        }
        // Deserialize randomness[0]
        if (transcript[1] > 0 && idx + transcript[1] <= (int)data.size()) {
          vector<F> r0;
          for (int i = 0; i < transcript[1]; ++i) {
            r0.push_back(data[idx++]);
          }
          P.randomness.push_back(r0);
        }
        // Deserialize vr[0], vr[1]
        if (idx + 2 <= (int)data.size()) {
          P.vr.push_back(data[idx++]);
          P.vr.push_back(data[idx++]);
        }
      }
    }
  } else if (serialized.proof_type == RANGE_PROOF || 
             serialized.proof_type == RANGE_PROOF_OPT || 
             serialized.proof_type == RANGE_PROOF_LOOKUP) {
    // Deserialize RANGE_PROOF
    // Format: [q_poly coefficients (3 per poly), c_poly coefficients (4 per poly),
    //          randomness[2], randomness[3], vr[0], vr[1], vr[2], 1-vr[2], vr[3], sum]
    const vector<F> &data = serialized.data;
    const vector<int> &transcript = serialized.transcript;
    
    if (transcript.size() >= 3 && data.size() > 0) {
      int idx = 0;
      // Estimate number of q_poly and c_poly from data size
      // We need to parse: q_poly (3 each), c_poly (4 each), randomness[2] (transcript[1]),
      // randomness[3] (transcript[2]), then 6 more values (vr[0], vr[1], vr[2], 1-vr[2], vr[3], sum)
      int remaining = data.size() - transcript[1] - transcript[2] - 6;
      int num_q_poly = 0, num_c_poly = 0;
      
      // Try to estimate: assume roughly equal numbers
      if (remaining >= 7) {
        num_q_poly = remaining / 7; // Rough estimate
        num_c_poly = (remaining - num_q_poly * 3) / 4;
      }
      
      // Deserialize q_poly
      for (int i = 0; i < num_q_poly && idx + 2 < (int)data.size(); ++i) {
        P.q_poly.push_back(quadratic_poly(data[idx], data[idx+1], data[idx+2]));
        idx += 3;
      }
      
      // Deserialize c_poly
      for (int i = 0; i < num_c_poly && idx + 3 < (int)data.size(); ++i) {
        P.c_poly.push_back(cubic_poly(data[idx], data[idx+1], data[idx+2], data[idx+3]));
        idx += 4;
      }
      
      // Deserialize randomness[2]
      if (transcript[1] > 0 && idx + transcript[1] <= (int)data.size()) {
        vector<F> r2;
        for (int i = 0; i < transcript[1]; ++i) {
          r2.push_back(data[idx++]);
        }
        P.randomness.push_back(vector<F>()); // [0]
        P.randomness.push_back(vector<F>()); // [1]
        P.randomness.push_back(r2); // [2]
      }
      
      // Deserialize randomness[3]
      if (transcript[2] > 0 && idx + transcript[2] <= (int)data.size()) {
        vector<F> r3;
        for (int i = 0; i < transcript[2]; ++i) {
          r3.push_back(data[idx++]);
        }
        if (P.randomness.size() < 3) {
          P.randomness.resize(3);
        }
        P.randomness.push_back(r3); // [3]
      }
      
      // Deserialize vr values
      if (idx + 5 <= (int)data.size()) {
        P.vr.push_back(data[idx++]);
        P.vr.push_back(data[idx++]);
        P.vr.push_back(data[idx++]);
        // Skip 1-vr[2]
        idx++;
        P.vr.push_back(data[idx++]);
      }
    }
  } else if (serialized.proof_type == GKR_PROOF) {
    const vector<F> &data = serialized.data;
    const vector<int> &transcript = serialized.transcript;

    if (transcript.size() >= 2) {
      int num_layers = transcript[1];
      int randomness_length_total = 0;
      for (int i = 0; i < num_layers && 2 + i < (int)transcript.size(); ++i) {
        randomness_length_total += transcript[2 + i];
      }
      int sig_start = 2 + num_layers;
      int num_sig = (int)transcript.size() - sig_start;
      int sig_entries_total = 0;
      for (int i = 0; i < num_sig; ++i) {
        sig_entries_total += transcript[sig_start + i];
      }

      int num_layers_for_vr = std::max(0, num_layers);
      int remaining = (int)data.size() - randomness_length_total - 2 * sig_entries_total - 2 * num_layers_for_vr - 1;
      int num_q_poly = remaining / 3;

      int idx = 0;
      for (int i = 0; i < num_q_poly && idx + 2 < (int)data.size(); ++i) {
        P.q_poly.push_back(quadratic_poly(data[idx], data[idx + 1], data[idx + 2]));
        idx += 3;
      }

      P.randomness.push_back(vector<F>());
      for (int layer = 0; layer < num_layers && idx < (int)data.size(); ++layer) {
        int len = transcript[2 + layer];
        vector<F> r;
        for (int j = 0; j < len && idx < (int)data.size(); ++j) {
          r.push_back(data[idx++]);
        }
        P.randomness.push_back(r);
      }

      for (int s = 0; s < num_sig && idx < (int)data.size(); ++s) {
        int len = transcript[sig_start + s];
        vector<F> sig_vec;
        vector<F> claims_vec;
        for (int j = 0; j < len && idx + 1 < (int)data.size(); ++j) {
          sig_vec.push_back(data[idx++]);
          claims_vec.push_back(data[idx++]);
        }
        P.sig.push_back(sig_vec);
        P.final_claims_v.push_back(claims_vec);
      }

      for (int layer = 0; layer < num_layers && idx + 1 < (int)data.size(); ++layer) {
        P.vr.push_back(data[idx++]);
        P.gr.push_back(data[idx++]);
      }

      if (idx < (int)data.size()) {
        F final_sum = data[idx];
        P.final_eval = final_sum;
      }
    }
  } else if (serialized.proof_type == LOGUP_PROOF) {
    const vector<F> &data = serialized.data;
    const vector<int> &transcript = serialized.transcript;
    if (!transcript.empty() && data.size() > 0) {
      int t_idx = 1;
      int idx = 0;

      auto consume = [&](int len, vector<F> &target) {
        target.clear();
        for (int i = 0; i < len && idx < (int)data.size(); ++i) {
          target.push_back(data[idx++]);
        }
      };

      int output_len = (t_idx < (int)transcript.size()) ? transcript[t_idx++] : 0;
      P.output.reserve(output_len);
      consume(output_len, P.output);

      int eval_len = (t_idx < (int)transcript.size()) ? transcript[t_idx++] : 0;
      P.eval_point.reserve(eval_len);
      consume(eval_len, P.eval_point);

      int final_r_len = (t_idx < (int)transcript.size()) ? transcript[t_idx++] : 0;
      P.final_r.reserve(final_r_len);
      consume(final_r_len, P.final_r);

      int sc_count = (t_idx < (int)transcript.size()) ? transcript[t_idx++] : 0;
      P.sc_challenges.clear();
      P.sc_challenges.reserve(sc_count);
      for (int sc = 0; sc < sc_count && t_idx < (int)transcript.size(); ++sc) {
        int len = transcript[t_idx++];
        vector<F> vec;
        vec.reserve(len);
        consume(len, vec);
        P.sc_challenges.push_back(std::move(vec));
      }

      int sig_count = (t_idx < (int)transcript.size()) ? transcript[t_idx++] : 0;
      P.sig.clear();
      P.sig.reserve(sig_count);
      for (int s = 0; s < sig_count && t_idx < (int)transcript.size(); ++s) {
        int len = transcript[t_idx++];
        vector<F> vec;
        vec.reserve(len);
        consume(len, vec);
        P.sig.push_back(std::move(vec));
      }

      int claims_count = (t_idx < (int)transcript.size()) ? transcript[t_idx++] : 0;
      P.final_claims_v.clear();
      P.final_claims_v.reserve(claims_count);
      for (int c = 0; c < claims_count && t_idx < (int)transcript.size(); ++c) {
        int len = transcript[t_idx++];
        vector<F> vec;
        vec.reserve(len);
        consume(len, vec);
        P.final_claims_v.push_back(std::move(vec));
      }

      if (idx < (int)data.size()) {
        P.final_eval = data[idx++];
      }

      // Ensure other members are initialised
      if (P.eval_point.empty()) {
        P.eval_point.push_back(F_ZERO);
      }
      if (P.final_r.empty()) {
        P.final_r.push_back(F_ZERO);
      }
    }
  } else {
    cerr << "[DeserializeProof] Warning: Unsupported proof type " << serialized.proof_type << endl;
  }
  
  return P;
}

// Helper: Convert field element to string representation
// Field-agnostic serialization
std::string FieldElementToString(const F &elem) {
  std::ostringstream oss;
#ifdef USE_FIELD_256
  // For 256-bit field: serialize all 4 limbs
  oss << elem.getLimb(0) << "," << elem.getLimb(1) << "," << elem.getLimb(2) << "," << elem.getLimb(3);
#else
  // For 61-bit field: serialize real and img
  oss << elem.real << "," << elem.img;
#endif
  return oss.str();
}

// Helper: Convert string to field element
F StringToFieldElement(const std::string &str) {
  std::istringstream iss(str);
  
#ifdef USE_FIELD_256
  // Parse "limb0,limb1,limb2,limb3" format for 256-bit field
  std::string l0_str, l1_str, l2_str, l3_str;
  if (!std::getline(iss, l0_str, ',')) return F(0);
  if (!std::getline(iss, l1_str, ',')) return F(0);
  if (!std::getline(iss, l2_str, ',')) return F(0);
  if (!std::getline(iss, l3_str)) return F(0);
  return F(std::stoull(l0_str), std::stoull(l1_str), std::stoull(l2_str), std::stoull(l3_str));
#else
  // Parse "real,img" format for 61-bit field
  std::string real_str, img_str;
  if (!std::getline(iss, real_str, ',')) return F(0);
  if (!std::getline(iss, img_str)) return F(0);
  unsigned long long real = std::stoull(real_str);
  unsigned long long img = std::stoull(img_str);
  return F(real, img);
#endif
}

// Serialize a proof to a string (for storage/transmission)
// Uses a simple format: proof_type|transcript_size|transcript_data|data_size|data
std::string SerializeProofToString(const struct proof &P) {
  try {
    SerializedProof serialized = SerializeProof(P);
    
    std::ostringstream oss;
    oss << serialized.proof_type << "|";
    oss << serialized.transcript.size() << "|";
    for (size_t i = 0; i < serialized.transcript.size(); ++i) {
      if (i > 0) oss << ",";
      oss << serialized.transcript[i];
    }
    oss << "|";
    oss << serialized.data.size() << "|";
    for (size_t i = 0; i < serialized.data.size(); ++i) {
      if (i > 0) oss << ",";
      oss << FieldElementToString(serialized.data[i]);
    }
    
    return oss.str();
  } catch (const std::exception &e) {
    std::cerr << "[ERROR SerializeProofToString] Exception: " << e.what() << std::endl;
    throw;
  } catch (...) {
    std::cerr << "[ERROR SerializeProofToString] Unknown exception" << std::endl;
    throw;
  }
}

// Deserialize a proof from a string
struct proof DeserializeProofFromString(const std::string &serialized_str) {
  SerializedProof serialized;
  
  std::istringstream iss(serialized_str);
  std::string token;
  
  // Parse proof_type
  if (!std::getline(iss, token, '|')) {
    cerr << "[DeserializeProofFromString] Error: Invalid format" << endl;
    proof empty_proof;
    return empty_proof;
  }
  serialized.proof_type = std::stoi(token);
  
  // Parse transcript_size
  if (!std::getline(iss, token, '|')) {
    cerr << "[DeserializeProofFromString] Error: Invalid format" << endl;
    proof empty_proof;
    return empty_proof;
  }
  int transcript_size = std::stoi(token);
  
  // Parse transcript data
  if (!std::getline(iss, token, '|')) {
    cerr << "[DeserializeProofFromString] Error: Invalid format" << endl;
    proof empty_proof;
    return empty_proof;
  }
  std::istringstream transcript_iss(token);
  for (int i = 0; i < transcript_size; ++i) {
    std::string item;
    if (!std::getline(transcript_iss, item, ',')) break;
    serialized.transcript.push_back(std::stoi(item));
  }
  
  // Parse data_size
  if (!std::getline(iss, token, '|')) {
    cerr << "[DeserializeProofFromString] Error: Invalid format" << endl;
    proof empty_proof;
    return empty_proof;
  }
  int data_size = std::stoi(token);
  
  // Parse data
  if (!std::getline(iss, token)) {
    cerr << "[DeserializeProofFromString] Error: Invalid format" << endl;
    proof empty_proof;
    return empty_proof;
  }
  std::istringstream data_iss(token);
  for (int i = 0; i < data_size; ++i) {
    std::string item;
    if (!std::getline(data_iss, item, ',')) break;
    serialized.data.push_back(StringToFieldElement(item));
  }
  
  return DeserializeProof(serialized);
}

