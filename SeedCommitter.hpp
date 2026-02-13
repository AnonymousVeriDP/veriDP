// SeedCommitter.hpp
#ifndef SEEDCOMMITTER_HPP
#define SEEDCOMMITTER_HPP

#include <openssl/evp.h>
#include <openssl/rand.h>
#include <vector>
#include <string>

class SeedCommitter {
public:
    SeedCommitter() = default;
    ~SeedCommitter() = default;

    // Generate unbiased seed for given context
    std::vector<unsigned char> generate(const std::string& trainingID, int t, int batchID);

    // Generate seed and return as hex string
    std::string generateHex(const std::string& trainingID, int t, int batchID);

private:
    // Hash function using EVP SHA-256
    std::vector<unsigned char> hashSHA256(const std::vector<unsigned char>& input);

    // Commitment: hash(k || context)
    std::vector<unsigned char> commit(const std::vector<unsigned char>& k, const std::string& context);

    // CSPRNG using OpenSSL
    std::vector<unsigned char> generateRandom(size_t length);

    // XOR two vectors
    std::vector<unsigned char> xorVec(const std::vector<unsigned char>& a,
                                      const std::vector<unsigned char>& b);

    // Convert bytes to hex string
    std::string toHex(const std::vector<unsigned char>& v);

    // Securely zero memory
    void secureZero(std::vector<unsigned char>& v);
};

#endif // SEEDCOMMITTER_HPP
