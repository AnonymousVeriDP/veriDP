//
// Created by 69029 on 6/25/2021.
//

//#include <mcl/bls12_381.hpp>

// Field selection: Define USE_FIELD_256 to use 256-bit BLS12-381 scalar field
// Otherwise uses 61-bit Mersenne prime field
#ifdef USE_FIELD_256
    #include "fieldElement256.hpp"
    #define F   virgo256::fieldElement256
    #define F_ONE   virgo256::fieldElement256::one()
    #define F_ZERO  virgo256::fieldElement256::zero()
#else
    #include "fieldElement.hpp"
    #define F   virgo::fieldElement
    #define F_ONE   virgo::fieldElement::one()
    #define F_ZERO  virgo::fieldElement::zero()
#endif

//using namespace mcl::bn;
using namespace std;


//#define F Fr
//#define F_ONE (Fr(1))
//#define F_ZERO (Fr(0))



typedef unsigned __int128 u128;
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned char u8;

typedef __int128 i128;
typedef long long i64;
typedef int i32;
typedef char i8;
