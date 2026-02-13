#pragma once
#include <string>
#include <ctime>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#ifdef _WIN32
#include <winsock2.h>
#include <windows.h>
#else
#include <unistd.h>
#include <sys/utsname.h>
#endif

struct BenchmarkRow {
    std::string timestamp;
    std::string hostname;
    int T, n, m, k;
    int PC_scheme;
    int Commitment_hash;
    int levels;
    int arity;
    long long param_count;
    double data_prove_time;
    double avg_update;
    double avg_backward;
    double avg_forward;
    double avg_logup;
    double avg_logup_prove;
    double avg_pogd;
    double agg_recursion_total;
    double agg_recursion_amort;
    double aggr_time_total;
    double aggr_time_avg;
    double aggr_prove_total;
    double aggr_prove_avg;
    double commit_time_avg;
    double recursion_verifier_time;
    double verifier_time;
    float pogd_proof_size;
    float final_proof_size;
    double peak_memory;
};

inline std::string now_iso8601() {
    auto now = std::time(nullptr);
    auto tm = *std::gmtime(&now);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

inline std::string get_hostname() {
    char hostname[256];
#ifdef _WIN32
    DWORD size = sizeof(hostname);
    if (GetComputerNameA(hostname, &size)) {
        return std::string(hostname);
    }
#else
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        return std::string(hostname);
    }
#endif
    return "unknown";
}

inline void append_benchmark_csv(const std::string& path, const BenchmarkRow& row) {
    std::ofstream out(path, std::ios::app);
    if (!out) {
        std::fprintf(stderr, "[bench] Failed to open %s for writing\n", path.c_str());
        return;
    }
    
    // Write header if file is empty
    out.seekp(0, std::ios::end);
    if (out.tellp() == 0) {
        out << "timestamp,hostname,T,n,m,k,PC_scheme,Commitment_hash,levels,arity,"
            << "param_count,data_prove_time,avg_update,avg_backward,avg_forward,"
            << "avg_logup,avg_logup_prove,avg_pogd,agg_recursion_total,agg_recursion_amort,"
            << "aggr_time_total,aggr_time_avg,aggr_prove_total,aggr_prove_avg,"
            << "commit_time_avg,recursion_verifier_time,verifier_time,"
            << "pogd_proof_size,final_proof_size,peak_memory\n";
    }
    
    out << row.timestamp << ","
        << row.hostname << ","
        << row.T << "," << row.n << "," << row.m << "," << row.k << ","
        << row.PC_scheme << "," << row.Commitment_hash << "," << row.levels << ","
        << row.arity << "," << row.param_count << ","
        << row.data_prove_time << "," << row.avg_update << "," << row.avg_backward << ","
        << row.avg_forward << "," << row.avg_logup << "," << row.avg_logup_prove << ","
        << row.avg_pogd << "," << row.agg_recursion_total << "," << row.agg_recursion_amort << ","
        << row.aggr_time_total << "," << row.aggr_time_avg << ","
        << row.aggr_prove_total << "," << row.aggr_prove_avg << ","
        << row.commit_time_avg << "," << row.recursion_verifier_time << ","
        << row.verifier_time << "," << row.pogd_proof_size << ","
        << row.final_proof_size << "," << row.peak_memory << "\n";
}
