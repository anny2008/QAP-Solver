#ifndef QAP_CORE_PROBLEM_H
#define QAP_CORE_PROBLEM_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cctype>
#include <unordered_map>
#include <regex>

class Problem {
public:
    // number of locations (n) and machines/facilities (m)
    int n = 0;
    int m = 0;
    std::vector<std::vector<double>> F; // (m x m)
    std::vector<std::vector<double>> D; // (n x n)
    std::unordered_map<int, int> fixed_assignments; // location -> facility

    Problem() = default;

    Problem(int n_, int m_,
            const std::vector<std::vector<double>>& F_,
            const std::vector<std::vector<double>>& D_)
        : n(n_), m(m_), F(F_), D(D_) {}

    // --- Existing QAPLIB reader (kept for backward compatibility)
    static Problem fromQAPLIB(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filepath);
        }
        int n;
        file >> n;

        std::vector<std::vector<double>> D(n, std::vector<double>(n));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                file >> D[i][j];

        std::vector<std::vector<double>> F(n, std::vector<double>(n));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                file >> F[i][j];

        return Problem(n, n, F, D);
    }

    // --- Raw matrix reader: "L M" then L*L values for D then M*M values for F
    static Problem fromRawMatrixFile(const std::string& filepath, const std::string& fixed_file = "") {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filepath);
        }

        int L, M;
        if (!(file >> L >> M)) {
            throw std::runtime_error("Raw matrix file must start with two integers: <L> <M>");
        }

        std::vector<double> values;
        values.reserve(static_cast<size_t>(L) * L + static_cast<size_t>(M) * M);
        double t;
        while (file >> t) values.push_back(t);

        if (values.size() != static_cast<size_t>(L) * L + static_cast<size_t>(M) * M) {
            throw std::runtime_error("Incorrect number of matrix values in raw file");
        }

        std::vector<std::vector<double>> D(L, std::vector<double>(L));
        std::vector<std::vector<double>> F(M, std::vector<double>(M));

        size_t p = 0;
        for (int i = 0; i < L; ++i)
            for (int j = 0; j < L; ++j)
                D[i][j] = values[p++];

        for (int i = 0; i < M; ++i)
            for (int j = 0; j < M; ++j)
                F[i][j] = values[p++];

        Problem problem(L, M, F, D);
        
        // Load fixed assignments if file is provided
        // if (!fixed_file.empty()) {
        //     problem.load_fixed_assignments(fixed_file);
        // }
        
        return problem;
    }

    // --- Auto-detect: try to infer format from token counts
    static Problem fromAuto(const std::string& filepath, const std::string& fixed_file = "") {
        // Load whole file and split tokens
        std::ifstream in(filepath);
        if (!in.is_open()) {
            throw std::runtime_error("Cannot open file: " + filepath);
        }
        std::ostringstream oss;
        oss << in.rdbuf();
        std::string content = oss.str();

        // tokenize by whitespace
        std::vector<std::string> tok;
        tok.reserve(content.size() / 2);
        std::string cur;
        for (char c : content) {
            if (std::isspace(static_cast<unsigned char>(c))) {
                if (!cur.empty()) { tok.push_back(cur); cur.clear(); }
            } else cur.push_back(c);
        }
        if (!cur.empty()) tok.push_back(cur);

        if (tok.empty())
            throw std::runtime_error("Empty instance file: " + filepath);

        auto is_integer = [](const std::string& s)->bool{
            if (s.empty()) return false;
            size_t k=0; if (s[0]=='+'||s[0]=='-') k=1;
            for (; k<s.size(); ++k) if (!std::isdigit(static_cast<unsigned char>(s[k]))) return false;
            return true;
        };

        Problem problem;
        
        // Case A: QAPLIB => first token n, total tokens = 1 + 2*n*n
        if (is_integer(tok[0])) {
            long long nll = std::stoll(tok[0]);
            if (nll > 0) {
                size_t expect = 1ull + 2ull * static_cast<size_t>(nll) * static_cast<size_t>(nll);
                if (tok.size() == expect) {
                    // Parse as QAPLIB
                    std::istringstream fin(content);
                    problem = fromQAPLIBStream(fin);
                    
                    // Load fixed assignments if file is provided
                    if (!fixed_file.empty()) {
                        problem.load_fixed_assignments(fixed_file);
                    }
                    return problem;
                }
            }
        }

        // Case B: Raw => first two tokens L M, total tokens = 2 + L*L + M*M
        if (tok.size() >= 2 && is_integer(tok[0]) && is_integer(tok[1])) {
            long long L = std::stoll(tok[0]);
            long long M = std::stoll(tok[1]);
            if (L > 0 && M > 0) {
                size_t expect = 2ull
                              + static_cast<size_t>(L) * static_cast<size_t>(L)
                              + static_cast<size_t>(M) * static_cast<size_t>(M);
                if (tok.size() == expect) {
                    std::istringstream fin(content);
                    problem = fromRawMatrixStream(fin);
                    
                    // Load fixed assignments if file is provided
                    if (!fixed_file.empty()) {
                        problem.load_fixed_assignments(fixed_file);
                    }
                    return problem;
                }
            }
        }

        throw std::runtime_error("Could not infer instance format (neither QAPLIB nor Raw).");
    }

    // Load fixed assignments from a file
    // Format: lines with x_<location>_<facility>= ... = 1
    // Extracts location and facility IDs using regex matching
    void load_fixed_assignments(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open fixed assignments file: " + filepath);
        }

        fixed_assignments.clear();
        std::regex pattern(R"(x_(\d+)_(\d+))");
        std::string line;
        int count = 0;

        while (std::getline(file, line)) {
            // Check if line contains "=" and ends with "= 1"
            if (line.find('=') == std::string::npos) {
                continue;
            }

            // Trim trailing whitespace
            auto end = line.find_last_not_of(" \t\r\n");
            if (end == std::string::npos) {
                continue;
            }
            line = line.substr(0, end + 1);

            // Check if line ends with "= 1"
            if (line.length() < 3 || line.substr(line.length() - 3) != "= 1") {
                continue;
            }

            // Extract numbers using regex
            std::smatch matches;
            if (std::regex_search(line, matches, pattern)) {
                try {
                    int location = std::stoi(matches[1].str());
                    int facility = std::stoi(matches[2].str());

                    // Validate bounds
                    if (location < 0 || location >= n || facility < 0 || facility >= m) {
                        std::cerr << "Warning: Invalid assignment indices "
                                  << "(location=" << location << ", facility=" << facility << ")" << std::endl;
                        continue;
                    }

                    fixed_assignments[location] = facility;
                    ++count;
                } catch (const std::exception &e) {
                    std::cerr << "Warning: Failed to parse assignment line: " << line << std::endl;
                    continue;
                }
            }
        }

        std::cout << "Loaded " << count << " fixed assignments from " << filepath << std::endl;
    }

    // Evaluate a permutation: assignment[i] = u
    double evaluate(const std::vector<int>& assignment) const {
        if (static_cast<int>(assignment.size()) != n)
            throw std::runtime_error("Assignment size != n");
        double obj = 0.0;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                obj += F[assignment[i]][assignment[j]] * D[i][j];
        return obj;
    }

private:
    // Helpers that parse from already-open streams
    static Problem fromQAPLIBStream(std::istream& in) {
        int n; in >> n;
        std::vector<std::vector<double>> D(n, std::vector<double>(n));
        std::vector<std::vector<double>> F(n, std::vector<double>(n));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) in >> D[i][j];
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) in >> F[i][j];
        return Problem(n, n, F, D);
    }

    static Problem fromRawMatrixStream(std::istream& in) {
        int L, M; in >> L >> M;
        std::vector<std::vector<double>> D(L, std::vector<double>(L));
        std::vector<std::vector<double>> F(M, std::vector<double>(M));
        for (int i = 0; i < L; ++i)
            for (int j = 0; j < L; ++j) in >> D[i][j];
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < M; ++j) in >> F[i][j];
        return Problem(L, M, F, D);
    }
};

#endif // QAP_CORE_PROBLEM_H