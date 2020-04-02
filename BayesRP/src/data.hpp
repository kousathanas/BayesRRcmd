#ifndef data_hpp
#define data_hpp

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <set>
#include <bitset>
#include <iomanip>
#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <boost/format.hpp>
#include "gadgets.hpp"
#include "common.h"


using namespace std;
using namespace Eigen;

class SnpInfo {
public:
    const string ID;
    const string a1; // the referece allele
    const string a2; // the coded allele
    const int chrom;
    const float genPos;
    const int physPos;

    int index;
    int window;
    int windStart;  // for window surrounding the SNP
    int windSize;   // for window surrounding the SNP
    float af;       // allele frequency
    bool included;  // flag for inclusion in panel
    bool isQTL;     // for simulation

    SnpInfo(const int idx, const string &id, const string &allele1, const string &allele2,
            const int chr, const float gpos, const int ppos)
    : ID(id), index(idx), a1(allele1), a2(allele2), chrom(chr), genPos(gpos), physPos(ppos) {
        window = 0;
        windStart = -1;
        windSize  = 0;
        af = -1;
        included = true;
        isQTL = false;
    };

    void resetWindow(void) {windStart = -1; windSize = 0;};
};


class IndInfo {
public:
    const string famID;
    const string indID;
    const string catID;    // catenated family and individual ID
    const string fatherID;
    const string motherID;
    const int famFileOrder; // original fam file order
    const int sex;  // 1: male, 2: female

    int index;
    bool kept;

    float phenotype;

    VectorXf covariates;  // covariates for fixed effects

    IndInfo(const int idx, const string &fid, const string &pid, const string &dad, const string &mom, const int sex)
    : famID(fid), indID(pid), catID(fid+":"+pid), fatherID(dad), motherID(mom), index(idx), famFileOrder(idx), sex(sex) {
        phenotype = -9;
        kept = true;
    }
};

using PpBedIndex = std::vector<IndexEntry>;

class Data {
public:
    Data();

    // Original data
    MatrixXd X;              // coefficient matrix for fixed effects
    MatrixXd Z;
    VectorXf D;              // 2pqn
    VectorXf y;              // phenotypes
    vector<int> G; 			 // groups
    VectorXd fail;			 // Failure indicator

    // Vectors for the sparse format solution
    std::vector<std::vector<int>> Zones; // Vector for each SNP: per SNP all the indices with 1 allele are written down
    std::vector<std::vector<int>> Ztwos; // Vector for each SNP: per SNP all the indices with 2 alleles are written down
    VectorXd means; //Mean for each SNP
    VectorXd sds;
    VectorXd mean_sd_ratio;


    MatrixXf XPX;            // X'X the MME lhs
    MatrixXf ZPX;            // Z'X the covariance matrix of SNPs and fixed effects
    VectorXf XPXdiag;        // X'X diagonal
    VectorXf ZPZdiag;        // Z'Z diagonal
    VectorXf XPy;            // X'y the MME rhs for fixed effects
    VectorXf ZPy;            // Z'y the MME rhs for snp effects

    VectorXf snp2pq;         // 2pq of SNPs
    VectorXf se;             // se from GWAS summary data
    VectorXf tss;            // total ss (ypy) for every SNP
    VectorXf b;              // beta from GWAS summary data
    VectorXf n;              // sample size for each SNP in GWAS

    vector<SnpInfo*> snpInfoVec;
    vector<IndInfo*> indInfoVec;

    map<string, SnpInfo*> snpInfoMap;
    map<string, IndInfo*> indInfoMap;
    
    unsigned numFixedEffects;

    unsigned numSnps = 0;
    unsigned numInds = 0;
    unsigned numNAs  = 0;

    vector<uint> NAsInds;
    vector<int>  blocksStarts;
    vector<int>  blocksEnds;
    uint         numBlocks = 0;


#ifdef USE_MPI

    unsigned numFixedEffects = 0;
    unsigned numSnps;
    unsigned numInds;

    unsigned numGroups = 1; // number of groups

    void preprocessCSVFile(const string &csvFile, const string &preprocessedCSVFile, const string &preprovessedCSVIndexFile, bool compress);
    void mapPreprocessBedFile(const string &preprocessedBedFile);
    void unmapPreprocessedBedFile();

    // MPI_File_read_at_all handling count argument larger than INT_MAX
    //
    template <typename T>
    void mpi_file_read_at_all(const size_t N, const MPI_Offset offset, const MPI_File fh, const MPI_Datatype MPI_DT, const int NREADS, T buffer, size_t &bytes) {

        int rank, dtsize;
        MPI_Status status;
        
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Type_size(MPI_DT, &dtsize);
        //printf("dtsize = %d vs %lu\n", dtsize, sizeof(buffer[0]));
        //fflush(stdout);
        assert(dtsize == sizeof(buffer[0]));
        
        if (NREADS == 0) return;
        
        int SPLIT_ON = check_int_overflow(size_t(ceil(double(N)/double(NREADS))), __LINE__, __FILE__);
        int count = SPLIT_ON;
        
        double totime = 0.0;
        bytes = 0;
        
        for (uint i=0; i<NREADS; ++i) {
            
            double t1 = -mysecond();

            const size_t iim = size_t(i) * size_t(SPLIT_ON);

            // Last iteration takes only the leftover
            if (i == NREADS-1) count = check_int_overflow(N - iim, __LINE__, __FILE__);
            
            //printf("read %d with count = %d x %lu = %lu Bytes to read\n", i, count, sizeof(buffer[0]), sizeof(buffer[0]) * size_t(count));
            //fflush(stdout);
            
            //check_mpi(MPI_File_read_at_all(fh, offset + iim * size_t(dtsize), &buffer[iim], count, MPI_DT, &status), __LINE__, __FILE__);
            check_mpi(MPI_File_read_at(fh, offset + iim * size_t(dtsize), &buffer[iim], count, MPI_DT, &status), __LINE__, __FILE__);
            t1 += mysecond();
            totime += t1;
            bytes += size_t(count) * size_t(dtsize);
            if (rank % 10 == 0) {
                printf("INFO   : rank %3d cumulated read time at %2d/%2d: %7.3f sec, avg time %7.3f, BW = %7.3f GB/s\n",
                       rank, i+1, NREADS, totime, totime / (i + 1), double(bytes) / totime / 1E9 );
                fflush(stdout);
            }

            //MPI_Barrier(MPI_COMM_WORLD);
        }

        //MPI_Barrier(MPI_COMM_WORLD);
    }


    // MPI_File_write_at_all handling count argument larger than INT_MAX
    //
    template <typename T>
    void mpi_file_write_at_all(const size_t N, MPI_Offset offset, MPI_File fh, MPI_Datatype MPI_DT, const int NWRITES, T buffer) 
    {
        int rank, dtsize;
        MPI_Status status;

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Type_size(MPI_DT, &dtsize);
        assert(dtsize == sizeof(buffer[0]));

        if (NWRITES == 0) return;

        int SPLIT_ON = check_int_overflow(size_t(ceil(double(N)/double(NWRITES))), __LINE__, __FILE__);
        int count = SPLIT_ON;

        for (uint i=0; i<NWRITES; ++i) {

            const size_t iim = size_t(i) * size_t(SPLIT_ON);

            // Last iteration takes only the leftover
            if (i == NWRITES-1) count = check_int_overflow(N - iim, __LINE__, __FILE__);

            check_mpi(MPI_File_write_at_all(fh, offset + iim * size_t(dtsize), &buffer[iim], count, MPI_DT, &status), __LINE__, __FILE__);
        }
    }

    void preprocess_data(const char* data, const uint NC, const uint NB, double* ppdata, const int rank);
#endif

    //EO to read definitions of blocks of markers to process
    void readMarkerBlocksFile(const string &markerBlocksFile);

    void readCSV(const string &filename, int cols);

    void readFamFile(const string &famFile);

    void readBimFile(const string &bimFile);
    void readBedFile_noMPI(const string &bedFile);
    void readBedFile_noMPI_unstandardised(const string &bedFile);

    void readPhenotypeFile(const string &phenFile);
    void readPhenotypeFile(const string &phenFile, const int numberIndividuals, VectorXd& dest);

    void readPhenotypeFileAndSetNanMask(const string &phenFile, const int numberIndividuals, VectorXd& phen, VectorXi& mask, uint& nas);

    void readPhenotypeFiles(const vector<string> &phenFile, const int numberIndividuals, MatrixXd& dest);

    void readPhenCovFiles(const string &phenFile, const string covFile, const int numberIndividuals, VectorXd& dest, const int rank);


    // Functions to read for bayesW
    //
    void readPhenFailCovFiles(const string &phenFile, const string covFile, const string &failFile, const int numberIndividuals, VectorXd& dest, VectorXd& dfail, const int rank);

    void readPhenFailFiles(const string &phenFile, const string &failFile, const int numberIndividuals, VectorXd& dest, VectorXd& dfail, const int rank);

    template<typename M>
    M readCSVFile(const string &covariateFile);

    // marion : annotation variables
    unsigned numGroups = 1;	// number of annotations
    void readGroupFile(const string &groupFile);
    void readCSVFile(const string &csvFile);
    void readCSVPhenFile( const string &csvFile);
    void readmSFile(const string& mSfile);
    //BayesW variables
    void readFailureFile(const string &failureFile);
};

#endif /* data_hpp */
