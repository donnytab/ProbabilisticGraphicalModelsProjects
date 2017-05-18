//
//  main.cpp
//  Graph-based Decoder
//
//  Created by Wendong Yuan on 10/10/16.
//  Student Number: 8778806
//

#include <iostream>
#include <stdlib.h>
#include <random>
#include <time.h>
#include <iomanip>
#include <math.h>
using namespace std;

// Large data sample
const int sample = 10000;

// Medium data sample
//const int sample = 5000;

// Small data sample
//const int sample = 1000;

// Large runtime
const int runtime = 1000;

// Medium runtime
//const int runtime = 500;

// Small runtime
//const int runtime = 100;

const double PI = 3.14159265358979323846264;

int codeword[7] = {1,0,1,0,1,0,1};

// Values for Gaussian noise generator
const double mean = 0.0000000000;
double variance[4] = {1.0000000000, 0.5000000000, 0.2500000000, 0.1250000000};
// 7/16, 7/32, 7/48, 7/64, 7/80, 7/96
double SNR[7] = {0.4375, 0.21875, 0.145833333, 0.109375, 0.0875, 0.07291667};

vector<vector<vector<double>>> MPAresults;
vector<vector<vector<double>>> SPAresults;
vector<vector<vector<int>>> FactorGraph;

// Transmitted codeword samples
double transCode[sample][7];
// Received codeword samples
double recvCode[sample][7];

int SPAerror = 0;
int MPAerror = 0;
double SPAbitErrorProbability;
double MPAbitErrorProbability;

void initializer(){
   
    for (int s=0; s<sample; s++)
    {
        for(int index=0; index<7; index++)
        {
            if(codeword[index] == 0)
                transCode[s][index] = 1;
            else
                transCode[s][index] = -1;
        }
    }

    for(int codeIndex=0; codeIndex<7;codeIndex++)
    {
        vector<vector<int>> comIndex;
        for(int node=0; node<3; node++)
        {
            vector<int> comNode;
            comNode.push_back(-1);
            comIndex.push_back(comNode);
        }
        FactorGraph.push_back(comIndex);
    }
    
    // Indexes for nodes communicated with the 1st codeword node
    FactorGraph[0][0][0]=2;
    FactorGraph[0][0][1]=4;
    FactorGraph[0][0][2]=6;
    
    // Indexes for nodes communicated with the 2nd codeword node
    FactorGraph[1][0][0]=2;
    FactorGraph[1][0][1]=5;
    FactorGraph[1][0][2]=6;

    // Indexes for nodes communicated with the 4th codeword node
    FactorGraph[3][0][0]=4;
    FactorGraph[3][0][1]=5;
    FactorGraph[3][0][2]=6;

    // Indexes for nodes communicated with the 3rd codeword node
    FactorGraph[2][0][0]=1;
    FactorGraph[2][0][1]=5;
    FactorGraph[2][0][2]=6;
    
    FactorGraph[2][1][0]=0;
    FactorGraph[2][1][1]=4;
    FactorGraph[2][1][2]=6;
    
    // Indexes for nodes communicated with the 5th codeword node
    FactorGraph[4][0][0]=3;
    FactorGraph[4][0][1]=5;
    FactorGraph[4][0][2]=6;
    
    FactorGraph[4][1][0]=0;
    FactorGraph[4][1][1]=2;
    FactorGraph[4][1][2]=6;
    
    // Indexes for nodes communicated with the 6th codeword node
    FactorGraph[5][0][0]=3;
    FactorGraph[5][0][1]=4;
    FactorGraph[5][0][2]=6;
    
    FactorGraph[5][1][0]=1;
    FactorGraph[5][1][1]=2;
    FactorGraph[5][1][2]=6;
    
    // Indexes for nodes communicated with the 7th codeword node
    FactorGraph[6][0][0]=3;
    FactorGraph[6][0][1]=4;
    FactorGraph[6][0][2]=5;
    
    FactorGraph[6][1][0]=1;
    FactorGraph[6][1][1]=2;
    FactorGraph[6][1][2]=5;
    
    FactorGraph[6][2][0]=0;
    FactorGraph[6][2][1]=2;
    FactorGraph[6][2][2]=4;
}

double calculateNormalDistribution(double mean, double variance, double recvCode, int type){
    double p;
    if(type==1)
        p = 1/(sqrt(2*PI*variance))*exp((-1)*(recvCode+1-mean)*(recvCode+1-mean)/(2*variance));
    else if(type==0)
        p = 1/(sqrt(2*PI*variance))*exp((-1)*(recvCode-1-mean)*(recvCode-1-mean)/(2*variance));
    else
    {
        cout<<"Type Error for probability calculation!"<<endl;
        exit(1);
    }
    return p;
}

double LOGSUM(double loga, double logb){
    double diff = loga - logb;
    double logaplusb;
    if (diff>23)
        logaplusb = loga;
    else if (diff<-23)
        logaplusb = logb;
    else
        logaplusb = logb + log(exp(diff) + 1);
    return logaplusb;
}

double MAX(double result1, double result2, double result3, double result4){
    double comp1 = result1 > result2 ? result1 : result2;
    double comp2 = result3 > result4 ? result3 : result4;
    return comp1 > comp2 ? comp1 : comp2;
}

// Decode using Sum-Product Algorithm
void SPA(int sampleIndex, int codeIndex,int nodeNum){
   
    for(int n=0;n<nodeNum;n++)
    {
        int node1 = FactorGraph[codeIndex][n][0];
        int node2 = FactorGraph[codeIndex][n][1];
        int node3 = FactorGraph[codeIndex][n][2];
        
        // Probability for the codeword node to be 0
        double probability_zero =
        SPAresults[sampleIndex][node1][1]*SPAresults[sampleIndex][node2][1]*SPAresults[sampleIndex][node3][0] +
        SPAresults[sampleIndex][node1][1]*SPAresults[sampleIndex][node2][0]*SPAresults[sampleIndex][node3][1] +
        SPAresults[sampleIndex][node1][0]*SPAresults[sampleIndex][node2][1]*SPAresults[sampleIndex][node3][1] +
        SPAresults[sampleIndex][node1][0]*SPAresults[sampleIndex][node2][0]*SPAresults[sampleIndex][node3][0];
        
        
        // Probability for the codeword node to be 1
        double probability_one =
        SPAresults[sampleIndex][node1][1]*SPAresults[sampleIndex][node2][1]*SPAresults[sampleIndex][node3][1] +
        SPAresults[sampleIndex][node1][1]*SPAresults[sampleIndex][node2][0]*SPAresults[sampleIndex][node3][0] +
        SPAresults[sampleIndex][node1][0]*SPAresults[sampleIndex][node2][1]*SPAresults[sampleIndex][node3][0] +
        SPAresults[sampleIndex][node1][0]*SPAresults[sampleIndex][node2][0]*SPAresults[sampleIndex][node3][1];
        
        SPAresults[sampleIndex][codeIndex][0] = LOGSUM(probability_zero, SPAresults[sampleIndex][codeIndex][0]);
        SPAresults[sampleIndex][codeIndex][1] = LOGSUM(probability_one, SPAresults[sampleIndex][codeIndex][1]);
    }
    
}

// Generate Gaussian noise for simulation
void GaussianNoiseGenerator(const double mean, double variance){
    for(int s=0; s<sample; s++)
        for(int index=0; index<7; index++)
        {
            // Use mt19937 algorithm to generate random values
            random_device randomGenerator;
            std::mt19937 randomMT(randomGenerator());
            // create the normal distribution
            normal_distribution<double> distribution(mean, sqrt(variance));
            double GaussianNoise = distribution(randomMT);
            recvCode[s][index] = transCode[s][index] + GaussianNoise;
        }
    
    for(int s=0; s<sample; s++){
        vector<vector<double>> sampleSPA;
        vector<vector<double>> sampleMPA;
        for(int index=0; index<7; index++)
        {
            vector<double> indexSPA;
            vector<double> indexMPA;
            
            double rc = recvCode[s][index];
            // Generate probabilities for SPA & MPA
            indexSPA.push_back(log(calculateNormalDistribution(mean,variance, rc, 0)));
            indexSPA.push_back(log(calculateNormalDistribution(mean,variance, rc, 1)));
            indexMPA.push_back(calculateNormalDistribution(mean, variance, rc, 0));
            indexMPA.push_back(calculateNormalDistribution(mean, variance, rc, 1));

            double temp;
            temp = indexMPA[0];
            indexMPA[0] = indexMPA[0] / (indexMPA[0] + indexMPA[1]);
            indexMPA[1] = indexMPA[1] / (temp + indexMPA[1]);
            
            sampleSPA.push_back(indexSPA);
            sampleMPA.push_back(indexMPA);
        }
        
        SPAresults.push_back(sampleSPA);
        MPAresults.push_back(sampleMPA);
    }
}

// Decode using Max-Product Algorithm
void MPA(int sampleIndex, int codeIndex,int nodeNum){

    for(int n=0;n<nodeNum;n++)
    {
        int node1 = FactorGraph[codeIndex][n][0];
        int node2 = FactorGraph[codeIndex][n][1];
        int node3 = FactorGraph[codeIndex][n][2];
    
        // Probability for the codeword node to be 0
        double probability_zero = MAX(
        MPAresults[sampleIndex][node1][1]*MPAresults[sampleIndex][node2][1]*MPAresults[sampleIndex][node3][0],
        MPAresults[sampleIndex][node1][1]*MPAresults[sampleIndex][node2][0]*MPAresults[sampleIndex][node3][1],
        MPAresults[sampleIndex][node1][0]*MPAresults[sampleIndex][node2][1]*MPAresults[sampleIndex][node3][1],
        MPAresults[sampleIndex][node1][0]*MPAresults[sampleIndex][node2][0]*MPAresults[sampleIndex][node3][0]);
    
        // Probability for the codeword node to be 1
        double probability_one = MAX(
        MPAresults[sampleIndex][node1][1]*MPAresults[sampleIndex][node2][1]*MPAresults[sampleIndex][node3][1],
        MPAresults[sampleIndex][node1][1]*MPAresults[sampleIndex][node2][0]*MPAresults[sampleIndex][node3][0],
        MPAresults[sampleIndex][node1][0]*MPAresults[sampleIndex][node2][1]*MPAresults[sampleIndex][node3][0],
        MPAresults[sampleIndex][node1][0]*MPAresults[sampleIndex][node2][0]*MPAresults[sampleIndex][node3][1]);
    
        double p_zero = probability_zero/(probability_zero + probability_one);
        double p_one = probability_one/(probability_zero + probability_one);
    
        MPAresults[sampleIndex][codeIndex][0] = p_zero > MPAresults[sampleIndex][codeIndex][0] ?
                                            p_zero : MPAresults[sampleIndex][codeIndex][0];
    
        MPAresults[sampleIndex][codeIndex][1] = p_one > MPAresults[sampleIndex][codeIndex][1] ?
                                            p_one : MPAresults[sampleIndex][codeIndex][1];
    }
}

void SPAdecoder(){
    for(int t=0; t<runtime; t++)
        for(int s=0; s<sample; s++)
            for(int codeIndex=0; codeIndex<7; codeIndex++)
            {
                if(codeIndex==0||codeIndex==1||codeIndex==3)
                    SPA(s, codeIndex, 1);
                else if(codeIndex==2||codeIndex==4||codeIndex==5)
                    SPA(s, codeIndex, 2);
                else
                    SPA(s, codeIndex, 3);
            }
}

void MPAdecoder(){
    for(int t=0; t<runtime; t++)
        for(int s=0; s<sample; s++)
            for(int codeIndex=0; codeIndex<7; codeIndex++)
            {
                if(codeIndex==0||codeIndex==1||codeIndex==3)
                    MPA(s, codeIndex, 1);
                else if(codeIndex==2||codeIndex==4||codeIndex==5)
                    MPA(s, codeIndex, 2);
                else
                    MPA(s, codeIndex, 3);
            }
}

void errorAnalyzer(){
    for(int s=0; s<sample; s++)
        for(int codeIndex=0; codeIndex<7; codeIndex++)
        {
            if((codeword[codeIndex]==0 && MPAresults[s][codeIndex][0]<=MPAresults[s][codeIndex][1])||
               (codeword[codeIndex]==1 && MPAresults[s][codeIndex][0]>=MPAresults[s][codeIndex][1]))
                MPAerror++;
            
            if((codeword[codeIndex]==0 && SPAresults[s][codeIndex][0]<=SPAresults[s][codeIndex][1])||
               (codeword[codeIndex]==1 && SPAresults[s][codeIndex][0]>=SPAresults[s][codeIndex][1]))
                SPAerror++;
        }
    
    MPAbitErrorProbability = (double)MPAerror/(7*sample)*100;
    SPAbitErrorProbability = (double)SPAerror/(7*sample)*100;
    
    cout<<"Number of bit error for Max Product Algorithm: "<<MPAerror<<endl;
    cout<<"Number of bit error for Sum Product Algorithm: "<<SPAerror<<endl;
    cout<<"Bit error probability for Max Product Algorithm: "<<MPAbitErrorProbability<<"%"<<endl;
    cout<<"Bit error probability for Sum Product Algorithm: "<<SPAbitErrorProbability<<"%"<<endl;
}

void destructor(){
    MPAerror = 0;
    SPAerror = 0;
    MPAbitErrorProbability = 0;
    SPAbitErrorProbability = 0;
    MPAresults.clear();
    SPAresults.clear();
}

int main(int argc, const char * argv[]) {
    cout<<"Codeword: ";
    for(int codeIndex=0; codeIndex<7; codeIndex++)
        cout<<codeword[codeIndex];
    cout<<endl<<endl;
    for(int var=0; var<4; var++)
    {
        cout<<"Variance: "<<variance[var]<<endl;
        initializer();
        GaussianNoiseGenerator(mean, variance[var]);
        MPAdecoder();
        SPAdecoder();
        errorAnalyzer();
        destructor();
        cout<<endl;
    }
    for(int var=0; var<6; var++)
    {
        cout<<"Eb/No: "<<2*(var+1)<<endl;
        initializer();
        GaussianNoiseGenerator(mean, (double)SNR[var]);
        MPAdecoder();
        SPAdecoder();
        errorAnalyzer();
        destructor();
        cout<<endl;
    }
    return 0;
}
