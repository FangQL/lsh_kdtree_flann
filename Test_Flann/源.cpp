#include "flann/flann.hpp"
#include "flann/io/hdf5.h"

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <ctime>
#include <random>
#include <string>
#include <fstream>
#include <math.h>
#include <vector>
#include <iterator>


using namespace flann;
using namespace std;

long long flann::count_calculate_distance_ = 0;
long flann::buckets_total_num = 0;
string GroundTruthFile = "sift_groundtruth.hdf5";

float Evluation_recall(string GroundTruth, string Result, int nn)
{
	Matrix<int> truth;
	Matrix<int> result;
	float recall = 0.0;

	load_from_file(truth, GroundTruth, "groundtruth");
	load_from_file(result, Result, "result");

	for (int i = 0; i < truth.rows; i++)
	{
		std::vector<int> TruthSet;
		std::vector<int> ResultSet;
		std::vector<int> IntersectSet;
		TruthSet.reserve(nn);
		ResultSet.reserve(nn);
		IntersectSet.reserve(nn);
		for (int j = 0; j < nn; j++)
		{
			TruthSet.push_back(*(truth[i] + j));
			ResultSet.push_back(*(result[i] + j));
		}
		std::sort(TruthSet.begin(), TruthSet.end());
		std::sort(ResultSet.begin(), ResultSet.end());
		std::set_intersection(TruthSet.begin(), TruthSet.end(), ResultSet.begin(), ResultSet.end(), std::back_inserter(IntersectSet));
		recall += (float(IntersectSet.size()) / nn);
	}
	recall = recall / truth.rows;

	return recall;
}

float Evluation_Error_ratio(string GroundTruth, string Result, int nn, Matrix<float> dataset, Matrix<float> query)
{
	Matrix<int> truth;
	Matrix<int> result;
	float error_ratio = 0.0;

	load_from_file(truth, GroundTruth, "groundtruth");
	load_from_file(result, Result, "result");

	for (int i = 0; i < truth.rows; i++)
	{
		for (int j = 0; j < nn; j++)
		{
			float distance_truth = 0.0;
			float distance_result = 0.0;
			float* truth_vector_begin = dataset[*(truth[i] + j)];
			if (*(result[i] + j) == -842150451)
				*(result[i] + j) = 0;
			float* result_vector_begin = dataset[*(result[i] + j)];
			float* query_vector_begin = query[i];

			for (int k = 0; k < dataset.cols; k++)
			{
				float diff_truth = *truth_vector_begin - *query_vector_begin;
				distance_truth += diff_truth*diff_truth;
				float diff_result = *result_vector_begin - *query_vector_begin;
				distance_result += diff_result*diff_result;
				truth_vector_begin++;
				result_vector_begin++;
				query_vector_begin++;
			}

			error_ratio += distance_result / distance_truth;
		}
	}
	error_ratio = error_ratio / int(truth.rows) / nn;

	return error_ratio;

}


void Research_For_Kdtree(string InputFileName, string OutputFileName, 
						 int TreeNum, int MaxCheck, float eps)
{
	time_t start1, start2, end, time_buildindex, time_findnn;//Calculation of time
	int nn = 20;
	count_calculate_distance_ = 0;
	ofstream RecordFile;
	RecordFile.open("RecordFile_KdTree.txt", ios::app);

	Matrix<float> dataset;
	Matrix<float> query;
	load_from_file(dataset, InputFileName, "base");
	load_from_file(query, InputFileName, "query");
	Matrix<int> indices(new int[query.rows*nn], query.rows, nn);
	Matrix<float> dists(new float[query.rows*nn], query.rows, nn);

	start1 = clock();
	Index<L2<float>> index(dataset, flann::KDTreeIndexParams(TreeNum));
	index.buildIndex();
	start2 = clock();
	cout << "Build_index_end" << endl;

	index.knnSearch(query, indices, dists, nn, flann::SearchParams(MaxCheck, eps));
	end = clock();
	cout << "Find_nn_end" << endl;

	vector<char[20]> ParamChar(3);
	sprintf(ParamChar[0], "%d", TreeNum);
	sprintf(ParamChar[1], "%d", MaxCheck);
	sprintf(ParamChar[2], "%.2f", eps);
	OutputFileName = OutputFileName + "TreeNum" + ParamChar[0] + '_' + "MaxCheck" + ParamChar[1] + '_' + "eps" + ParamChar[2] + ".hdf5";
	
	flann::save_to_file(indices, OutputFileName, "result");

	time_buildindex = start2 - start1;
	time_findnn = end - start2;
	int IndexMemory = index.usedMemory();
	float recall = Evluation_recall(GroundTruthFile, OutputFileName, nn);
	float error_ratio = Evluation_Error_ratio(GroundTruthFile, OutputFileName, nn, dataset, query);

	RecordFile << OutputFileName << ':' << endl;
	RecordFile << "time_buildindex:" << time_buildindex << endl;
	RecordFile << "time_findnn:" << time_findnn << endl;
	RecordFile << "index_memory:" << IndexMemory << endl;
	RecordFile << "distance_cal:" << count_calculate_distance_ << endl;
	RecordFile << "recall:" << recall << endl;
	RecordFile << "errorratio:" << error_ratio << endl;
	RecordFile << endl;
	RecordFile.close();

	cout << "END" << endl;
	cout << endl;

	delete[] dataset.ptr();
	delete[] query.ptr();
	delete[] indices.ptr();
	delete[] dists.ptr();

}


void Research_For_LSH(string InputFileName, string OutputFileName,
	int TableNum, int KeySize, unsigned int ProbeLevel)
{
	time_t start1, start2, end, time_buildindex, time_findnn;//Calculation of time
	int nn = 20;
	count_calculate_distance_ = 0;
	buckets_total_num = 0;
	ofstream RecordFile;
	RecordFile.open("RecordFile_LSH.txt", ios::app);

	Matrix<float> dataset;
	Matrix<float> query;
	load_from_file(dataset, InputFileName, "base");
	load_from_file(query, InputFileName, "query");
	Matrix<int> indices(new int[query.rows*nn], query.rows, nn);
	Matrix<float> dists(new float[query.rows*nn], query.rows, nn);

	start1 = clock();
	Index<L2<float>> index(dataset, flann::LshIndexParams(TableNum, KeySize, ProbeLevel));
	index.buildIndex();
	start2 = clock();
	cout << "Build_index_end" << endl;

	index.knnSearch(query, indices, dists, nn, flann::SearchParams());
	end = clock();
	cout << "Find_nn_end" << endl;

	vector<char[20]> ParamChar(5);
	sprintf(ParamChar[0], "%d", TableNum);
	sprintf(ParamChar[1], "%d", KeySize);
	sprintf(ParamChar[2], "%d", ProbeLevel);
	OutputFileName = OutputFileName + "TableNum" + ParamChar[0] + '_' + "KeySize" + ParamChar[1] + '_' + "ProbeLevel" + ParamChar[2] + ".hdf5";

	flann::save_to_file(indices, OutputFileName, "result");

	time_buildindex = start2 - start1;
	time_findnn = end - start2;
	int IndexMemory = index.usedMemory();
	float recall = Evluation_recall(GroundTruthFile, OutputFileName, nn);
	float error_ratio = Evluation_Error_ratio(GroundTruthFile, OutputFileName, nn, dataset, query);

	RecordFile << OutputFileName << ':' << endl;
	RecordFile << "time_buildindex:" << time_buildindex << endl;
	RecordFile << "time_findnn:" << time_findnn << endl;
	RecordFile << "index_memory:" << IndexMemory << endl;
	RecordFile << "distance_cal:" << count_calculate_distance_ << endl;
	RecordFile << "buckets_total_num:" << buckets_total_num << endl;
	RecordFile << "recall:" << recall << endl;
	RecordFile << "errorratio:" << error_ratio << endl;
	RecordFile << endl;
	RecordFile.close();

	cout << "END" << endl;
	cout << endl;

	delete[] dataset.ptr();
	delete[] query.ptr();
	delete[] indices.ptr();
	delete[] dists.ptr();

}


int main(int argc, char** argv)
{
	
	/*Research For KdTree*/
	//Research_For_Kdtree("sift_base.hdf5", "sift", 10, 1000, 0.1);
	//Research_For_Kdtree("sift_base.hdf5", "sift", 10, 2000, 0.1);
	//Research_For_Kdtree("sift_base.hdf5", "sift", 10, 3000, 0.1);
	//Research_For_Kdtree("sift_base.hdf5", "sift", 10, 500, 0.1);
	//Research_For_Kdtree("sift_base.hdf5", "sift", 10, 4000, 0.1);
	//Research_For_Kdtree("sift_base.hdf5", "sift", 10, 6000, 0.1);

	//Research_For_Kdtree("sift_base.hdf5", "sift", 20, 1000, 0.1);
	//Research_For_Kdtree("sift_base.hdf5", "sift", 30, 1000, 0.1);
	//Research_For_Kdtree("sift_base.hdf5", "sift", 40, 1000, 0.1);

	//Research_For_Kdtree("sift_base.hdf5", "sift", 10, 1000, 0.01);
	//Research_For_Kdtree("sift_base.hdf5", "sift", 10, 1000, 0.5);
	Research_For_Kdtree("sift_base.hdf5", "sift", 10, 1000, 5.0);

	/*Research For LSH*/
	/*
	Research_For_LSH("sift_base.hdf5", "sift_W500_LSH", 1, 6, 1);

	Research_For_LSH("sift_base.hdf5", "sift_W500_LSH", 3, 6, 1);

	Research_For_LSH("sift_base.hdf5", "sift_W500_LSH", 5, 6, 0);
	Research_For_LSH("sift_base.hdf5", "sift_W500_LSH", 5, 6, 1);
	
	Research_For_LSH("sift_base.hdf5", "sift_W500_LSH", 10, 6, 0);
	Research_For_LSH("sift_base.hdf5", "sift_W500_LSH", 10, 6, 1);

	Research_For_LSH("sift_base.hdf5", "sift_W500_LSH", 15, 6, 0);
	Research_For_LSH("sift_base.hdf5", "sift_W500_LSH", 15, 6, 1);

	Research_For_LSH("sift_base.hdf5", "sift_W500_LSH", 20, 6, 0);
	Research_For_LSH("sift_base.hdf5", "sift_W500_LSH", 20, 6, 1);

	Research_For_LSH("sift_base.hdf5", "sift_W500_LSH", 40, 6, 0);

	Research_For_LSH("sift_base.hdf5", "sift_W500_LSH", 80, 6, 0);

	Research_For_LSH("sift_base.hdf5", "sift_W500_LSH", 50, 6, 0);
	Research_For_LSH("sift_base.hdf5", "sift_W500_LSH", 60, 6, 0);
	Research_For_LSH("sift_base.hdf5", "sift_W500_LSH", 70, 6, 0);
	
	Research_For_LSH("sift_base.hdf5", "sift_W500_LSH", 8, 6, 1);
	Research_For_LSH("sift_base.hdf5", "sift_W500_LSH", 13, 6, 1);
	Research_For_LSH("sift_base.hdf5", "sift_W500_LSH", 18, 6, 1);
	*/

	return 0;
}
