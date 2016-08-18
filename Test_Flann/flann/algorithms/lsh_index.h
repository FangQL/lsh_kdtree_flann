/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
 *
 * THE BSD LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

/***********************************************************************
 * Author: Vincent Rabaud
 *************************************************************************/

#ifndef FLANN_LSH_INDEX_H_
#define FLANN_LSH_INDEX_H_

#include <algorithm>
#include <cassert>
#include <cstring>
#include <map>
#include <vector>

#include "flann/general.h"
//#include "flann/algorithms/nn_index.h"
#include "flann/util/matrix.h"
#include "flann/util/result_set.h"
#include "flann/util/heap.h"
#include "flann/util/lsh_table.h"
#include "flann/util/allocator.h"
#include "flann/util/random.h"
#include "flann/util/saving.h"

namespace flann
{

struct LshIndexParams : public IndexParams
{
    LshIndexParams(unsigned int table_number = 12, unsigned int key_size = 20, unsigned int multi_probe_level = 2)
    {
        (* this)["algorithm"] = FLANN_INDEX_LSH;
        // The number of hash tables to use
        (*this)["table_number"] = table_number;
        // The length of the key in the hash tables
        (*this)["key_size"] = key_size;
        // Number of levels to use in multi-probe (0 for standard LSH)
        (*this)["multi_probe_level"] = multi_probe_level;
    }
};



/**
 * Randomized kd-tree index
 *
 * Contains the k-d trees and other information for indexing a set of points
 * for nearest-neighbor matching.
 */
template<typename Distance>
class LshIndex// : public NNIndex<Distance>
{
public:
    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;

    //typedef NNIndex<Distance> BaseClass;

    /** Constructor
     * @param params parameters passed to the LSH algorithm
     * @param d the distance used
     */
    LshIndex(const IndexParams& params = LshIndexParams(), Distance d = Distance()) :
		distance_(d), last_id_(0), size_(0), size_at_build_(0), veclen_(0),
		index_params_(params), removed_(false), removed_count_(0), data_ptr_(NULL)
    {
        table_number_ = get_param<unsigned int>(index_params_,"table_number",12);
        key_size_ = get_param<unsigned int>(index_params_,"key_size",20);
        multi_probe_level_ = get_param<unsigned int>(index_params_,"multi_probe_level",2);

        fill_xor_mask(0, key_size_, multi_probe_level_, xor_masks_);
    }


    /** Constructor
     * @param input_data dataset with the input features
     * @param params parameters passed to the LSH algorithm
     * @param d the distance used
     */
    LshIndex(const Matrix<ElementType>& input_data, const IndexParams& params = LshIndexParams(), Distance d = Distance()) :
		distance_(d), last_id_(0), size_(0), size_at_build_(0), veclen_(0),
		index_params_(params), removed_(false), removed_count_(0), data_ptr_(NULL)
    {
        table_number_ = get_param<unsigned int>(index_params_,"table_number",12);
        key_size_ = get_param<unsigned int>(index_params_,"key_size",20);
        multi_probe_level_ = get_param<unsigned int>(index_params_,"multi_probe_level",2);

		//change form size_t to int 
		std::vector<int> zero_mask_(key_size_, 0);
//        fill_xor_mask(zero_mask_, key_size_, multi_probe_level_, xor_masks_);

        setDataset(input_data);
    }

    LshIndex(const LshIndex& other) : BaseClass(other),
    	tables_(other.tables_),
    	table_number_(other.table_number_),
    	key_size_(other.key_size_),
    	multi_probe_level_(other.multi_probe_level_),
    	xor_masks_(other.xor_masks_)
    {
    }
    
    LshIndex& operator=(LshIndex other)
    {
    	this->swap(other);
    	return *this;
    }

    virtual ~LshIndex()
    {
    	freeIndex();
    }


	LshIndex* clone() const
    {
    	return new LshIndex(*this);
    }
    
    //using BaseClass::buildIndex;
	/*Copy from NN_Index*/
	void buildIndex()
	{
		freeIndex();

		cleanRemovedPoints();

		buildIndexImpl();

		size_at_build_ = size_;
	}

    void addPoints(const Matrix<ElementType>& points, float rebuild_threshold = 2)
    {
        assert(points.cols==veclen_);
        size_t old_size = size_;

        extendDataset(points);
        
        if (rebuild_threshold>1 && size_at_build_*rebuild_threshold<size_) {
            buildIndex();
        }
        else {
            for (unsigned int i = 0; i < table_number_; ++i) {
                lsh::LshTable<ElementType>& table = tables_[i];                
                for (size_t i=old_size;i<size_;++i) {
                    table.add(i, points_[i]);
                }            
            }
        }
    }


    flann_algorithm_t getType() const
    {
        return FLANN_INDEX_LSH;
    }


    template<typename Archive>
    void serialize(Archive& ar)
    {
    	ar.setObject(this);

    	ar & *static_cast<NNIndex<Distance>*>(this);

    	ar & table_number_;
    	ar & key_size_;
    	ar & multi_probe_level_;

    	ar & xor_masks_;
    	ar & tables_;

    	if (Archive::is_loading::value) {
            index_params_["algorithm"] = getType();
            index_params_["table_number"] = table_number_;
            index_params_["key_size"] = key_size_;
            index_params_["multi_probe_level"] = multi_probe_level_;
    	}
    }

    void saveIndex(FILE* stream)
    {
    	serialization::SaveArchive sa(stream);
    	sa & *this;
    }

    void loadIndex(FILE* stream)
    {
    	serialization::LoadArchive la(stream);
    	la & *this;
    }

    /**
     * Computes the index memory usage
     * Returns: memory used by the index
     */
    int usedMemory() const
    {
        return size_ * sizeof(int);
		
    }

	/*inline function copy from NN_Index*/
	inline size_t veclen() const
	{
		return veclen_;
	}

    /**
     * \brief Perform k-nearest neighbor search
     * \param[in] queries The query points for which to find the nearest neighbors
     * \param[out] indices The indices of the nearest neighbors found
     * \param[out] dists Distances to the nearest neighbors found
     * \param[in] knn Number of nearest neighbors to return
     * \param[in] params Search parameters
     */
    int knnSearch(const Matrix<ElementType>& queries,
						Matrix<size_t>& indices,
    					Matrix<DistanceType>& dists,
    					size_t knn,
    					const SearchParams& params) const
    {
        assert(queries.cols == veclen_);
        assert(indices.rows >= queries.rows);
        assert(dists.rows >= queries.rows);
        assert(indices.cols >= knn);
        assert(dists.cols >= knn);

        int count = 0;
        if (params.use_heap==FLANN_True) {
#pragma omp parallel num_threads(params.cores)
        	{
        		KNNUniqueResultSet<DistanceType> resultSet(knn);
#pragma omp for schedule(static) reduction(+:count)
        		for (int i = 0; i < (int)queries.rows; i++) {
        			resultSet.clear();
        			findNeighbors(resultSet, queries[i], params);
        			size_t n = std::min(resultSet.size(), knn);
        			resultSet.copy(indices[i], dists[i], n, params.sorted);
        			indices_to_ids(indices[i], indices[i], n);
        			count += n;
        		}
        	}
        }
        else {
#pragma omp parallel num_threads(params.cores)
        	{
        		//KNNResultSet<DistanceType> resultSet(knn);
				KNNUniqueResultSet<DistanceType> resultSet(knn);
#pragma omp for schedule(static) reduction(+:count)
        		for (int i = 0; i < (int)queries.rows; i++) {
        			resultSet.clear();
        			findNeighbors(resultSet, queries[i], params);
        			size_t n = std::min(resultSet.size(), knn);
        			resultSet.copy(indices[i], dists[i], n, params.sorted);
        			indices_to_ids(indices[i], indices[i], n);
        			count += n;
        		}
        	}
        }

        return count;
    }

	/*Copy from above for size_t to int*/
	int knnSearch(const Matrix<ElementType>& queries,
		Matrix<int>& indices,
		Matrix<DistanceType>& dists,
		size_t knn,
		const SearchParams& params) const
	{
		flann::Matrix<size_t> indices_(new size_t[indices.rows*indices.cols], indices.rows, indices.cols);

		int result = knnSearch(queries, indices_, dists, knn, params);

		for (size_t i = 0; i<indices.rows; ++i) {
			for (size_t j = 0; j<indices.cols; ++j) {
				indices[i][j] = indices_[i][j];
			}
		}
		delete[] indices_.ptr();
		return result;
	}

    /**
     * \brief Perform k-nearest neighbor search
     * \param[in] queries The query points for which to find the nearest neighbors
     * \param[out] indices The indices of the nearest neighbors found
     * \param[out] dists Distances to the nearest neighbors found
     * \param[in] knn Number of nearest neighbors to return
     * \param[in] params Search parameters
     */
    int knnSearch(const Matrix<ElementType>& queries,
					std::vector< std::vector<size_t> >& indices,
					std::vector<std::vector<DistanceType> >& dists,
    				size_t knn,
    				const SearchParams& params) const
    {
        assert(queries.cols == veclen_);
		if (indices.size() < queries.rows ) indices.resize(queries.rows);
		if (dists.size() < queries.rows ) dists.resize(queries.rows);

		int count = 0;
		if (params.use_heap==FLANN_True) {
#pragma omp parallel num_threads(params.cores)
			{
				KNNUniqueResultSet<DistanceType> resultSet(knn);
#pragma omp for schedule(static) reduction(+:count)
				for (int i = 0; i < (int)queries.rows; i++) {
					resultSet.clear();
					findNeighbors(resultSet, queries[i], params);
					size_t n = std::min(resultSet.size(), knn);
					indices[i].resize(n);
					dists[i].resize(n);
					if (n > 0) {
						resultSet.copy(&indices[i][0], &dists[i][0], n, params.sorted);
						indices_to_ids(&indices[i][0], &indices[i][0], n);
					}
					count += n;
				}
			}
		}
		else {
#pragma omp parallel num_threads(params.cores)
			{
				KNNUniqueResultSet<DistanceType> resultSet(knn);
#pragma omp for schedule(static) reduction(+:count)
				for (int i = 0; i < (int)queries.rows; i++) {
					resultSet.clear();
					findNeighbors(resultSet, queries[i], params);
					size_t n = std::min(resultSet.size(), knn);
					indices[i].resize(n);
					dists[i].resize(n);
					if (n > 0) {
						resultSet.copy(&indices[i][0], &dists[i][0], n, params.sorted);
						indices_to_ids(&indices[i][0], &indices[i][0], n);
					}
					count += n;
				}
			}
		}

		return count;
    }

    /**
     * Find set of nearest neighbors to vec. Their indices are stored inside
     * the result object.
     *
     * Params:
     *     result = the result object in which the indices of the nearest-neighbors are stored
     *     vec = the vector for which to search the nearest neighbors
     *     maxCheck = the maximum number of restarts (in a best-bin-first manner)
     */
    //void findNeighbors(ResultSet<DistanceType>& result, const ElementType* vec, const SearchParams& /*searchParams*/) const
	void findNeighbors(KNNUniqueResultSet<DistanceType>& result, const ElementType* vec, const SearchParams& /*searchParams*/) const
    {
        getNeighbors(vec, result); 
    }

protected:

    /**
     * Builds the index
     */
    void buildIndexImpl()
    {
        tables_.resize(table_number_);
        std::vector<std::pair<size_t,ElementType*> > features;
        features.reserve(points_.size());
        for (size_t i=0;i<points_.size();++i) {
			features.push_back(std::make_pair(i, points_[i]));
        }
        for (unsigned int i = 0; i < table_number_; ++i) {
            lsh::LshTable<ElementType>& table = tables_[i];
            table = lsh::LshTable<ElementType>(veclen_, key_size_);

            // Add the features to the table
            table.add(features);
			buckets_total_num += table.usedMemory();
        }
    }

	/*Copy from NN_Index*/
	void cleanRemovedPoints()
	{
		if (!removed_) return;

		size_t last_idx = 0;
		for (size_t i = 0; i<size_; ++i) {
			if (!removed_points_.test(i)) {
				points_[last_idx] = points_[i];
				ids_[last_idx] = ids_[i];
				removed_points_.reset(last_idx);
				++last_idx;
			}
		}
		points_.resize(last_idx);
		ids_.resize(last_idx);
		removed_points_.resize(last_idx);
		size_ = last_idx;
		removed_count_ = 0;
	}//END_CleanRemovedPoints

	 /*Copy from NN_Index*/
	size_t id_to_index(size_t id)
	{
		if (ids_.size() == 0) {
			return id;
		}
		size_t point_index = size_t(-1);
		if (ids_[id] == id) {
			return id;
		}
		else {
			// binary search
			size_t start = 0;
			size_t end = ids_.size();

			while (start<end) {
				size_t mid = (start + end) / 2;
				if (ids_[mid] == id) {
					point_index = mid;
					break;
				}
				else if (ids_[mid]<id) {
					start = mid + 1;
				}
				else {
					end = mid;
				}
			}
		}
		return point_index;
	}

	/*Copy from NN_Index*/
	void indices_to_ids(const size_t* in, size_t* out, size_t size) const
	{
		if (removed_) {
			for (size_t i = 0; i<size; ++i) {
				out[i] = ids_[in[i]];
			}
		}
	}

    void freeIndex()
    {
        /* nothing to do here */
    }

	/*---------------------NN_Index Parameters --------------------------*/

	/**
	* The distance functor
	*/
	Distance distance_;

	/**
	* Each index point has an associated ID. IDs are assigned sequentially in
	* increasing order. This indicates the ID assigned to the last point added to the
	* index.
	*/
	size_t last_id_;

	/**
	* Number of points in the index (and database)
	*/
	size_t size_;

	/**
	* Number of features in the dataset when the index was last built.
	*/
	size_t size_at_build_;

	/**
	* Size of one point in the index (and database)
	*/
	size_t veclen_;

	/**
	* Parameters of the index.
	*/
	IndexParams index_params_;

	/**
	* Flag indicating if at least a point was removed from the index
	*/
	bool removed_;

	/**
	* Array used to mark points removed from the index
	*/
	DynamicBitset removed_points_;

	/**
	* Number of points removed from the index
	*/
	size_t removed_count_;

	/**
	* Array of point IDs, returned by nearest-neighbour operations
	*/
	std::vector<size_t> ids_;

	/**
	* Point data
	*/
	std::vector<ElementType*> points_;

	/**
	* Pointer to dataset memory if allocated by this index, otherwise NULL
	*/
	ElementType* data_ptr_;

private:

	/*Copy from NN_Index*/
	void setDataset(const Matrix<ElementType>& dataset)
	{
		size_ = dataset.rows;
		veclen_ = dataset.cols;
		last_id_ = 0;

		ids_.clear();
		removed_points_.clear();
		removed_ = false;
		removed_count_ = 0;

		points_.resize(size_);
		for (size_t i = 0; i<size_; ++i) {
			points_[i] = dataset[i];
		}
	}//END_SETDATA

    /** Defines the comparator on score and index
     */
    typedef std::pair<float, unsigned int> ScoreIndexPair;
    struct SortScoreIndexPairOnSecond
    {
        bool operator()(const ScoreIndexPair& left, const ScoreIndexPair& right) const
        {
            return left.second < right.second;
        }
    };

    /** Fills the different xor masks to use when getting the neighbors in multi-probe LSH
     * @param key the key we build neighbors from
     * @param lowest_index the lowest index of the bit set
     * @param level the multi-probe level we are at
     * @param xor_masks all the xor mask
     */
	/*Original version*/
	/*
    void fill_xor_mask(lsh::BucketKey key, int lowest_index, unsigned int level,
                       std::vector<lsh::BucketKey>& xor_masks)
    {
        xor_masks.push_back(key);
        if (level == 0) return;
        for (int index = lowest_index - 1; index >= 0; --index) {
            // Create a new key
            lsh::BucketKey new_key = key | (1 << index);
            fill_xor_mask(new_key, index, level - 1, xor_masks);
        }
    }
	*/

	/*
	void fill_xor_mask(lsh::BucketKey key, int lowest_index, unsigned int level,
		std::vector<lsh::BucketKey>& xor_masks)
	{
		xor_masks.push_back(key);
		if (level == 0) return;

		for (int index = 0; index < key_size_; index++)
		{
			lsh::BucketKey tmpkey = key;
			tmpkey[index] = 1;
			xor_masks.push_back(tmpkey);
		}

		for (int index = 0; index < key_size_; index++)
		{
			lsh::BucketKey tmpkey = key;
			tmpkey[index] = -1;
			xor_masks.push_back(tmpkey);
		}
		
	}
	*/

    /** Performs the approximate nearest-neighbor search.
     * @param vec the feature to analyze
     * @param do_radius flag indicating if we check the radius too
     * @param radius the radius if it is a radius search
     * @param do_k flag indicating if we limit the number of nn
     * @param k_nn the number of nearest neighbors
     * @param checked_average used for debugging
     */
    void getNeighbors(const ElementType* vec, bool do_radius, float radius, bool do_k, unsigned int k_nn,
                      float& checked_average)
    {
        static std::vector<ScoreIndexPair> score_index_heap;

        if (do_k) {
            unsigned int worst_score = std::numeric_limits<unsigned int>::max();
            typename std::vector<lsh::LshTable<ElementType> >::const_iterator table = tables_.begin();
            typename std::vector<lsh::LshTable<ElementType> >::const_iterator table_end = tables_.end();
            for (; table != table_end; ++table) {
                size_t key = table->getKey(vec);
                std::vector<lsh::BucketKey>::const_iterator xor_mask = xor_masks_.begin();
                std::vector<lsh::BucketKey>::const_iterator xor_mask_end = xor_masks_.end();
                for (; xor_mask != xor_mask_end; ++xor_mask) {
                    size_t sub_key = key ^ (*xor_mask);
                    const lsh::Bucket* bucket = table->getBucketFromKey(sub_key);
                    if (bucket == 0) continue;

                    // Go over each descriptor index
                    std::vector<lsh::FeatureIndex>::const_iterator training_index = bucket->begin();
                    std::vector<lsh::FeatureIndex>::const_iterator last_training_index = bucket->end();
                    DistanceType hamming_distance;

                    // Process the rest of the candidates
                    for (; training_index < last_training_index; ++training_index) {
                    	if (removed_ && removed_points_.test(*training_index)) continue;
                        hamming_distance = distance_(vec, points_[*training_index].point, veclen_);

                        if (hamming_distance < worst_score) {
                            // Insert the new element
                            score_index_heap.push_back(ScoreIndexPair(hamming_distance, training_index));
                            std::push_heap(score_index_heap.begin(), score_index_heap.end());

                            if (score_index_heap.size() > (unsigned int)k_nn) {
                                // Remove the highest distance value as we have too many elements
                                std::pop_heap(score_index_heap.begin(), score_index_heap.end());
                                score_index_heap.pop_back();
                                // Keep track of the worst score
                                worst_score = score_index_heap.front().first;
                            }
                        }
                    }
                }
            }
        }
        else {
            typename std::vector<lsh::LshTable<ElementType> >::const_iterator table = tables_.begin();
            typename std::vector<lsh::LshTable<ElementType> >::const_iterator table_end = tables_.end();
            for (; table != table_end; ++table) {
                size_t key = table->getKey(vec);
                std::vector<lsh::BucketKey>::const_iterator xor_mask = xor_masks_.begin();
                std::vector<lsh::BucketKey>::const_iterator xor_mask_end = xor_masks_.end();
                for (; xor_mask != xor_mask_end; ++xor_mask) {
                    size_t sub_key = key ^ (*xor_mask);
                    const lsh::Bucket* bucket = table->getBucketFromKey(sub_key);
                    if (bucket == 0) continue;

                    // Go over each descriptor index
                    std::vector<lsh::FeatureIndex>::const_iterator training_index = bucket->begin();
                    std::vector<lsh::FeatureIndex>::const_iterator last_training_index = bucket->end();
                    DistanceType hamming_distance;

                    // Process the rest of the candidates
                    for (; training_index < last_training_index; ++training_index) {
                    	if (removed_ && removed_points_.test(*training_index)) continue;
                        // Compute the Hamming distance
                        hamming_distance = distance_(vec, points_[*training_index].point, veclen_);
                        if (hamming_distance < radius) score_index_heap.push_back(ScoreIndexPair(hamming_distance, training_index));
                    }
                }
            }
        }
    }

    /** Performs the approximate nearest-neighbor search.
     * This is a slower version than the above as it uses the ResultSet
     * @param vec the feature to analyze
     */
    //void getNeighbors(const ElementType* vec, ResultSet<DistanceType>& result) const
	void getNeighbors(const ElementType* vec, KNNUniqueResultSet<DistanceType>& result) const
    {
		typename std::vector<lsh::LshTable<ElementType> >::const_iterator table = tables_.begin();
		typename std::vector<lsh::LshTable<ElementType> >::const_iterator table_end = tables_.end();
		for (; table != table_end; ++table)
		{
			std::vector<float> key = table->getKey(vec);
			std::vector<int> int_key(key_size_);
			for (int i = 0; i < key_size_; i++)
				int_key[i] = floor(key[i]);
			int perturbation_step = key_size_;

			const lsh::Bucket* bucket = table->getBucketFromKey(int_key);
			if (bucket == 0)
			{

			}
			else
			{
				std::vector<lsh::FeatureIndex>::const_iterator training_index = bucket->begin();
				std::vector<lsh::FeatureIndex>::const_iterator last_training_index = bucket->end();
				DistanceType L2_distance;

				// Process the rest of the candidates
				for (; training_index < last_training_index; ++training_index) 
				{
					if (removed_ && removed_points_.test(*training_index)) continue;
					count_calculate_distance_++;
					L2_distance = distance_(vec, points_[*training_index], veclen_);
					result.addPoint(L2_distance, *training_index);
				}
			}

			/*Mutli probe = 1*/
			if (multi_probe_level_ == 1)
			{
				std::vector<std::pair<float, int>> perturbation_pair;

				for (int i = 0; i < key_size_; i++)
				{
					float dis_to_floor = key[i] - int_key[i];
					perturbation_pair.push_back(std::make_pair(dis_to_floor, i));
					perturbation_pair.push_back(std::make_pair(1.0 - dis_to_floor, i + key_size_));
				}

				std::sort(perturbation_pair.begin(), perturbation_pair.end());

				for (int PairIndex = 0; PairIndex < perturbation_step; PairIndex++)
				{
					std::vector<int> sub_key = int_key;
					int ModifyIndex = perturbation_pair[PairIndex].second % key_size_;
					sub_key[ModifyIndex] = sub_key[ModifyIndex] + 2 * (perturbation_pair[PairIndex].second / key_size_) - 1;
					const lsh::Bucket* bucket = table->getBucketFromKey(sub_key);
					if (bucket == 0) continue;

					std::vector<lsh::FeatureIndex>::const_iterator training_index = bucket->begin();
					std::vector<lsh::FeatureIndex>::const_iterator last_training_index = bucket->end();
					DistanceType L2_distance;

					// Process the rest of the candidates
					for (; training_index < last_training_index; ++training_index) {
						if (removed_ && removed_points_.test(*training_index)) continue;
						count_calculate_distance_++;
						L2_distance = distance_(vec, points_[*training_index], veclen_);
						result.addPoint(L2_distance, *training_index);
					}
				}
			}
				/*End Mutil probe 1*/
				
				/*Just test*/
			if (multi_probe_level_ == 2)
			{
				std::vector<std::pair<float, int>> perturbation_pair;

				for (int i = 0; i < key_size_; i++)
				{
					float dis_to_floor = key[i] - int_key[i];
					perturbation_pair.push_back(std::make_pair(dis_to_floor, i));
					perturbation_pair.push_back(std::make_pair(1.0 - dis_to_floor, i + key_size_));
				}

				std::sort(perturbation_pair.begin(), perturbation_pair.end());

				for (int PairIndex = 0; PairIndex < perturbation_step; PairIndex++)
				{
					std::vector<int> sub_key = int_key;
					int ModifyIndex = perturbation_pair[PairIndex].second % key_size_;
					sub_key[ModifyIndex] = sub_key[ModifyIndex] + 2 * (perturbation_pair[PairIndex].second / key_size_) - 1;
					const lsh::Bucket* bucket = table->getBucketFromKey(sub_key);
					if (bucket == 0) continue;

					std::vector<lsh::FeatureIndex>::const_iterator training_index = bucket->begin();
					std::vector<lsh::FeatureIndex>::const_iterator last_training_index = bucket->end();

					std::vector<lsh::FeatureIndex>::const_iterator check_index = bucket->begin();
					std::vector<lsh::FeatureIndex>::const_iterator last_check_index = bucket->end();

					DistanceType L2_distance;

					// Process the rest of the candidates
					for (; training_index < last_training_index; ++training_index) {
						if (removed_ && removed_points_.test(*training_index)) continue;
						count_calculate_distance_++;
						L2_distance = distance_(vec, points_[*training_index], veclen_);
						result.addPoint(L2_distance, *training_index);
					}
				}
				/*End test*/
			}

		}
		/*
        typename std::vector<lsh::LshTable<ElementType> >::const_iterator table = tables_.begin();
        typename std::vector<lsh::LshTable<ElementType> >::const_iterator table_end = tables_.end();
        for (; table != table_end; ++table) {
            std::vector<size_t> key = table->getKey(vec);
            std::vector<lsh::BucketKey>::const_iterator xor_mask = xor_masks_.begin();
            std::vector<lsh::BucketKey>::const_iterator xor_mask_end = xor_masks_.end();
            for (; xor_mask != xor_mask_end; ++xor_mask) {
                size_t sub_key = key ^ (*xor_mask);
                const lsh::Bucket* bucket = table->getBucketFromKey(sub_key);
                if (bucket == 0) continue;

                // Go over each descriptor index
                std::vector<lsh::FeatureIndex>::const_iterator training_index = bucket->begin();
                std::vector<lsh::FeatureIndex>::const_iterator last_training_index = bucket->end();
                DistanceType hamming_distance;

                // Process the rest of the candidates
                for (; training_index < last_training_index; ++training_index) {
                	if (removed_ && removed_points_.test(*training_index)) continue;
                    // Compute the Hamming distance

					count_calculate_distance_++;

                    hamming_distance = distance_(vec, points_[*training_index], veclen_);
                    result.addPoint(hamming_distance, *training_index);
                }
            }
        }
		*/
    }


    void swap(LshIndex& other)
    {
    	BaseClass::swap(other);
    	std::swap(tables_, other.tables_);
    	std::swap(size_at_build_, other.size_at_build_);
    	std::swap(table_number_, other.table_number_);
    	std::swap(key_size_, other.key_size_);
    	std::swap(multi_probe_level_, other.multi_probe_level_);
    	std::swap(xor_masks_, other.xor_masks_);
    }

    /** The different hash tables */
    std::vector<lsh::LshTable<ElementType> > tables_;
    
    /** table number */
    unsigned int table_number_;
    /** key size */
    unsigned int key_size_;
    /** How far should we look for neighbors in multi-probe LSH */
    unsigned int multi_probe_level_;

    /** The XOR masks to apply to a key to get the neighboring buckets */
	std::vector<lsh::BucketKey> xor_masks_;

    //USING_BASECLASS_SYMBOLS
};
}

#endif //FLANN_LSH_INDEX_H_
