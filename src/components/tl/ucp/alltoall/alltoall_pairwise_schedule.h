/**
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef ALLTOALL_PAIRWISE_SCHEDULE_H_
#define ALLTOALL_PAIRWISE_SCHEDULE_H_

#include "utils/ucc_coll_utils.h"
#include <stdint.h>
#include <string.h>
#include <limits.h>

/*
 * Convert node endpoint maps into a node-interleaved ring order.
 *
 * node_maps contains one team-rank map per node, with the same number of ranks
 * on every node. rank_order maps a virtual ring label to a team rank, while
 * rank_labels provides the inverse mapping. UCC may optimize a node's backing
 * rank array into a strided map, so callers must evaluate the generic map
 * rather than depend on a materialized rank_map.
 */
static inline ucc_status_t
ucc_tl_ucp_alltoall_topo_ring_build_map(const ucc_ep_map_t *node_maps,
                                        ucc_rank_t size, ucc_rank_t nnodes,
                                        ucc_rank_t ppn,
                                        ucc_rank_t *rank_order,
                                        ucc_rank_t *rank_labels)
{
    ucc_rank_t label, local_rank, node, rank;

    if (!node_maps || !rank_order || !rank_labels || nnodes <= 1 || ppn <= 1 ||
        size / nnodes != ppn || size % nnodes != 0) {
        return UCC_ERR_INVALID_PARAM;
    }

    for (rank = 0; rank < size; rank++) {
        rank_labels[rank] = UCC_RANK_INVALID;
    }

    for (local_rank = 0; local_rank < ppn; local_rank++) {
        for (node = 0; node < nnodes; node++) {
            if (node_maps[node].ep_num != ppn) {
                return UCC_ERR_INVALID_PARAM;
            }
            rank  = ucc_ep_map_eval(node_maps[node], local_rank);
            label = local_rank * nnodes + node;
            if (rank >= size || rank_labels[rank] != UCC_RANK_INVALID) {
                return UCC_ERR_INVALID_PARAM;
            }
            rank_order[label] = rank;
            rank_labels[rank] = label;
        }
    }

    return UCC_OK;
}

static inline uint32_t ucc_tl_ucp_alltoall_topo_xorshift32(uint32_t *state)
{
    uint32_t value = *state;

    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    *state = value;
    return value;
}

static inline void ucc_tl_ucp_alltoall_topo_staggered_score(
    const ucc_rank_t *node_order, ucc_rank_t size, ucc_rank_t nnodes,
    ucc_rank_t ppn, ucc_rank_t *positions, ucc_rank_t *histogram,
    ucc_rank_t *peak, uint64_t *sum_squares)
{
    ucc_rank_t local_rank, node, pos, first, second, step;

    memset(histogram, 0, size * sizeof(*histogram));
    for (local_rank = 0; local_rank < ppn; local_rank++) {
        for (pos = 0; pos < nnodes; pos++) {
            node = node_order[local_rank * nnodes + pos];
            positions[node * ppn + local_rank] = local_rank * nnodes + pos;
        }
    }
    for (node = 0; node < nnodes; node++) {
        for (first = 0; first < ppn; first++) {
            for (second = 0; second < ppn; second++) {
                if (first == second) {
                    continue;
                }
                step = (positions[node * ppn + first] + size -
                        positions[node * ppn + second]) % size;
                histogram[step]++;
            }
        }
    }

    *peak        = 0;
    *sum_squares = 0;
    for (step = 1; step < size; step++) {
        *peak = ucc_max(*peak, histogram[step]);
        *sum_squares += (uint64_t)histogram[step] * histogram[step];
    }
}

/*
 * Build a topology-aware ring without globally synchronized local rounds.
 *
 * Each local-rank layer still contains every node exactly once, but the node
 * permutation differs between layers. A small deterministic search at team
 * creation selects the permutation set that minimizes, lexicographically,
 * the peak number of node-local pairs in one step and then the squared load
 * variation. The hot collective path remains the same one-array lookup as the
 * original topology ring.
 */
static inline ucc_status_t ucc_tl_ucp_alltoall_topo_staggered_build_map(
    const ucc_ep_map_t *node_maps, ucc_rank_t size, ucc_rank_t nnodes,
    ucc_rank_t ppn, ucc_rank_t *rank_order, ucc_rank_t *rank_labels)
{
    const uint32_t candidate_count = 4096;
    ucc_rank_t *positions = NULL, *histogram = NULL;
    ucc_rank_t candidate, local_rank, node, pos, swap_pos, label, rank;
    ucc_rank_t peak, best_peak = UCC_RANK_INVALID;
    uint64_t score, best_score = UINT64_MAX;
    uint32_t seed;
    ucc_status_t status = UCC_OK;

    if (!node_maps || !rank_order || !rank_labels || nnodes <= 1 || ppn <= 1 ||
        size / nnodes != ppn || size % nnodes != 0) {
        return UCC_ERR_INVALID_PARAM;
    }
    for (node = 0; node < nnodes; node++) {
        if (node_maps[node].ep_num != ppn) {
            return UCC_ERR_INVALID_PARAM;
        }
    }

    positions = ucc_malloc(size * sizeof(*positions),
                           "alltoall_topo_staggered_positions");
    histogram = ucc_malloc(size * sizeof(*histogram),
                           "alltoall_topo_staggered_histogram");
    if (!positions || !histogram) {
        status = UCC_ERR_NO_MEMORY;
        goto out;
    }

    for (candidate = 0; candidate < candidate_count; candidate++) {
        for (local_rank = 0; local_rank < ppn; local_rank++) {
            for (pos = 0; pos < nnodes; pos++) {
                rank_order[local_rank * nnodes + pos] = pos;
            }
            if (local_rank == 0) {
                continue;
            }
            seed = 0x9e3779b9u ^ (uint32_t)nnodes * 0x85ebca6bu ^
                   (uint32_t)ppn * 0xc2b2ae35u ^
                   (uint32_t)candidate * 0x27d4eb2du ^
                   (uint32_t)local_rank * 0x165667b1u;
            if (seed == 0) {
                seed = 0xa341316cu;
            }
            for (pos = nnodes - 1; pos > 0; pos--) {
                swap_pos = ucc_tl_ucp_alltoall_topo_xorshift32(&seed) %
                           (pos + 1);
                node = rank_order[local_rank * nnodes + pos];
                rank_order[local_rank * nnodes + pos] =
                    rank_order[local_rank * nnodes + swap_pos];
                rank_order[local_rank * nnodes + swap_pos] = node;
            }
        }
        ucc_tl_ucp_alltoall_topo_staggered_score(
            rank_order, size, nnodes, ppn, positions, histogram, &peak,
            &score);
        if (peak < best_peak || (peak == best_peak && score < best_score)) {
            best_peak  = peak;
            best_score = score;
            memcpy(rank_labels, rank_order, size * sizeof(*rank_labels));
        }
    }

    memcpy(positions, rank_labels, size * sizeof(*positions));
    for (rank = 0; rank < size; rank++) {
        rank_labels[rank] = UCC_RANK_INVALID;
    }
    for (label = 0; label < size; label++) {
        local_rank = label / nnodes;
        node       = positions[label];
        rank       = ucc_ep_map_eval(node_maps[node], local_rank);
        if (rank >= size || rank_labels[rank] != UCC_RANK_INVALID) {
            status = UCC_ERR_INVALID_PARAM;
            goto out;
        }
        rank_order[label] = rank;
        rank_labels[rank] = label;
    }

out:
    ucc_free(histogram);
    ucc_free(positions);
    return status;
}

static inline ucc_rank_t
ucc_tl_ucp_alltoall_topo_ring_peer(const ucc_rank_t *rank_order,
                                   const ucc_rank_t *rank_labels,
                                   ucc_rank_t rank, ucc_rank_t size,
                                   ucc_rank_t step, int is_send)
{
    ucc_rank_t label = rank_labels[rank];
    ucc_rank_t peer_label;

    peer_label = is_send ? (label - step + size) % size :
                           (label + step) % size;
    return rank_order[peer_label];
}

#endif
