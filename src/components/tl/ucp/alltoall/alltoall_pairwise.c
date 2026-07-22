/**
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "alltoall.h"
#include "core/ucc_progress_queue.h"
#include "utils/ucc_math.h"
#include "utils/ucc_time.h"
#include "tl_ucp_sendrecv.h"

/* TODO: add as parameters */
#define MSG_MEDIUM 66000
#define NP_THRESH 32

static inline ucc_rank_t get_recv_peer(ucc_rank_t rank, ucc_rank_t size,
                                       ucc_rank_t step)
{
    return (rank + step) % size;
}

static inline ucc_rank_t get_send_peer(ucc_rank_t rank, ucc_rank_t size,
                                       ucc_rank_t step)
{
    return (rank - step + size) % size;
}

/*
 * Return rank's peer in a round-robin matching. Unlike the ring shifts above,
 * every non-self peer in a matching round exchanges data in both directions.
 * For an odd team, the standard virtual participant represents the local self
 * exchange, so every step still has exactly one peer per rank.
 *
 * This is the small scheduling experiment inspired by Continuous-Phase
 * AllReduce: NUM_POSTS is the number of matching rounds that may be locally
 * outstanding. It deliberately keeps the existing two-sided UCP transport and
 * completion accounting; it does not claim to provide remote credits.
 */
static inline ucc_rank_t get_matching_peer(ucc_rank_t rank, ucc_rank_t size,
                                           ucc_rank_t step)
{
    ucc_rank_t circle_size, rotating_size, round, delta, peer;

    if ((size & 1) == 0) {
        /* Keep self last so a shallow window starts with a remote matching. */
        if (step == size - 1) {
            return rank;
        }
        circle_size = size;
        round       = step;
    } else {
        /* The virtual participant supplies one self exchange per round. */
        circle_size = size + 1;
        round       = step;
    }

    rotating_size = circle_size - 1;
    if (rank == rotating_size) {
        peer = round;
    } else {
        delta = (rank + rotating_size - round) % rotating_size;
        peer  = (delta == 0) ? rotating_size :
                               (round + rotating_size - delta) % rotating_size;
    }

    return (peer == size) ? rank : peer;
}

static inline ucc_rank_t
get_peer(const ucc_tl_ucp_team_t *team, ucc_rank_t rank, ucc_rank_t size,
         ucc_rank_t step, int is_send)
{
    if (UCC_TL_UCP_TEAM_LIB(team)->cfg.alltoall_pairwise_schedule ==
        UCC_TL_UCP_ALLTOALL_PAIRWISE_SCHEDULE_MATCHING) {
        return get_matching_peer(rank, size, step);
    }
    return is_send ? get_send_peer(rank, size, step) :
                     get_recv_peer(rank, size, step);
}

static ucc_rank_t get_num_posts(const ucc_tl_ucp_team_t *team,
                                const ucc_coll_args_t *args)
{
    unsigned long posts = UCC_TL_UCP_TEAM_LIB(team)->cfg.alltoall_pairwise_num_posts;
    ucc_rank_t    tsize = UCC_TL_TEAM_SIZE(team);
    size_t data_size;

    data_size = (size_t)args->src.info.count *
                ucc_dt_size(args->src.info.datatype);
    if (posts == UCC_ULUNITS_AUTO) {
        if ((data_size > MSG_MEDIUM) && (tsize > NP_THRESH)) {
            /* use pairwise algorithm */
            posts = 1;
        } else {
            /* use linear algorithm */
            posts = 0;
        }
    }

    posts = (posts > tsize || posts == 0) ? tsize: posts;
    return posts;
}

static uint8_t get_size_bin(size_t data_size)
{
    uint8_t size_bin = 0;

    while ((data_size >>= 1) &&
           (size_bin < UCC_TL_UCP_A2A_ADAPTIVE_SIZE_BINS - 1)) {
        size_bin++;
    }
    return size_bin;
}

static ucc_rank_t adaptive_get_num_posts(ucc_tl_ucp_team_t *team,
                                         ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_a2a_adaptive_state_t *state;
    ucc_rank_t seed, size;
    size_t peer_size;
    uint8_t size_bin;

    size = UCC_TL_TEAM_SIZE(team);
    seed = get_num_posts(team, &TASK_ARGS(task));
    peer_size = (size_t)(TASK_ARGS(task).src.info.count / size) *
                ucc_dt_size(TASK_ARGS(task).src.info.datatype);
    size_bin = get_size_bin(peer_size);
    state = &team->a2a_adaptive[size_bin];

    if (state->best_posts == 0) {
        state->best_posts  = seed;
        state->trial_posts = seed;
    }

    task->alltoall_pairwise.size_bin  = size_bin;
    task->alltoall_pairwise.num_posts = state->trial_posts;
    task->alltoall_pairwise.sync_needed = !state->converged;
    return state->trial_posts;
}

static void adaptive_record(ucc_tl_ucp_team_t *team,
                            ucc_tl_ucp_task_t *task, double elapsed)
{
    ucc_tl_ucp_a2a_adaptive_state_t *state;
    ucc_tl_ucp_lib_config_t *cfg = &UCC_TL_UCP_TEAM_LIB(team)->cfg;
    ucc_rank_t size = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t next;
    uint32_t nsamples;
    double average, keep_limit;

    if (!task->alltoall_pairwise.enabled ||
        task->alltoall_pairwise.recorded) {
        return;
    }
    task->alltoall_pairwise.recorded = 1;
    state = &team->a2a_adaptive[task->alltoall_pairwise.size_bin];
    if (state->converged) {
        return;
    }
    if (!state->primed) {
        /* Do not attribute lazy endpoint/protocol setup to the seed depth. */
        state->primed = 1;
        return;
    }
    state->time_sum += elapsed;
    state->samples++;
    nsamples = ucc_max(1, cfg->alltoall_pairwise_adaptive_num_samples);
    if (state->samples < nsamples) {
        return;
    }

    average = state->time_sum / state->samples;
    state->samples  = 0;
    state->time_sum = 0;

    if (state->trial_posts == state->best_posts && state->best_time == 0) {
        state->best_time = average;
    } else {
        keep_limit = state->best_time *
                     (1.0 - ucc_min(100,
                                    cfg->alltoall_pairwise_adaptive_min_gain) /
                                100.0);
        if (average < keep_limit) {
            state->best_posts = state->trial_posts;
            state->best_time  = average;
        } else if ((state->upper_posts == 0) ||
                   (state->trial_posts < state->upper_posts)) {
            state->upper_posts = state->trial_posts;
        }
    }

    if (state->upper_posts > state->best_posts + 1) {
        next = state->best_posts +
               (state->upper_posts - state->best_posts) / 2;
    } else if (state->upper_posts != 0) {
        next = state->best_posts;
    } else {
        next = ucc_min(size, 2 * state->best_posts);
    }
    state->trial_posts = next;
    state->converged   = (next == state->best_posts);

    if (state->converged) {
        tl_info(UCC_TL_UCP_TEAM_LIB(team),
                "adaptive alltoall converged rank %u peer_bin %u posts %u "
                "%.3f us",
                UCC_TL_TEAM_RANK(team), task->alltoall_pairwise.size_bin,
                state->best_posts, state->best_time * 1e6);
    } else if (UCC_TL_TEAM_RANK(team) == 0) {
        tl_info(UCC_TL_UCP_TEAM_LIB(team),
                "adaptive alltoall peer_bin %u measured posts %u %.3f us; "
                "best %u %.3f us, next %u, upper %u",
                task->alltoall_pairwise.size_bin,
                task->alltoall_pairwise.num_posts, average * 1e6,
                state->best_posts, state->best_time * 1e6,
                state->trial_posts, state->upper_posts);
    }
}

void ucc_tl_ucp_alltoall_pairwise_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task  = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team  = TASK_TEAM(task);
    ptrdiff_t          sbuf  = (ptrdiff_t)TASK_ARGS(task).src.info.buffer;
    ptrdiff_t          rbuf  = (ptrdiff_t)TASK_ARGS(task).dst.info.buffer;
    ucc_memory_type_t  smem  = TASK_ARGS(task).src.info.mem_type;
    ucc_memory_type_t  rmem  = TASK_ARGS(task).dst.info.mem_type;
    ucc_rank_t         grank = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         gsize = UCC_TL_TEAM_SIZE(team);
    int                polls = 0;
    ucc_rank_t         peer, nreqs;
    size_t             data_size;

    nreqs = task->alltoall_pairwise.enabled ?
            task->alltoall_pairwise.num_posts :
            get_num_posts(team, &TASK_ARGS(task));
    data_size = (size_t)(TASK_ARGS(task).src.info.count / gsize) *
                ucc_dt_size(TASK_ARGS(task).src.info.datatype);
    if (nreqs > 1) {
        task->flags |= UCC_TL_UCP_TASK_FLAG_MULTI_SEND;
    } else {
        task->flags &= ~UCC_TL_UCP_TASK_FLAG_MULTI_SEND;
    }
    while ((task->tagged.send_posted < gsize ||
            task->tagged.recv_posted < gsize) &&
           (polls++ < task->n_polls)) {
        ucp_worker_progress(UCC_TL_UCP_TEAM_CTX(team)->worker.ucp_worker);
        while ((task->tagged.recv_posted < gsize) &&
               ((task->tagged.recv_posted - task->tagged.recv_completed) <
                nreqs)) {
            peer = get_peer(team, grank, gsize, task->tagged.recv_posted, 0);
            UCPCHECK_GOTO(ucc_tl_ucp_recv_nb((void *)(rbuf + peer * data_size),
                                             data_size, rmem, peer, team, task),
                          task, out);
            polls = 0;
        }
        while ((task->tagged.send_posted < gsize) &&
               ((task->tagged.send_posted - task->tagged.send_completed) <
                nreqs)) {
            peer = get_peer(team, grank, gsize, task->tagged.send_posted, 1);
            UCPCHECK_GOTO(ucc_tl_ucp_send_nb((void *)(sbuf + peer * data_size),
                                             data_size, smem, peer, team, task),
                          task, out);
            polls = 0;
        }
    }
    if (task->alltoall_pairwise.sync_req != NULL) {
        task->super.status =
            ucc_collective_test(
                &task->alltoall_pairwise.sync_req->task->super);
        if (task->super.status == UCC_INPROGRESS) {
            return;
        }
        ucc_service_coll_finalize(task->alltoall_pairwise.sync_req);
        task->alltoall_pairwise.sync_req = NULL;
        if (task->super.status == UCC_OK) {
            adaptive_record(team, task,
                            task->alltoall_pairwise.max_time_ns / 1e9);
        }
        goto out;
    }

    if ((task->tagged.send_posted < gsize) ||
        (task->tagged.recv_posted < gsize)) {
        return;
    }

    task->super.status = ucc_tl_ucp_test(task);
    if ((task->super.status == UCC_OK) &&
        task->alltoall_pairwise.enabled &&
        task->alltoall_pairwise.sync_needed) {
        task->alltoall_pairwise.local_time_ns =
            (uint64_t)((ucc_get_time() -
                        task->alltoall_pairwise.start_time) * 1e9);
        task->alltoall_pairwise.max_time_ns = 0;
        task->super.status = ucc_service_allreduce(
            UCC_TL_CORE_TEAM(team),
            &task->alltoall_pairwise.local_time_ns,
            &task->alltoall_pairwise.max_time_ns, UCC_DT_UINT64, 1,
            UCC_OP_MAX, task->subset,
            &task->alltoall_pairwise.sync_req);
        if (task->super.status == UCC_OK) {
            task->super.status = UCC_INPROGRESS;
            return;
        }
    }
out:
    if (task->super.status != UCC_INPROGRESS) {
        if (!task->alltoall_pairwise.sync_needed) {
            adaptive_record(team, task, 0);
        }
        UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task,
                                         "ucp_alltoall_pairwise_done", 0);
    }
}

ucc_status_t ucc_tl_ucp_alltoall_pairwise_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_alltoall_pairwise_start", 0);
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);
    task->alltoall_pairwise.enabled =
        UCC_TL_UCP_TEAM_LIB(team)->cfg.alltoall_pairwise_adaptive;
    task->alltoall_pairwise.recorded = 0;
    task->alltoall_pairwise.sync_needed = 0;
    task->alltoall_pairwise.sync_req = NULL;
    if (task->alltoall_pairwise.enabled) {
        adaptive_get_num_posts(team, task);
        task->alltoall_pairwise.start_time = ucc_get_time();
    }

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_ucp_alltoall_pairwise_init_common(ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);
    ucc_coll_args_t   *args = &TASK_ARGS(task);
    size_t data_size;

    task->super.post     = ucc_tl_ucp_alltoall_pairwise_start;
    task->super.progress = ucc_tl_ucp_alltoall_pairwise_progress;

    task->n_polls = ucc_max(1, task->n_polls);
    if (UCC_TL_UCP_TEAM_CTX(team)->cfg.pre_reg_mem) {
        data_size =
            (size_t)args->src.info.count * ucc_dt_size(args->src.info.datatype);
        ucc_tl_ucp_pre_register_mem(team, args->src.info.buffer, data_size,
                                    args->src.info.mem_type);
        ucc_tl_ucp_pre_register_mem(team, args->dst.info.buffer, data_size,
                                    args->dst.info.mem_type);
    }

    return UCC_OK;
}
