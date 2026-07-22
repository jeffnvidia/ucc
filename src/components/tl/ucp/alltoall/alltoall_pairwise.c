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
    ucc_rank_t size;
    size_t peer_size;
    uint8_t size_bin;

    size = UCC_TL_TEAM_SIZE(team);
    peer_size = (size_t)(TASK_ARGS(task).src.info.count / size) *
                ucc_dt_size(TASK_ARGS(task).src.info.datatype);
    size_bin = get_size_bin(peer_size);
    state = &team->a2a_adaptive[size_bin];

    task->alltoall_pairwise.size_bin = size_bin;
    if (peer_size == 0) {
        state->posts = get_num_posts(team, &TASK_ARGS(task));
        task->alltoall_pairwise.num_posts = state->posts;
        task->alltoall_pairwise.bootstrapping = 0;
        return state->posts;
    }
    if (state->posts != 0) {
        task->alltoall_pairwise.num_posts = state->posts;
        task->alltoall_pairwise.bootstrapping = 0;
        return state->posts;
    }
    if (!state->primed) {
        /* Each round uses a different endpoint. Prime all peers once before
         * measuring steady-state request lifetime on the next collective.
         */
        task->alltoall_pairwise.num_posts = 1;
        task->alltoall_pairwise.priming = 1;
        task->alltoall_pairwise.bootstrapping = 0;
        return 1;
    }

    /* Leave one cold round, timed rounds, and one round after switching. */
    if (UCC_TL_UCP_TEAM_LIB(team)->cfg.
            alltoall_pairwise_adaptive_profile_rounds) {
        task->alltoall_pairwise.num_samples =
            ucc_min((size > 2) ? size - 2 : 0,
                    UCC_TL_UCP_A2A_PROFILE_MAX_ROUNDS);
    } else {
        task->alltoall_pairwise.num_samples =
            ucc_min(ucc_max(1, UCC_TL_UCP_TEAM_LIB(team)->cfg.
                                      alltoall_pairwise_adaptive_num_samples),
                    (size > 2) ? size - 2 : 0);
    }
    task->alltoall_pairwise.num_posts = 1;
    task->alltoall_pairwise.bootstrapping =
        task->alltoall_pairwise.num_samples != 0;
    if (!task->alltoall_pairwise.bootstrapping) {
        state->posts = get_num_posts(team, &TASK_ARGS(task));
        task->alltoall_pairwise.num_posts = state->posts;
    }
    return task->alltoall_pairwise.num_posts;
}

static ucc_status_t adaptive_get_bandwidth(ucc_tl_ucp_team_t *team,
                                           size_t peer_size,
                                           double *bandwidth)
{
    ucp_ep_evaluate_perf_param_t param;
    ucp_ep_evaluate_perf_attr_t attr1, attr2;
    ucp_ep_h ep;
    ucc_rank_t peer;
    ucc_status_t ucc_status;
    ucs_status_t status;
    double delta;

    for (peer = 0; peer < UCC_TL_TEAM_SIZE(team); peer++) {
        ucc_rank_t candidate =
            get_peer(team, UCC_TL_TEAM_RANK(team), UCC_TL_TEAM_SIZE(team),
                     peer, 1);
        if (candidate != UCC_TL_TEAM_RANK(team)) {
            peer = candidate;
            break;
        }
    }
    if (peer == UCC_TL_TEAM_SIZE(team)) {
        return UCC_ERR_NOT_SUPPORTED;
    }
    ucc_status = ucc_tl_ucp_get_ep(team, peer, &ep);
    if (ucc_status != UCC_OK) {
        return ucc_status;
    }

    param.field_mask = UCP_EP_PERF_PARAM_FIELD_MESSAGE_SIZE;
    attr1.field_mask = UCP_EP_PERF_ATTR_FIELD_ESTIMATED_TIME;
    attr2.field_mask = UCP_EP_PERF_ATTR_FIELD_ESTIMATED_TIME;
    param.message_size = peer_size;
    status = ucp_ep_evaluate_perf(ep, &param, &attr1);
    if (status != UCS_OK) {
        return ucs_status_to_ucc_status(status);
    }
    param.message_size = 2 * peer_size;
    status = ucp_ep_evaluate_perf(ep, &param, &attr2);
    if (status != UCS_OK) {
        return ucs_status_to_ucc_status(status);
    }

    delta = attr2.estimated_time - attr1.estimated_time;
    if (delta <= 0) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    /* The slope of the non-self endpoint estimate is the directional
     * interface bandwidth used to inject one peer payload.
     */
    *bandwidth = peer_size / delta;
    return UCC_OK;
}

static ucc_status_t adaptive_bootstrap_sample(ucc_tl_ucp_team_t *team,
                                              ucc_tl_ucp_task_t *task,
                                              size_t peer_size)
{
    ucc_rank_t completed, posts;
    uint32_t   completed_samples, new_samples, i;
    ucc_status_t status;
    double elapsed, fill, now, interval;

    if (!task->alltoall_pairwise.bootstrapping) {
        return UCC_OK;
    }

    completed = ucc_min(task->tagged.send_completed,
                        task->tagged.recv_completed);
    if (!task->alltoall_pairwise.timing_started) {
        if (completed < 1) {
            return UCC_OK;
        }
        /* The first matching round initializes endpoints and UCX protocols. */
        task->alltoall_pairwise.sample_start_completed = completed;
        task->alltoall_pairwise.start_time = ucc_get_time();
        task->alltoall_pairwise.last_sample_time =
            task->alltoall_pairwise.start_time;
        task->alltoall_pairwise.timing_started = 1;
        return UCC_OK;
    }

    completed_samples =
        completed - task->alltoall_pairwise.sample_start_completed;
    if (completed_samples > task->alltoall_pairwise.sample_count) {
        now = ucc_get_time();
        new_samples = completed_samples -
                      task->alltoall_pairwise.sample_count;
        interval = (now - task->alltoall_pairwise.last_sample_time) /
                   new_samples;
        for (i = 0; i < new_samples &&
                    task->alltoall_pairwise.sample_count <
                        UCC_TL_UCP_A2A_PROFILE_MAX_ROUNDS; i++) {
            task->alltoall_pairwise.round_ns[
                task->alltoall_pairwise.sample_count++] =
                (uint64_t)(interval * 1e9);
        }
        task->alltoall_pairwise.last_sample_time = now;
    }

    if (completed_samples < task->alltoall_pairwise.num_samples) {
        return UCC_OK;
    }

    elapsed = (ucc_get_time() - task->alltoall_pairwise.start_time) /
              task->alltoall_pairwise.num_samples;
    status = adaptive_get_bandwidth(team, peer_size,
                                    &task->alltoall_pairwise.bandwidth);
    if (status == UCC_OK) {
        fill = task->alltoall_pairwise.bandwidth * elapsed / peer_size;
        posts = (ucc_rank_t)fill;
        if ((double)posts < fill) {
            posts++;
        }
        posts = ucc_max(1, ucc_min(UCC_TL_TEAM_SIZE(team), posts));
    } else {
        /* Preserve a usable fallback if an older UCX cannot evaluate the EP. */
        posts = get_num_posts(team, &TASK_ARGS(task));
        task->alltoall_pairwise.bandwidth = 0;
    }

    task->alltoall_pairwise.local_metrics[0] = (uint64_t)(elapsed * 1e9);
    task->alltoall_pairwise.local_metrics[1] = posts;
    task->alltoall_pairwise.global_metrics[0] = 0;
    task->alltoall_pairwise.global_metrics[1] = 0;
    if (UCC_TL_UCP_TEAM_LIB(team)->cfg.
            alltoall_pairwise_adaptive_profile_rounds) {
        char   rounds[4096];
        size_t offset = 0;

        rounds[0] = '\0';
        for (i = 0; i < task->alltoall_pairwise.sample_count; i++) {
            int written = ucc_snprintf_safe(
                rounds + offset, sizeof(rounds) - offset, "%s%lu",
                (i == 0) ? "" : ",",
                (unsigned long)task->alltoall_pairwise.round_ns[i]);
            if ((written < 0) || ((size_t)written >= sizeof(rounds) - offset)) {
                break;
            }
            offset += written;
        }
        tl_info(UCC_TL_UCP_TEAM_LIB(team),
                "adaptive alltoall rounds rank %u team %u peer_bytes %lu "
                "samples %u round_ns %s",
                UCC_TL_TEAM_RANK(team), UCC_TL_TEAM_SIZE(team),
                (unsigned long)peer_size,
                task->alltoall_pairwise.sample_count, rounds);
    }
    status = ucc_service_allreduce(
        UCC_TL_CORE_TEAM(team), task->alltoall_pairwise.local_metrics,
        task->alltoall_pairwise.global_metrics, UCC_DT_UINT64, 2, UCC_OP_MAX,
        task->subset, &task->alltoall_pairwise.sync_req);
    return (status == UCC_OK) ? UCC_INPROGRESS : status;
}

static unsigned alltoall_pairwise_worker_progress(ucc_tl_ucp_team_t *team,
                                                  ucc_tl_ucp_task_t *task)
{
    uint32_t send_before, recv_before, send_after, recv_after;
    uint32_t send_burst, recv_burst, imbalance;
    uint64_t elapsed_ns;
    double   start, stop, since_last_ns;
    unsigned count;

    if (!task->alltoall_pairwise.progress_profile) {
        return ucp_worker_progress(UCC_TL_UCP_TEAM_CTX(team)->worker.ucp_worker);
    }

    send_before = task->tagged.send_completed;
    recv_before = task->tagged.recv_completed;
    start = ucc_get_time();
    since_last_ns = (start - task->alltoall_pairwise.occupancy_sample_time) *
                    1e9;
    task->alltoall_pairwise.send_outstanding_ns += since_last_ns *
        (task->tagged.send_posted - send_before);
    task->alltoall_pairwise.recv_outstanding_ns += since_last_ns *
        (task->tagged.recv_posted - recv_before);
    count = ucp_worker_progress(UCC_TL_UCP_TEAM_CTX(team)->worker.ucp_worker);
    stop = ucc_get_time();
    elapsed_ns = (uint64_t)((stop - start) * 1e9);
    send_after = task->tagged.send_completed;
    recv_after = task->tagged.recv_completed;
    send_burst = send_after - send_before;
    recv_burst = recv_after - recv_before;
    /* Completion time inside ucp_worker_progress is not observable here.
     * The trapezoid is an unbiased approximation for the occupancy integral. */
    task->alltoall_pairwise.send_outstanding_ns += elapsed_ns *
        ((task->tagged.send_posted - send_before) +
         (task->tagged.send_posted - send_after)) / 2.0;
    task->alltoall_pairwise.recv_outstanding_ns += elapsed_ns *
        ((task->tagged.recv_posted - recv_before) +
         (task->tagged.recv_posted - recv_after)) / 2.0;
    task->alltoall_pairwise.occupancy_sample_time = stop;
    imbalance = (task->tagged.send_completed > task->tagged.recv_completed) ?
                task->tagged.send_completed - task->tagged.recv_completed :
                task->tagged.recv_completed - task->tagged.send_completed;

    task->alltoall_pairwise.progress_calls++;
    task->alltoall_pairwise.progress_ns += elapsed_ns;
    task->alltoall_pairwise.max_progress_ns =
        ucc_max(task->alltoall_pairwise.max_progress_ns, elapsed_ns);
    task->alltoall_pairwise.max_send_burst =
        ucc_max(task->alltoall_pairwise.max_send_burst, send_burst);
    task->alltoall_pairwise.max_recv_burst =
        ucc_max(task->alltoall_pairwise.max_recv_burst, recv_burst);
    task->alltoall_pairwise.max_completion_imbalance =
        ucc_max(task->alltoall_pairwise.max_completion_imbalance, imbalance);

    if ((send_burst > 0) &&
        (task->tagged.send_posted < UCC_TL_TEAM_SIZE(team)) &&
        (task->tagged.send_posted == task->tagged.send_completed)) {
        task->alltoall_pairwise.send_drain_events++;
        task->alltoall_pairwise.refill_pending = 1;
        task->alltoall_pairwise.progress_return_time = stop;
    }
    if ((recv_burst > 0) &&
        (task->tagged.recv_posted < UCC_TL_TEAM_SIZE(team)) &&
        (task->tagged.recv_posted == task->tagged.recv_completed)) {
        task->alltoall_pairwise.recv_drain_events++;
    }
    return count;
}

static ucc_status_t alltoall_pairwise_test(ucc_tl_ucp_team_t *team,
                                           ucc_tl_ucp_task_t *task)
{
    int polls = 0;

    if (UCC_TL_UCP_TASK_P2P_COMPLETE(task)) {
        return UCC_OK;
    }
    while (polls++ < task->n_polls) {
        if (UCC_TL_UCP_TASK_P2P_COMPLETE(task)) {
            return UCC_OK;
        }
        alltoall_pairwise_worker_progress(team, task);
    }
    return UCC_INPROGRESS;
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
    ucc_rank_t         peer, nreqs = 0;
    ucc_status_t       status;
    size_t             data_size;

    data_size = (size_t)(TASK_ARGS(task).src.info.count / gsize) *
                ucc_dt_size(TASK_ARGS(task).src.info.datatype);
    if (task->alltoall_pairwise.sync_req != NULL) {
        task->super.status =
            ucc_collective_test(
                &task->alltoall_pairwise.sync_req->task->super);
        if (task->super.status == UCC_INPROGRESS) {
            return;
        }
        ucc_service_coll_finalize(task->alltoall_pairwise.sync_req);
        task->alltoall_pairwise.sync_req = NULL;
        if (task->super.status != UCC_OK) {
            goto out;
        }
        team->a2a_adaptive[task->alltoall_pairwise.size_bin].posts =
            task->alltoall_pairwise.global_metrics[1];
        team->a2a_adaptive[task->alltoall_pairwise.size_bin].request_time =
            task->alltoall_pairwise.global_metrics[0] / 1e9;
        team->a2a_adaptive[task->alltoall_pairwise.size_bin].bandwidth =
            task->alltoall_pairwise.bandwidth;
        task->alltoall_pairwise.num_posts =
            team->a2a_adaptive[task->alltoall_pairwise.size_bin].posts;
        task->alltoall_pairwise.bootstrapping = 0;
        task->super.status = UCC_INPROGRESS;
        if (grank == 0) {
            tl_info(UCC_TL_UCP_TEAM_LIB(team),
                    "adaptive alltoall bootstrap peer_bin %u request %.3f us "
                    "ucx_bw %.3f GB/s posts %u",
                    task->alltoall_pairwise.size_bin,
                    team->a2a_adaptive[task->alltoall_pairwise.size_bin].
                        request_time * 1e6,
                    task->alltoall_pairwise.bandwidth / 1e9,
                    task->alltoall_pairwise.num_posts);
        }
    }

    nreqs = task->alltoall_pairwise.enabled ?
            task->alltoall_pairwise.num_posts :
            get_num_posts(team, &TASK_ARGS(task));
    if (nreqs > 1) {
        task->flags |= UCC_TL_UCP_TASK_FLAG_MULTI_SEND;
    } else {
        task->flags &= ~UCC_TL_UCP_TASK_FLAG_MULTI_SEND;
    }
    while ((task->tagged.send_posted < gsize ||
            task->tagged.recv_posted < gsize) &&
           (polls++ < task->n_polls)) {
        alltoall_pairwise_worker_progress(team, task);
        status = adaptive_bootstrap_sample(team, task, data_size);
        if (status == UCC_INPROGRESS) {
            return;
        } else if (status != UCC_OK) {
            task->super.status = status;
            goto out;
        }
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
        if (task->alltoall_pairwise.refill_pending) {
            uint64_t refill_ns = (uint64_t)(
                (ucc_get_time() -
                 task->alltoall_pairwise.progress_return_time) * 1e9);
            task->alltoall_pairwise.refill_ns += refill_ns;
            task->alltoall_pairwise.max_refill_ns =
                ucc_max(task->alltoall_pairwise.max_refill_ns, refill_ns);
            task->alltoall_pairwise.refill_pending = 0;
        }
    }

    if ((task->tagged.send_posted < gsize) ||
        (task->tagged.recv_posted < gsize)) {
        return;
    }

    task->super.status = alltoall_pairwise_test(team, task);
    if ((task->super.status == UCC_OK) &&
        task->alltoall_pairwise.priming) {
        team->a2a_adaptive[task->alltoall_pairwise.size_bin].primed = 1;
        if (grank == 0) {
            tl_info(UCC_TL_UCP_TEAM_LIB(team),
                    "adaptive alltoall primed peer_bin %u at posts 1",
                    task->alltoall_pairwise.size_bin);
        }
    }
out:
    if ((task->super.status != UCC_INPROGRESS) &&
        task->alltoall_pairwise.progress_profile &&
        !task->alltoall_pairwise.progress_profile_logged) {
        double wall_ns = (ucc_get_time() -
            task->alltoall_pairwise.progress_profile_start_time) * 1e9;
        double send_lifetime_ns = task->tagged.send_completed ?
            task->alltoall_pairwise.send_outstanding_ns /
                task->tagged.send_completed : 0.0;
        double recv_lifetime_ns = task->tagged.recv_completed ?
            task->alltoall_pairwise.recv_outstanding_ns /
                task->tagged.recv_completed : 0.0;

        tl_info(UCC_TL_UCP_TEAM_LIB(team),
                "alltoall progress profile rank %u team %u peer_bytes %lu "
                "posts %u wall_ns %lu calls %u progress_ns %lu "
                "max_progress_ns %lu "
                "max_send_burst %u max_recv_burst %u send_drains %u "
                "recv_drains %u refill_ns %lu max_refill_ns %lu "
                "max_completion_imbalance %u "
                "send_outstanding_ns %.0f recv_outstanding_ns %.0f "
                "send_lifetime_ns %.0f recv_lifetime_ns %.0f "
                "mean_send_outstanding %.3f mean_recv_outstanding %.3f",
                grank, gsize, (unsigned long)data_size, nreqs,
                (unsigned long)wall_ns,
                task->alltoall_pairwise.progress_calls,
                (unsigned long)task->alltoall_pairwise.progress_ns,
                (unsigned long)task->alltoall_pairwise.max_progress_ns,
                task->alltoall_pairwise.max_send_burst,
                task->alltoall_pairwise.max_recv_burst,
                task->alltoall_pairwise.send_drain_events,
                task->alltoall_pairwise.recv_drain_events,
                (unsigned long)task->alltoall_pairwise.refill_ns,
                (unsigned long)task->alltoall_pairwise.max_refill_ns,
                task->alltoall_pairwise.max_completion_imbalance,
                task->alltoall_pairwise.send_outstanding_ns,
                task->alltoall_pairwise.recv_outstanding_ns,
                send_lifetime_ns, recv_lifetime_ns,
                task->alltoall_pairwise.send_outstanding_ns / wall_ns,
                task->alltoall_pairwise.recv_outstanding_ns / wall_ns);
        task->alltoall_pairwise.progress_profile_logged = 1;
    }
    if (task->super.status != UCC_INPROGRESS) {
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
    task->alltoall_pairwise.priming = 0;
    task->alltoall_pairwise.bootstrapping = 0;
    task->alltoall_pairwise.timing_started = 0;
    task->alltoall_pairwise.progress_profile = 0;
    task->alltoall_pairwise.progress_profile_logged = 0;
    task->alltoall_pairwise.refill_pending = 0;
    task->alltoall_pairwise.num_samples = 0;
    task->alltoall_pairwise.sample_start_completed = 0;
    task->alltoall_pairwise.sample_count = 0;
    task->alltoall_pairwise.bandwidth = 0;
    task->alltoall_pairwise.progress_calls = 0;
    task->alltoall_pairwise.max_send_burst = 0;
    task->alltoall_pairwise.max_recv_burst = 0;
    task->alltoall_pairwise.send_drain_events = 0;
    task->alltoall_pairwise.recv_drain_events = 0;
    task->alltoall_pairwise.max_completion_imbalance = 0;
    task->alltoall_pairwise.progress_ns = 0;
    task->alltoall_pairwise.max_progress_ns = 0;
    task->alltoall_pairwise.refill_ns = 0;
    task->alltoall_pairwise.max_refill_ns = 0;
    task->alltoall_pairwise.send_outstanding_ns = 0;
    task->alltoall_pairwise.recv_outstanding_ns = 0;
    task->alltoall_pairwise.sync_req = NULL;
    if (UCC_TL_UCP_TEAM_LIB(team)->cfg.alltoall_pairwise_progress_profile) {
        size_t peer_size =
            (size_t)(TASK_ARGS(task).src.info.count / UCC_TL_TEAM_SIZE(team)) *
            ucc_dt_size(TASK_ARGS(task).src.info.datatype);
        ucc_tl_ucp_a2a_adaptive_state_t *state =
            &team->a2a_adaptive[get_size_bin(peer_size)];

        task->alltoall_pairwise.progress_profile =
            state->progress_profile_calls == 1;
        state->progress_profile_calls++;
        if (task->alltoall_pairwise.progress_profile) {
            task->alltoall_pairwise.progress_profile_start_time =
                ucc_get_time();
            task->alltoall_pairwise.occupancy_sample_time =
                task->alltoall_pairwise.progress_profile_start_time;
        }
    }
    if (task->alltoall_pairwise.enabled) {
        adaptive_get_num_posts(team, task);
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
