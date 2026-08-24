/**
 * Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "alltoall.h"
#include "alltoall_pairwise_schedule.h"
#include "core/ucc_progress_queue.h"
#include "utils/ucc_math.h"
#include "utils/ucc_malloc.h"
#include "tl_ucp_sendrecv.h"

#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

/* TODO: add as parameters */
#define MSG_MEDIUM 66000
#define NP_THRESH 32

typedef enum {
    UCC_TL_UCP_A2A_TRACE_START,
    UCC_TL_UCP_A2A_TRACE_RECV_POST,
    UCC_TL_UCP_A2A_TRACE_SEND_POST,
    UCC_TL_UCP_A2A_TRACE_PROGRESS,
    UCC_TL_UCP_A2A_TRACE_DONE
} ucc_tl_ucp_a2a_trace_event_type_t;

typedef struct {
    uint64_t ns;
    uint32_t send_posted;
    uint32_t send_completed;
    uint32_t recv_posted;
    uint32_t recv_completed;
    uint32_t step;
    uint32_t peer;
    uint64_t task_progress_calls;
    uint64_t worker_progress_calls;
    uint8_t  type;
} ucc_tl_ucp_a2a_trace_event_t;

typedef struct {
    ucc_tl_ucp_a2a_trace_event_t *events;
    uint32_t                      count;
    uint32_t                      capacity;
    uint32_t                      sequence;
    int                           overflow;
    uint64_t                      task_progress_calls;
    uint64_t                      worker_progress_calls;
} ucc_tl_ucp_a2a_trace_state_t;

#define UCC_TL_UCP_A2A_TRACE_STATE(_task)                                     \
    ((ucc_tl_ucp_a2a_trace_state_t *)(_task)->plugin_data)

static uint64_t ucc_tl_ucp_a2a_trace_now_ns(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ((uint64_t)ts.tv_sec * 1000000000ull) + (uint64_t)ts.tv_nsec;
}

static void
ucc_tl_ucp_a2a_trace_record(ucc_tl_ucp_task_t *task,
                            ucc_tl_ucp_a2a_trace_event_type_t type,
                            uint32_t step, uint32_t peer)
{
    ucc_tl_ucp_a2a_trace_state_t *state = UCC_TL_UCP_A2A_TRACE_STATE(task);
    ucc_tl_ucp_a2a_trace_event_t *event;

    if (!state->events) {
        return;
    }
    if (state->count == state->capacity) {
        state->overflow = 1;
        return;
    }

    event                 = &state->events[state->count++];
    event->ns             = ucc_tl_ucp_a2a_trace_now_ns();
    event->send_posted    = task->tagged.send_posted;
    event->send_completed = task->tagged.send_completed;
    event->recv_posted    = task->tagged.recv_posted;
    event->recv_completed = task->tagged.recv_completed;
    event->step           = step;
    event->peer           = peer;
    event->task_progress_calls   = state->task_progress_calls;
    event->worker_progress_calls = state->worker_progress_calls;
    event->type           = type;
}

static void ucc_tl_ucp_a2a_trace_progress_change(
    ucc_tl_ucp_task_t *task, uint32_t send_completed,
    uint32_t recv_completed)
{
    if (task->tagged.send_completed != send_completed ||
        task->tagged.recv_completed != recv_completed) {
        ucc_tl_ucp_a2a_trace_record(task, UCC_TL_UCP_A2A_TRACE_PROGRESS,
                                    UINT32_MAX, UINT32_MAX);
    }
}

static unsigned
ucc_tl_ucp_a2a_trace_worker_progress(ucc_tl_ucp_task_t *task,
                                     ucc_tl_ucp_team_t *team)
{
    ucc_tl_ucp_a2a_trace_state_t *state = UCC_TL_UCP_A2A_TRACE_STATE(task);

    if (state->events) {
        state->worker_progress_calls++;
    }
    return ucp_worker_progress(
        UCC_TL_UCP_TEAM_CTX(team)->worker.ucp_worker);
}

static ucc_status_t
ucc_tl_ucp_a2a_trace_test(ucc_tl_ucp_task_t *task,
                          ucc_tl_ucp_team_t *team)
{
    int polls = 0;

    if (UCC_TL_UCP_TASK_P2P_COMPLETE(task)) {
        return UCC_OK;
    }
    while (polls++ < task->n_polls) {
        if (UCC_TL_UCP_TASK_P2P_COMPLETE(task)) {
            return UCC_OK;
        }
        ucc_tl_ucp_a2a_trace_worker_progress(task, team);
    }
    return UCC_INPROGRESS;
}

static const char *ucc_tl_ucp_a2a_trace_event_name(uint8_t type)
{
    static const char *names[] = {"start", "recv_post", "send_post",
                                  "progress", "done"};

    return type < sizeof(names) / sizeof(names[0]) ? names[type] : "unknown";
}

static ucc_status_t
ucc_tl_ucp_alltoall_pairwise_trace_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);
    ucc_tl_ucp_a2a_trace_state_t *state = UCC_TL_UCP_A2A_TRACE_STATE(task);
    const char *schedule =
        team->cfg.alltoall_pairwise_schedule ==
                UCC_TL_UCP_ALLTOALL_PAIRWISE_SCHEDULE_RING_TOPOLOGY
            ? "ring_topology"
            : "ring";
    ucc_rank_t rank = UCC_TL_TEAM_RANK(team);
    char trace_path[PATH_MAX];
    FILE *output = stdout;
    uint32_t i;

    if (team->cfg.alltoall_pairwise_trace_dir &&
        team->cfg.alltoall_pairwise_trace_dir[0] != '\0') {
        snprintf(trace_path, sizeof(trace_path), "%s/rank_%06u.log",
                 team->cfg.alltoall_pairwise_trace_dir, rank);
        output = fopen(trace_path, "a");
        if (!output) {
            tl_warn(UCC_TL_UCP_TEAM_LIB(team),
                    "failed to open alltoall trace file %s; using stdout",
                    trace_path);
            output = stdout;
        }
    }

    for (i = 0; i < state->count; i++) {
        ucc_tl_ucp_a2a_trace_event_t *event = &state->events[i];
        fprintf(output,
                "A2A_PHASE_TRACE rank=%u schedule=%s sequence=%u event=%s "
                "ns=%" PRIu64 " step=%u peer=%u sp=%u sc=%u rp=%u rc=%u "
                "tpc=%" PRIu64 " wpc=%" PRIu64 "\n",
                rank, schedule, state->sequence,
                ucc_tl_ucp_a2a_trace_event_name(event->type), event->ns,
                event->step, event->peer, event->send_posted,
                event->send_completed, event->recv_posted,
                event->recv_completed, event->task_progress_calls,
                event->worker_progress_calls);
    }
    fprintf(output,
            "A2A_PHASE_TRACE_END rank=%u schedule=%s sequence=%u events=%u "
            "overflow=%d task_progress_calls=%" PRIu64 " "
            "worker_progress_calls=%" PRIu64 "\n",
            rank, schedule, state->sequence, state->count, state->overflow,
            state->task_progress_calls, state->worker_progress_calls);
    if (output == stdout) {
        fflush(output);
    } else {
        fclose(output);
    }
    ucc_free(state->events);
    state->events = NULL;
    return ucc_tl_ucp_coll_finalize(coll_task);
}

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

static inline ucc_rank_t
get_peer(const ucc_tl_ucp_team_t *team, ucc_rank_t rank, ucc_rank_t size,
         ucc_rank_t step, int is_send)
{
    if (team->cfg.alltoall_pairwise_schedule ==
            UCC_TL_UCP_ALLTOALL_PAIRWISE_SCHEDULE_RING_TOPOLOGY &&
        team->alltoall_topo_ring.enabled) {
        return ucc_tl_ucp_alltoall_topo_ring_peer(
            team->alltoall_topo_ring.rank_order,
            team->alltoall_topo_ring.rank_labels, rank, size, step, is_send);
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

    ucc_tl_ucp_a2a_trace_state_t *trace_state =
        UCC_TL_UCP_A2A_TRACE_STATE(task);
    if (trace_state->events) {
        trace_state->task_progress_calls++;
    }
    nreqs     = get_num_posts(team, &TASK_ARGS(task));
    data_size = (size_t)(TASK_ARGS(task).src.info.count / gsize) *
                ucc_dt_size(TASK_ARGS(task).src.info.datatype);
    while ((task->tagged.send_posted < gsize ||
            task->tagged.recv_posted < gsize) &&
           (polls++ < task->n_polls)) {
        uint32_t send_completed = task->tagged.send_completed;
        uint32_t recv_completed = task->tagged.recv_completed;
        ucc_tl_ucp_a2a_trace_worker_progress(task, team);
        ucc_tl_ucp_a2a_trace_progress_change(task, send_completed,
                                             recv_completed);
        while ((task->tagged.recv_posted < gsize) &&
               ((task->tagged.recv_posted - task->tagged.recv_completed) <
                nreqs)) {
            uint32_t step = task->tagged.recv_posted;
            peer = get_peer(team, grank, gsize, step, 0);
            UCPCHECK_GOTO(ucc_tl_ucp_recv_nb((void *)(rbuf + peer * data_size),
                                             data_size, rmem, peer, team, task),
                          task, out);
            ucc_tl_ucp_a2a_trace_record(task,
                                        UCC_TL_UCP_A2A_TRACE_RECV_POST,
                                        step, peer);
            polls = 0;
        }
        while ((task->tagged.send_posted < gsize) &&
               ((task->tagged.send_posted - task->tagged.send_completed) <
                nreqs)) {
            uint32_t step = task->tagged.send_posted;
            peer = get_peer(team, grank, gsize, step, 1);
            UCPCHECK_GOTO(ucc_tl_ucp_send_nb((void *)(sbuf + peer * data_size),
                                             data_size, smem, peer, team, task),
                          task, out);
            ucc_tl_ucp_a2a_trace_record(task,
                                        UCC_TL_UCP_A2A_TRACE_SEND_POST,
                                        step, peer);
            polls = 0;
        }
    }
    if ((task->tagged.send_posted < gsize) ||
        (task->tagged.recv_posted < gsize)) {
        return;
    }

    {
        uint32_t send_completed = task->tagged.send_completed;
        uint32_t recv_completed = task->tagged.recv_completed;
        task->super.status = ucc_tl_ucp_a2a_trace_test(task, team);
        ucc_tl_ucp_a2a_trace_progress_change(task, send_completed,
                                             recv_completed);
    }
out:
    if (task->super.status != UCC_INPROGRESS) {
        ucc_tl_ucp_a2a_trace_record(task, UCC_TL_UCP_A2A_TRACE_DONE,
                                    UINT32_MAX, UINT32_MAX);
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
    ucc_tl_ucp_a2a_trace_record(task, UCC_TL_UCP_A2A_TRACE_START,
                                UINT32_MAX, UINT32_MAX);

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_ucp_alltoall_pairwise_init_common(ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);
    ucc_coll_args_t   *args = &TASK_ARGS(task);
    size_t data_size;

    task->super.post     = ucc_tl_ucp_alltoall_pairwise_start;
    task->super.progress = ucc_tl_ucp_alltoall_pairwise_progress;

    {
        ucc_tl_ucp_a2a_trace_state_t *state = UCC_TL_UCP_A2A_TRACE_STATE(task);
        uint32_t sequence = team->alltoall_pairwise_trace_sequence++;

        memset(state, 0, sizeof(*state));
        state->sequence = sequence;
        if (team->cfg.alltoall_pairwise_trace &&
            sequence >= team->cfg.alltoall_pairwise_trace_skip &&
            sequence - team->cfg.alltoall_pairwise_trace_skip <
                team->cfg.alltoall_pairwise_trace_count) {
            state->capacity = 6 * UCC_TL_TEAM_SIZE(team) + 16;
            state->events = ucc_calloc(state->capacity,
                                       sizeof(*state->events),
                                       "alltoall_pairwise_trace");
            if (state->events) {
                task->super.finalize =
                    ucc_tl_ucp_alltoall_pairwise_trace_finalize;
            }
        }
    }

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
