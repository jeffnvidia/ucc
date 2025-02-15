/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef TL_MLX5_MCAST_HELPER_H_
#define TL_MLX5_MCAST_HELPER_H_
#include "tl_mlx5_mcast_progress.h"
#include "utils/ucc_math.h"
#include "tl_mlx5.h"

static inline ucc_status_t ucc_tl_mlx5_mcast_poll_send(ucc_tl_mlx5_mcast_coll_comm_t *comm)
{
    struct ibv_wc wc;
    int           num_comp;
    
    num_comp = ibv_poll_cq(comm->scq, 1, &wc);
    
    tl_trace(comm->lib, "polled send completions: %d", num_comp);
    
    if (num_comp < 0) {
        tl_error(comm->lib, "send queue poll completion failed %d", num_comp);
        return UCC_ERR_NO_MESSAGE;
    } else if (num_comp > 0) {
        if (IBV_WC_SUCCESS != wc.status) {
           tl_error(comm->lib, "mcast_poll_send: %s err %d num_comp",
                    ibv_wc_status_str(wc.status), num_comp);
            return UCC_ERR_NO_MESSAGE;
        }
        comm->pending_send -= num_comp;
    }

    return UCC_OK;
}

static inline ucc_status_t ucc_tl_mlx5_mcast_send(ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                                  ucc_tl_mlx5_mcast_coll_req_t *req,
                                                  int num_packets, const int zcopy)
{
    struct ibv_send_wr *swr            = &comm->mcast.swr;
    struct ibv_sge     *ssg            = &comm->mcast.ssg;
    int                 max_per_packet = comm->max_per_packet;
    int                 offset         = req->offset, i;
    struct ibv_send_wr *bad_wr;
    struct pp_packet   *pp;
    int                 rc;
    int                 length;
    ucc_status_t        status;

    for (i = 0; i < num_packets; i++) {
        if (comm->params.sx_depth <=
               (comm->pending_send * comm->params.scq_moderation + comm->tx)) {
            status = ucc_tl_mlx5_mcast_poll_send(comm);
            if (UCC_OK != status) {
                return status;
            }
            break;
        }

        if (NULL == (pp = ucc_tl_mlx5_mcast_buf_get_free(comm))) {
            break;
        }

        ucc_assert(pp->context == 0);

        __builtin_prefetch((void*) pp->buf);
        __builtin_prefetch(PTR_OFFSET(req->ptr, offset));

        length      = req->to_send == 1 ? req->last_pkt_len : max_per_packet;
        pp->length  = length;
        pp->psn     = comm->psn;
        ssg[0].addr = (uintptr_t) PTR_OFFSET(req->ptr, offset);

        if (zcopy) {
            pp->context = (uintptr_t) PTR_OFFSET(req->ptr, offset);
        } else {
            memcpy((void*) pp->buf, PTR_OFFSET(req->ptr, offset), length);
            ssg[0].addr = (uint64_t) pp->buf;
        }

        ssg[0].length     = length;
        ssg[0].lkey       = req->mr->lkey;
        swr[0].wr_id      = MCAST_BCASTSEND_WR;
        swr[0].imm_data   = htonl(pp->psn);
        swr[0].send_flags = (length <= comm->max_inline) ? IBV_SEND_INLINE : 0;

        comm->r_window[pp->psn & (comm->wsize-1)] = pp;
        comm->psn++;
        req->to_send--;
        offset += length;
        comm->tx++;

        if (comm->tx == comm->params.scq_moderation) {
            swr[0].send_flags |= IBV_SEND_SIGNALED;
            comm->tx           = 0;
            comm->pending_send++;
        }

        tl_trace(comm->lib, "post_send, psn %d, length %d, zcopy %d, signaled %d",
                 pp->psn, pp->length, zcopy, swr[0].send_flags & IBV_SEND_SIGNALED);

        if (0 != (rc = ibv_post_send(comm->mcast.qp, &swr[0], &bad_wr))) {
            tl_error(comm->lib, "post send failed: ret %d, start_psn %d, to_send %d, "
                    "to_recv %d, length %d, psn %d, inline %d",
                     rc, req->start_psn, req->to_send, req->to_recv,
                     length, pp->psn, length <= comm->max_inline);
            return UCC_ERR_NO_MESSAGE;
        }

        status = ucc_tl_mlx5_mcast_check_nack_requests(comm, pp->psn);
        if (UCC_OK != status) {
            return status;
        }
    }

    req->offset = offset;

    return UCC_OK;
}

static inline ucc_status_t ucc_tl_mlx5_mcast_process_pp(ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                                       ucc_tl_mlx5_mcast_coll_req_t *req,
                                                       struct pp_packet *pp,
                                                       int *num_left, int in_pending_queue)
{
    ucc_status_t status = UCC_OK;

    if (PSN_RECEIVED(pp->psn, comm) || pp->psn < comm->last_acked) {
        /* This psn was already received */
        ucc_assert(pp->context == 0);
        if (in_pending_queue) {
            /* this pp belongs to pending queue so remove it */
            ucc_list_del(&pp->super);
        }
        /* add pp to the free pool */
        ucc_list_add_tail(&comm->bpool, &pp->super);
    } else if (PSN_IS_IN_RANGE(pp->psn, req, comm)) {
        if (*num_left <= 0 && !in_pending_queue) {
            /* we just received this packet and it is in order, but there is no
             * more space in window so we need to place this packet in the
             * pending queue for future processings */
            ucc_list_add_tail(&comm->pending_q, &pp->super);
        } else {
            __builtin_prefetch(PTR_OFFSET(req->ptr, PSN_TO_RECV_OFFSET(pp->psn, req, comm)));
            __builtin_prefetch((void*) pp->buf);
            if (in_pending_queue) {
                ucc_list_del(&pp->super);
            }
            status = ucc_tl_mlx5_mcast_process_packet(comm, req, pp);
            if (UCC_OK != status) {
                return status;
            }
            (*num_left)--;
        }
    } else if (!in_pending_queue) {
        /* add pp to the pending queue as it is out of order */
        ucc_list_add_tail(&comm->pending_q, &pp->super);
    }

    return status;
}

/* this function return the number of mcast recv packets that
 * are left or -1 in case of error */
static inline int ucc_tl_mlx5_mcast_recv(ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                         ucc_tl_mlx5_mcast_coll_req_t *req,
                                         int num_left, int *pending_q_size)
{
    struct pp_packet *pp;
    struct pp_packet *next;
    uint64_t          id;
    struct ibv_wc    *wc;
    int               num_comp;
    int               i;
    int               real_num_comp;
    ucc_status_t      status;

    /* check if we have already received something */
    ucc_list_for_each_safe(pp, next, &comm->pending_q, super) {
        status = ucc_tl_mlx5_mcast_process_pp(comm, req, pp, &num_left, 1);
        if (UCC_OK != status) {
            return -1;
        }
        (*pending_q_size)++;
    }

    wc = ucc_malloc(sizeof(struct ibv_wc) * POLL_PACKED, "WC");
    if (!wc) {
        tl_error(comm->lib, "ucc_malloc failed");
        return -1;
    }

    while (num_left > 0)
    {
        memset(wc, 0, sizeof(struct ibv_wc) * POLL_PACKED);
        num_comp = ibv_poll_cq(comm->rcq, POLL_PACKED, wc);

        if (num_comp < 0) {
            tl_error(comm->lib, "recv queue poll completion failed %d", num_comp);
            ucc_free(wc);
            return -1;
        } else if (num_comp == 0) {
            break;
        }

        real_num_comp = num_comp;

        for (i = 0; i < real_num_comp; i++) {
            if (IBV_WC_SUCCESS != wc[i].status) {
                tl_error(comm->lib, "mcast_recv: %s err pending_recv %d wr_id %ld"
                         " num_comp %d byte_len %d",
                         ibv_wc_status_str(wc[i].status), comm->pending_recv,
                         wc[i].wr_id, num_comp, wc[i].byte_len);
                ucc_free(wc);
                return -1;
            }

            id         = wc[i].wr_id;
            pp         = (struct pp_packet*) (id);
            pp->length = wc[i].byte_len - GRH_LENGTH;
            pp->psn    = ntohl(wc[i].imm_data);

            tl_trace(comm->lib, "completion: psn %d, length %d, already_received %d, "
                                " psn in req %d, req_start %d, req_num packets"
                                " %d, to_send %d, to_recv %d, num_left %d",
                                pp->psn, pp->length, PSN_RECEIVED(pp->psn,
                                comm) > 0, PSN_IS_IN_RANGE(pp->psn, req,
                                comm), req->start_psn, req->num_packets,
                                req->to_send, req->to_recv, num_left);

            status = ucc_tl_mlx5_mcast_process_pp(comm, req, pp, &num_left, 0);
            if (UCC_OK != status) {
                return -1;
            }
        }

        comm->pending_recv -= num_comp;
        status = ucc_tl_mlx5_mcast_post_recv_buffers(comm);
        if (UCC_OK != status) {
            return -1;
        }
    }

    ucc_free(wc);
    return num_left;
}

static inline ucc_status_t ucc_tl_mlx5_mcast_poll_recv(ucc_tl_mlx5_mcast_coll_comm_t *comm)
{
    ucc_status_t      status = UCC_OK;
    struct pp_packet *pp;
    struct ibv_wc     wc;
    int               num_comp;
    uint64_t          id;
    int               length;
    uint32_t          psn;

    do {
        num_comp = ibv_poll_cq(comm->rcq, 1, &wc);

        if (num_comp > 0) {
            
            if (IBV_WC_SUCCESS != wc.status) {
                tl_error(comm->lib, "mcast_poll_recv: %s err %d num_comp",
                        ibv_wc_status_str(wc.status), num_comp);
                status = UCC_ERR_NO_MESSAGE;
                return status;
            }

            // Make sure we received all in order.
            id     = wc.wr_id;
            length = wc.byte_len - GRH_LENGTH;
            psn    = ntohl(wc.imm_data);
            pp     = (struct pp_packet*) id;

            if (psn >= comm->psn) {
                ucc_assert(!PSN_RECEIVED(psn, comm));
                pp->psn    = psn;
                pp->length = length;
                ucc_list_add_tail(&comm->pending_q, &pp->super);
            } else {
                ucc_assert(pp->context == 0);
                ucc_list_add_tail(&comm->bpool, &pp->super);
            }

            comm->pending_recv--;
            status = ucc_tl_mlx5_mcast_post_recv_buffers(comm);
            if (UCC_OK != status) {
                return status;
            }
        } else if (num_comp != 0) {
            tl_error(comm->lib, "mcast_poll_recv: %d num_comp", num_comp);
            status = UCC_ERR_NO_MESSAGE;
            return status;
        }
    } while (num_comp);

    return status;
}

static inline ucc_status_t ucc_tl_mlx5_mcast_reliable(ucc_tl_mlx5_mcast_coll_comm_t *comm)
{
    ucc_status_t status = UCC_OK;

    if (comm->racks_n != comm->child_n || comm->sacks_n != comm->parent_n ||
           comm->nack_requests) { 
        if (comm->pending_send) {
            status = ucc_tl_mlx5_mcast_poll_send(comm);
            if (UCC_OK != status) {
                return status;
            }
        }
        
        if (comm->parent_n) {
            status = ucc_tl_mlx5_mcast_poll_recv(comm);
            if (UCC_OK != status) {
                return status;
            }
        }
        
        status = ucc_tl_mlx5_mcast_check_nack_requests_all(comm);
        if (UCC_OK != status) {
            return status;
        }
    }

    if (comm->parent_n && !comm->reliable_in_progress) {
        status = ucc_tl_mlx5_mcast_reliable_send(comm);
        if (UCC_OK != status) {
            return status;
        }
    }

    if (!comm->reliable_in_progress) {
        comm->reliable_in_progress = 1;
    }

    if (comm->racks_n == comm->child_n && comm->sacks_n == comm->parent_n &&
           0 == comm->nack_requests) {
        // Reset for next round.
        memset(comm->parents,  0, sizeof(comm->parents));
        memset(comm->children, 0, sizeof(comm->children));

        comm->racks_n = comm->child_n  = 0;
        comm->sacks_n = comm->parent_n = 0;
        comm->reliable_in_progress     = 0;

        return UCC_OK;
    }

    return UCC_INPROGRESS;
}

ucc_status_t ucc_tl_mlx5_probe_ip_over_ib(char* ib_dev_list,
                                          struct sockaddr_storage *addr);

ucc_status_t ucc_tl_mlx5_setup_mcast(ucc_tl_mlx5_mcast_coll_comm_t *comm);

ucc_status_t ucc_tl_mlx5_mcast_init_qps(ucc_tl_mlx5_mcast_coll_context_t *ctx,
                                        ucc_tl_mlx5_mcast_coll_comm_t *comm);

ucc_status_t ucc_tl_mlx5_mcast_setup_qps(ucc_tl_mlx5_mcast_coll_context_t *ctx,
                                         ucc_tl_mlx5_mcast_coll_comm_t *comm);

ucc_status_t ucc_tl_mlx5_clean_mcast_comm(ucc_tl_mlx5_mcast_coll_comm_t *comm);

#endif /* TL_MLX5_MCAST_HELPER_H_ */
