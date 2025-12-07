#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IB_TRACE_CAPACITY (1u << 20)

typedef struct IbTraceRecord {
  uint64_t t_ns;
  uint64_t wr_id;
  uint32_t size;
  uint16_t dev;
  uint16_t qp;
  uint8_t opcode;
  uint8_t is_send;
  uint8_t phase;
  uint8_t status;
  uint32_t extra;
} IbTraceRecord;

#ifdef IB_TRACE_ENABLE

typedef struct IbTraceBuffer {
  IbTraceRecord records[IB_TRACE_CAPACITY];
  uint64_t write_index;
} IbTraceBuffer;

extern IbTraceBuffer g_ib_trace;

void ib_trace_log(uint64_t wr_id,
                  uint32_t size,
                  uint16_t dev,
                  uint16_t qp,
                  uint8_t opcode,
                  uint8_t is_send,
                  uint8_t phase,
                  uint8_t status,
                  uint32_t extra);

void ib_trace_dump_from_env(const char* env_name);

#define IB_TRACE_POST_SEND(wr_id, size, dev, qp, opcode, extra) \
  ib_trace_log((wr_id), (uint32_t)(size), (uint16_t)(dev),      \
               (uint16_t)(qp), (uint8_t)(opcode), 1, 0, 0xff,   \
               (uint32_t)(extra))

#define IB_TRACE_POST_RECV(wr_id, size, dev, qp, opcode, extra) \
  ib_trace_log((wr_id), (uint32_t)(size), (uint16_t)(dev),      \
               (uint16_t)(qp), (uint8_t)(opcode), 0, 0, 0xff,   \
               (uint32_t)(extra))

#define IB_TRACE_COMPLETE(wr_id, size, dev, qp, opcode, status, is_send, extra) \
  ib_trace_log((wr_id), (uint32_t)(size), (uint16_t)(dev),                      \
               (uint16_t)(qp), (uint8_t)(opcode), (uint8_t)(is_send), 1,        \
               (uint8_t)(status), (uint32_t)(extra))

#else

#define IB_TRACE_POST_SEND(...)
#define IB_TRACE_POST_RECV(...)
#define IB_TRACE_COMPLETE(...)
static inline void ib_trace_dump_from_env(const char* env_name) {
  (void)env_name;
}

#endif

#ifdef __cplusplus
}
#endif
