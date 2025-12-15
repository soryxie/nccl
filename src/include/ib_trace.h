#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IB_TRACE_CAPACITY (1u << 20)

// 默认 flush 间隔（毫秒），可通过环境变量 NCCL_IB_TRACE_FLUSH_MS 覆盖
#define IB_TRACE_DEFAULT_FLUSH_INTERVAL_MS 1000

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

// 同步 dump（程序退出时使用）
void ib_trace_dump_from_env(const char* env_name);

// 异步 flush API
// 启动后台线程，定时增量 flush 到文件
// env_name: 环境变量名，指定输出文件路径
// flush_interval_ms: flush 间隔（毫秒），0 表示使用默认值或环境变量
void ib_trace_start_async(const char* env_name, int flush_interval_ms);

// 停止后台线程，flush 剩余数据
void ib_trace_stop_async(void);

// 手动触发一次 flush（可选）
void ib_trace_flush_now(void);

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
static inline void ib_trace_start_async(const char* env_name, int flush_interval_ms) {
  (void)env_name;
  (void)flush_interval_ms;
}
static inline void ib_trace_stop_async(void) {}
static inline void ib_trace_flush_now(void) {}

#endif

#ifdef __cplusplus
}
#endif
