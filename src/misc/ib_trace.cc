#include "ib_trace.h"

#ifdef IB_TRACE_ENABLE

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#ifdef __cplusplus
extern "C" {
#endif

IbTraceBuffer g_ib_trace = {};

static inline uint64_t ib_trace_now_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

void ib_trace_log(uint64_t wr_id,
                  uint32_t size,
                  uint16_t dev,
                  uint16_t qp,
                  uint8_t opcode,
                  uint8_t is_send,
                  uint8_t phase,
                  uint8_t status,
                  uint32_t extra) {
  uint64_t idx = __atomic_fetch_add(&g_ib_trace.write_index, 1, __ATOMIC_RELAXED);
  IbTraceRecord* rec = &g_ib_trace.records[idx & (IB_TRACE_CAPACITY - 1)];
  rec->t_ns = ib_trace_now_ns();
  rec->wr_id = wr_id;
  rec->size = size;
  rec->dev = dev;
  rec->qp = qp;
  rec->opcode = opcode;
  rec->is_send = is_send;
  rec->phase = phase;
  rec->status = status;
  rec->extra = extra;
}

typedef struct IbTraceFileHeader {
  char magic[8];
  uint32_t version;
  uint32_t record_size;
  uint64_t count;
} IbTraceFileHeader;

static void ib_trace_write_all(int fd, const void* buffer, size_t total) {
  const char* ptr = (const char*)buffer;
  size_t remaining = total;
  while (remaining > 0) {
    ssize_t written = write(fd, ptr, remaining);
    if (written < 0) {
      if (errno == EINTR) continue;
      return;
    }
    if (written == 0) return;
    ptr += written;
    remaining -= (size_t)written;
  }
}

static void ib_trace_dump_to_file(const char* path) {
  if (!path || !path[0]) return;

  uint64_t total = __atomic_load_n(&g_ib_trace.write_index, __ATOMIC_RELAXED);
  uint64_t cap = IB_TRACE_CAPACITY;
  uint64_t count = total < cap ? total : cap;
  if (count == 0) return;

  int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) return;

  IbTraceFileHeader header;
  memset(&header, 0, sizeof(header));
  memcpy(header.magic, "IBTRACE", 7);
  header.version = 1;
  header.record_size = sizeof(IbTraceRecord);
  header.count = count;

  ib_trace_write_all(fd, &header, sizeof(header));

  uint64_t start = (total > cap) ? (total - cap) : 0;
  uint64_t first_index = start & (cap - 1);
  uint64_t first_batch = count;
  if (first_batch > (cap - first_index)) {
    first_batch = cap - first_index;
  }

  ib_trace_write_all(fd, &g_ib_trace.records[first_index],
                     first_batch * sizeof(IbTraceRecord));

  if (first_batch < count) {
    ib_trace_write_all(fd, &g_ib_trace.records[0],
                       (count - first_batch) * sizeof(IbTraceRecord));
  }

  close(fd);
}

void ib_trace_dump_from_env(const char* env_name) {
  if (!env_name || !env_name[0]) return;
  const char* path = getenv(env_name);
  if (!path || !path[0]) return;
  ib_trace_dump_to_file(path);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IB_TRACE_ENABLE
