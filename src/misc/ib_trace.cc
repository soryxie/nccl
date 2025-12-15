#include "ib_trace.h"

#ifdef IB_TRACE_ENABLE

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
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

// ============================================================================
// 异步 flush 相关状态
// ============================================================================
static struct {
  pthread_t thread;
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  int running;
  int initialized;
  int fd;                    // 输出文件描述符
  uint64_t flushed_index;    // 已经 flush 到文件的索引
  uint64_t total_flushed;    // 累计 flush 的记录数
  int flush_interval_ms;     // flush 间隔
  char file_path[4096];      // 输出文件路径
} g_ib_trace_async = {
  .thread = 0,
  .mutex = PTHREAD_MUTEX_INITIALIZER,
  .cond = PTHREAD_COND_INITIALIZER,
  .running = 0,
  .initialized = 0,
  .fd = -1,
  .flushed_index = 0,
  .total_flushed = 0,
  .flush_interval_ms = IB_TRACE_DEFAULT_FLUSH_INTERVAL_MS,
  .file_path = {0}
};

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

// ============================================================================
// 同步 dump（原有功能，用于程序退出时一次性 dump）
// ============================================================================
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
  fprintf(stderr,
          "NCCL_IB_TRACE: dumping %llu records to \"%s\".\n",
          (unsigned long long)count, path);

  int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) {
    fprintf(stderr,
            "NCCL_IB_TRACE: failed to open \"%s\" for writing: %s\n",
            path, strerror(errno));
    return;
  }

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
  fprintf(stderr,
          "NCCL_IB_TRACE: dump to \"%s\" completed (header + %llu records).\n",
          path, (unsigned long long)count);
}

void ib_trace_dump_from_env(const char* env_name) {
  if (!env_name || !env_name[0]) return;
  const char* path = getenv(env_name);
  if (!path || !path[0]) return;
  ib_trace_dump_to_file(path);
}

// ============================================================================
// 异步 flush 实现
// ============================================================================

// 日志级别：0=关闭, 1=基本信息, 2=详细信息
static int g_ib_trace_log_level = -1;  // -1 表示未初始化

static int ib_trace_get_log_level(void) {
  if (g_ib_trace_log_level < 0) {
    const char* env = getenv("NCCL_IB_TRACE_LOG_LEVEL");
    if (env && env[0]) {
      g_ib_trace_log_level = atoi(env);
    } else {
      g_ib_trace_log_level = 1;  // 默认输出基本信息
    }
  }
  return g_ib_trace_log_level;
}

// 增量 flush：只写入自上次 flush 以来的新记录
static void ib_trace_flush_incremental_unlocked(void) {
  if (g_ib_trace_async.fd < 0) return;

  uint64_t current = __atomic_load_n(&g_ib_trace.write_index, __ATOMIC_RELAXED);
  uint64_t flushed = g_ib_trace_async.flushed_index;

  if (current <= flushed) return;  // 没有新记录

  uint64_t cap = IB_TRACE_CAPACITY;
  uint64_t new_records = current - flushed;

  // 如果新记录超过缓冲区容量，说明发生了覆盖，只能 flush 最新的 cap 条
  if (new_records > cap) {
    flushed = current - cap;
    uint64_t lost = new_records - cap;
    new_records = cap;
    fprintf(stderr,
            "NCCL_IB_TRACE: WARNING - buffer overflow, %llu records lost.\n",
            (unsigned long long)lost);
  }

  // 计算要写入的范围
  uint64_t start_index = flushed & (cap - 1);
  uint64_t end_index = current & (cap - 1);

  if (start_index < end_index) {
    // 连续区域
    ib_trace_write_all(g_ib_trace_async.fd,
                       &g_ib_trace.records[start_index],
                       (end_index - start_index) * sizeof(IbTraceRecord));
  } else if (start_index > end_index || new_records == cap) {
    // 跨越边界，分两段写入
    // 第一段：从 start_index 到缓冲区末尾
    ib_trace_write_all(g_ib_trace_async.fd,
                       &g_ib_trace.records[start_index],
                       (cap - start_index) * sizeof(IbTraceRecord));
    // 第二段：从缓冲区开头到 end_index
    if (end_index > 0) {
      ib_trace_write_all(g_ib_trace_async.fd,
                         &g_ib_trace.records[0],
                         end_index * sizeof(IbTraceRecord));
    }
  }

  g_ib_trace_async.flushed_index = current;
  g_ib_trace_async.total_flushed += new_records;

  // 输出日志
  int log_level = ib_trace_get_log_level();
  if (log_level >= 2) {
    // 详细日志：每次 flush 都输出
    fprintf(stderr,
            "NCCL_IB_TRACE: [ASYNC FLUSH] +%llu records (total: %llu)\n",
            (unsigned long long)new_records,
            (unsigned long long)g_ib_trace_async.total_flushed);
  } else if (log_level >= 1) {
    // 基本日志：每 10000 条输出一次
    static uint64_t last_logged = 0;
    if (g_ib_trace_async.total_flushed - last_logged >= 10000) {
      fprintf(stderr,
              "NCCL_IB_TRACE: [ASYNC] total flushed: %llu records\n",
              (unsigned long long)g_ib_trace_async.total_flushed);
      last_logged = g_ib_trace_async.total_flushed;
    }
  }
}

// 后台线程函数
static void* ib_trace_async_thread(void* arg) {
  (void)arg;

  int log_level = ib_trace_get_log_level();
  uint64_t flush_count = 0;

  if (log_level >= 1) {
    fprintf(stderr, "NCCL_IB_TRACE: [ASYNC] background thread started (tid=%ld)\n",
            (long)pthread_self());
  }

  pthread_mutex_lock(&g_ib_trace_async.mutex);
  while (g_ib_trace_async.running) {
    // 计算超时时间
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += g_ib_trace_async.flush_interval_ms / 1000;
    ts.tv_nsec += (g_ib_trace_async.flush_interval_ms % 1000) * 1000000;
    if (ts.tv_nsec >= 1000000000) {
      ts.tv_sec++;
      ts.tv_nsec -= 1000000000;
    }

    // 等待超时或被唤醒
    pthread_cond_timedwait(&g_ib_trace_async.cond,
                           &g_ib_trace_async.mutex, &ts);

    // 执行增量 flush
    uint64_t before = g_ib_trace_async.total_flushed;
    ib_trace_flush_incremental_unlocked();
    flush_count++;

    // 详细日志：输出每次 flush 尝试
    if (log_level >= 2 && g_ib_trace_async.total_flushed == before) {
      fprintf(stderr, "NCCL_IB_TRACE: [ASYNC] flush #%llu - no new records\n",
              (unsigned long long)flush_count);
    }
  }
  pthread_mutex_unlock(&g_ib_trace_async.mutex);

  if (log_level >= 1) {
    fprintf(stderr, "NCCL_IB_TRACE: [ASYNC] background thread exiting (flush_count=%llu)\n",
            (unsigned long long)flush_count);
  }

  return NULL;
}

void ib_trace_start_async(const char* env_name, int flush_interval_ms) {
  pthread_mutex_lock(&g_ib_trace_async.mutex);

  if (g_ib_trace_async.initialized) {
    pthread_mutex_unlock(&g_ib_trace_async.mutex);
    return;  // 已经初始化
  }

  // 获取文件路径
  const char* path = NULL;
  if (env_name && env_name[0]) {
    path = getenv(env_name);
  }
  if (!path || !path[0]) {
    fprintf(stderr, "NCCL_IB_TRACE: async mode - no output file specified.\n");
    pthread_mutex_unlock(&g_ib_trace_async.mutex);
    return;
  }

  strncpy(g_ib_trace_async.file_path, path, sizeof(g_ib_trace_async.file_path) - 1);

  // 获取 flush 间隔
  if (flush_interval_ms > 0) {
    g_ib_trace_async.flush_interval_ms = flush_interval_ms;
  } else {
    const char* interval_env = getenv("NCCL_IB_TRACE_FLUSH_MS");
    if (interval_env && interval_env[0]) {
      int val = atoi(interval_env);
      if (val > 0) {
        g_ib_trace_async.flush_interval_ms = val;
      }
    }
  }

  // 打开文件（追加模式用于异步写入，但我们用新文件格式）
  // 异步模式不写 header，直接写 records（可以用单独工具解析）
  g_ib_trace_async.fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (g_ib_trace_async.fd < 0) {
    fprintf(stderr,
            "NCCL_IB_TRACE: async mode - failed to open \"%s\": %s\n",
            path, strerror(errno));
    pthread_mutex_unlock(&g_ib_trace_async.mutex);
    return;
  }

  // 写入文件头
  IbTraceFileHeader header;
  memset(&header, 0, sizeof(header));
  memcpy(header.magic, "IBTRACES", 8);  // 'S' for streaming
  header.version = 2;  // 版本 2 表示流式格式
  header.record_size = sizeof(IbTraceRecord);
  header.count = 0;  // 流式模式，count 在最后更新
  ib_trace_write_all(g_ib_trace_async.fd, &header, sizeof(header));

  g_ib_trace_async.flushed_index = 0;
  g_ib_trace_async.total_flushed = 0;
  g_ib_trace_async.running = 1;
  g_ib_trace_async.initialized = 1;

  // 启动后台线程
  if (pthread_create(&g_ib_trace_async.thread, NULL,
                     ib_trace_async_thread, NULL) != 0) {
    fprintf(stderr, "NCCL_IB_TRACE: failed to create async thread.\n");
    close(g_ib_trace_async.fd);
    g_ib_trace_async.fd = -1;
    g_ib_trace_async.running = 0;
    g_ib_trace_async.initialized = 0;
    pthread_mutex_unlock(&g_ib_trace_async.mutex);
    return;
  }

  fprintf(stderr,
          "NCCL_IB_TRACE: async mode started. flush_interval=%dms, file=\"%s\"\n",
          g_ib_trace_async.flush_interval_ms, path);

  pthread_mutex_unlock(&g_ib_trace_async.mutex);
}

void ib_trace_stop_async(void) {
  pthread_mutex_lock(&g_ib_trace_async.mutex);

  if (!g_ib_trace_async.initialized || !g_ib_trace_async.running) {
    pthread_mutex_unlock(&g_ib_trace_async.mutex);
    return;
  }

  g_ib_trace_async.running = 0;
  pthread_cond_signal(&g_ib_trace_async.cond);
  pthread_mutex_unlock(&g_ib_trace_async.mutex);

  // 等待线程结束
  pthread_join(g_ib_trace_async.thread, NULL);

  pthread_mutex_lock(&g_ib_trace_async.mutex);

  // 最后一次 flush
  ib_trace_flush_incremental_unlocked();

  // 更新文件头中的 count
  if (g_ib_trace_async.fd >= 0) {
    // 回到文件头更新 count
    lseek(g_ib_trace_async.fd, offsetof(IbTraceFileHeader, count), SEEK_SET);
    uint64_t total = g_ib_trace_async.total_flushed;
    ib_trace_write_all(g_ib_trace_async.fd, &total, sizeof(total));

    close(g_ib_trace_async.fd);
    g_ib_trace_async.fd = -1;

    fprintf(stderr,
            "NCCL_IB_TRACE: async mode stopped. total flushed=%llu records to \"%s\"\n",
            (unsigned long long)total, g_ib_trace_async.file_path);
  }

  g_ib_trace_async.initialized = 0;
  pthread_mutex_unlock(&g_ib_trace_async.mutex);
}

void ib_trace_flush_now(void) {
  pthread_mutex_lock(&g_ib_trace_async.mutex);
  if (g_ib_trace_async.initialized && g_ib_trace_async.running) {
    ib_trace_flush_incremental_unlocked();
  }
  pthread_mutex_unlock(&g_ib_trace_async.mutex);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IB_TRACE_ENABLE
