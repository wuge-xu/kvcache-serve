import os
import time

from app.scheduler.redis_queue import redis_queue


def main():
    timeout_seconds = float(
        os.getenv("PROCESSING_TIMEOUT_SECONDS", "30")
    )
    interval_seconds = float(
        os.getenv("REAPER_INTERVAL_SECONDS", "5")
    )
    max_recoveries = int(
        os.getenv("MAX_RECOVERIES", "2")
    )

    print("[Reaper] Processing task reaper started.")
    print(
        f"[Reaper] timeout={timeout_seconds}s, "
        f"interval={interval_seconds}s, "
        f"max_recoveries={max_recoveries}"
    )

    while True:
        try:
            result = redis_queue.recover_stale_tasks(
                timeout_seconds=timeout_seconds,
                max_recoveries=max_recoveries,
            )

            if result["stale"] > 0:
                print(
                    "[Reaper] Scan result: "
                    f"scanned={result['scanned']}, "
                    f"stale={result['stale']}, "
                    f"requeued={result['requeued']}, "
                    f"dead_lettered={result['dead_lettered']}, "
                    f"skipped={result['skipped']}"
                )

        except Exception as error:
            print(f"[Reaper] Scan failed: {error}")

        time.sleep(interval_seconds)


if __name__ == "__main__":
    main()
