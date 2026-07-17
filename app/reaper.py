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

    print("[Reaper] Processing task reaper started.")
    print(
        f"[Reaper] timeout={timeout_seconds}s, "
        f"interval={interval_seconds}s"
    )

    while True:
        try:
            result = redis_queue.recover_stale_tasks(
                timeout_seconds=timeout_seconds,
            )

            if result["stale"] > 0:
                print(
                    "[Reaper] Scan result: "
                    f"scanned={result['scanned']}, "
                    f"stale={result['stale']}, "
                    f"requeued={result['requeued']}, "
                    f"skipped={result['skipped']}"
                )

        except Exception as error:
            print(f"[Reaper] Scan failed: {error}")

        time.sleep(interval_seconds)


if __name__ == "__main__":
    main()
