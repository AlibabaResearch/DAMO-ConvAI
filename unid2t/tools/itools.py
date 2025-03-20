import os
import psutil


def get_current_memory_gb():
    pid = os.getpid()

    p = psutil.Process(pid)

    info = p.memory_full_info()

    return info.uss / 1024. / 1024. / 1024.