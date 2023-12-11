from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

def read_tensorboard_data(tensorboard_log_path, val_name):
    ea = event_accumulator.EventAccumulator(tensorboard_log_path)
    ea.Reload()
    
    print("All scalers:")
    print(ea.scalars.Keys())

    val = ea.scalars.Items(val_name)
    return val

def plot(vals, val_names, max_step=None):
    plt.figure()

    for val, val_name in zip(vals, val_names):
        x = [i.step for i in val]
        y = [i.value for i in val]

        if max_step is not None:
            x = [i for i in x if i < max_step]
            y = y[:len(x)]

        plt.plot(x, y, label=val_name)

    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    refine_uie_tensorboard_log_path = "tensorboard_logs/events.out.tfevents.1654419004.dsw32050-7df697f45c-6bwkm.44438.0"
    refine_t5_tensorboard_log_path = "tensorboard_logs/events.out.tfevents.1654361305.g64h07153.cloud.sqa.nt12.129194.0"
    uie_t5_tensorboard_log_path = "tensorboard_logs/events.out.tfevents.1654275965.eflops-common033255085104.NT12.106708.0"
    
    val_name = "train/loss"

    refine_uie_val = read_tensorboard_data(refine_uie_tensorboard_log_path, val_name)
    refine_t5_val = read_tensorboard_data(refine_t5_tensorboard_log_path, val_name)
    uie_t5_val = read_tensorboard_data(uie_t5_tensorboard_log_path, val_name)

    vals = [refine_uie_val, refine_t5_val, uie_t5_val]
    val_names = ["refine_uie_loss", "refine_t5_loss", "uie_t5_loss"]
    max_step = 20000
    plot(vals, val_names, max_step=max_step)