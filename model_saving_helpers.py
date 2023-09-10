

from os import path
import shutil


def get_model_save_name(model_name: str, model_hparams: dict, optimizer_hparams: dict, experiment_config: dict):
  channels = model_hparams["channels"]
  gconv_layers = model_hparams["gconv_layers"] if model_name == "GCNN" else ""
  conv_layers = model_hparams["conv_layers"] if model_name == "STCNN" else ""
  kernel_size = model_hparams["kernel_size"]
  batch_size = experiment_config["batch_size"]
  max_epochs = experiment_config["max_epochs"]
  seed = experiment_config["seed"]
  lr = optimizer_hparams["lr"]
  weight_decay = optimizer_hparams["weight_decay"]
  order = model_hparams["group"].order if "group" in model_hparams else ""
  name_parts = [
      model_name,
      f"order={order}" if order else "",
      f"gconv_layers={gconv_layers}" if gconv_layers else "",
      f"conv_layers={conv_layers}" if conv_layers else "",
      f"channels={channels}",
      f"kernel_size={kernel_size}",
      f"batch_size={batch_size}",
      f"lr={lr}",
      f"weight_decay={weight_decay}",
      f"max-epochs={max_epochs}",
      f"seed={seed}"
  ]
  return '-'.join([name_part for name_part in name_parts if len(name_part)])


def get_checkpoint_file_name(model_name: str, model_hparams: dict, optimizer_hparams: dict, experiment_config: dict, experiment_run: int = None):
  model_save_name = get_model_save_name(
      model_name,
      model_hparams,
      optimizer_hparams,
      experiment_config
  )
  if experiment_run == None:
    return f"checkpoint-{model_save_name}.ckpt"
  else:
    return f"checkpoint-run={experiment_run}-{model_save_name}.ckpt"


def get_metrics_file_name(model_name: str, model_hparams: dict, optimizer_hparams: dict, experiment_config: dict, experiment_run: int = None):
  model_save_name = get_model_save_name(
      model_name,
      model_hparams,
      optimizer_hparams,
      experiment_config
  )
  if experiment_run == None:
    return f"metrics-{model_save_name}.csv"
  else:
    return f"metrics-run={experiment_run}-{model_save_name}.csv"


def save_metrics_and_checkpoint(
    trainer,
    model_name: str,
    model_hparams: dict,
    optimizer_hparams: dict,
    experiment_config: dict,
    drive_results_path: str,
    experiment_run: int = None,
):
  checkpoint_file_name = get_checkpoint_file_name(
      model_name,
      model_hparams,
      optimizer_hparams,
      experiment_config,
      experiment_run
  )
  metrics_file_name = get_metrics_file_name(
      model_name,
      model_hparams,
      optimizer_hparams,
      experiment_config,
      experiment_run
  )

  drive_checkpoint_file_name = path.join(
      drive_results_path,
      checkpoint_file_name
  )
  drive_metrics_file_name = path.join(drive_results_path, metrics_file_name)

  shutil.copyfile(trainer.checkpoint_callback.best_model_path,
                  checkpoint_file_name)
  shutil.copyfile(checkpoint_file_name, drive_checkpoint_file_name)
  shutil.copyfile(f"{trainer.logger.log_dir}/metrics.csv", metrics_file_name)
  shutil.copyfile(metrics_file_name, drive_metrics_file_name)


def remove_old_metrics_and_checkpoint():
  shutil.rmtree('logs')
