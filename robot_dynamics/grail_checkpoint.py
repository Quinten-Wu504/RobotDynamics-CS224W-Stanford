import torch

from robot_dynamics.data_audit import source_version


def save_checkpoint(path, model, optimizer, epoch, best, logs, generator, identity):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_score": best,
        "logs": logs,
        "generator": generator.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all()
        if next(model.parameters()).is_cuda
        else [],
        "identity": identity,
        **source_version(),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
