import argparse
import json
import math
import random
import time
from pathlib import Path

import minitorch
from project.run_scalar import ScalarTrain
from project.run_tensor import TensorTrain
from project.run_fast_tensor import FastTrain, FastTensorBackend


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['scalar', 'tensor', 'fast', 'cuda'], default='scalar')
    p.add_argument('--dataset', choices=list(minitorch.datasets), default='Simple')
    p.add_argument('--hidden', type=int, default=10)
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--points', type=int, default=50)
    p.add_argument('--rate', type=float, default=0.1)
    args = p.parse_args()
    random.seed(0)
    data = minitorch.datasets[args.dataset](args.points)
    if args.mode == 'scalar':
        trainer = ScalarTrain(args.hidden)
    elif args.mode == 'tensor':
        trainer = TensorTrain(args.hidden)
    else:
        backend = FastTensorBackend if args.mode == 'fast' else minitorch.TensorBackend(minitorch.CudaOps)
        trainer = FastTrain(args.hidden, backend)
    rows = []
    loss_history = []
    start = time.perf_counter()

    def log(epoch, total_loss, correct, losses):
        if not math.isfinite(float(total_loss)):
            raise FloatingPointError('Non-finite loss')
        row = dict(epoch=epoch, loss=float(total_loss), correct=int(correct),
                   elapsed_seconds=time.perf_counter() - start)
        rows.append(row)
        loss_history[:] = [float(x) for x in losses]
        print(json.dumps(row), flush=True)

    trainer.train(data, args.rate, max_epochs=args.epochs, log_fn=log)
    output = Path('results')
    output.mkdir(exist_ok=True)
    (output / f'{args.mode}-{args.dataset}-h{args.hidden}-n{args.points}.json').write_text(json.dumps(dict(config=vars(args), metrics=rows, loss_history=loss_history, epoch_seconds=trainer.epoch_times), indent=2))


if __name__ == '__main__':
    main()
