import time
"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch


def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class Network(minitorch.Module):
    def __init__(self, hidden):
        super().__init__()
        self.layer1 = Linear(2, hidden)
        self.layer2 = Linear(hidden, hidden)
        self.layer3 = Linear(hidden, 1)

    def forward(self, x):
        return self.layer3(self.layer2(self.layer1(x).relu()).relu()).sigmoid()


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)

    def forward(self, x):
        batch, width = x.shape
        out_size = self.weights.value.shape[1]
        return (x.view(batch, width, 1) * self.weights.value.view(1, width, out_size)).sum(1).view(batch, out_size) + self.bias.value


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        self.epoch_times = []
        for epoch in range(1, self.max_epochs + 1):
            epoch_start = time.perf_counter()
            total_loss = 0.0
            correct = 0
            optim.zero_grad()


            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)


            optim.step()
            self.epoch_times.append(time.perf_counter() - epoch_start)


            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)

