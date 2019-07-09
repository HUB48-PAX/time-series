import matplotlib.pyplot as plt
import numpy as np
import torch

def random_uniform(min_, max_):
    return torch.FloatTensor([1]).uniform_(min_, max_)

class SineModule(torch.nn.Module):
    def __init__(self, A = None, omega = None, phi = None, B = None):
        super().__init__()

        A = random_uniform(0, 1) if A is None else A
        omega = random_uniform(0, 1) if omega is None else omega
        phi = random_uniform(0, 1) if phi is None else phi
        B = random_uniform(0, 1) if B is None else B

        self.A = torch.nn.Parameter(A)
        self.omega = torch.nn.Parameter(omega)
        self.phi = torch.nn.Parameter(phi)
        self.B = torch.nn.Parameter(B)

    def forward(self, x):
        return self.A * torch.sin(self.omega * x + self.phi) + self.B
        

class TimeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.functions = torch.nn.ModuleList()

    def new_function(self):
        function = SineModule()
        self.functions.append(function)

    def forward(self, x):
        result = 0
        for f in self.functions:
            result = result + f(x)

        return result

class EventFilter(torch.nn.Module):
    def __init__(self, threshold = None):
        super().__init__()

        threshold = random_uniform(0, 1) if threshold is None else threshold
        self.threshold = threshold
    
    def forward(self, x):
        value = x - self.threshold

        # 1 if x > threshold, 0 if x == threshold,
        # -1 if x < threshold
        sign = torch.sign(value)

        return (sign + 1) / 2

def run():
    input_data = []
    output_data = []

    # Increasing this number will favour event
    # answers
    event_weight = 40

    # Generate fake data
    for i in range(100):
        if i % 10 == 0 and i % 3 != 0:
            input_data += [i] * event_weight
            output_data += [1] * event_weight
        else:
            input_data.append(i)
            output_data.append(0)

    input_data = torch.tensor(input_data, dtype=torch.float)
    output_data = torch.tensor(output_data, dtype=torch.float)

    learning_rate = 0.004

    max_steps = 1500

    error = np.inf

    function_count = 10

    # Mean Absolute Error
    criterion = torch.nn.L1Loss()

    time_model = TimeModel()

    for _ in range(function_count):
        time_model.new_function()
    
    optimiser = torch.optim.Adam(time_model.parameters(), lr=learning_rate)
    
    # Sine wave Regression
    for i in range(max_steps):
        # Clear the gradients (important)
        optimiser.zero_grad()
        output = time_model(input_data)

        loss = criterion(output, output_data)
        loss.backward()

        optimiser.step()
        print('Sub-Iteration {}, Loss: {}'.format(i, loss))

    # Threshold regression
    # We cannot use autograd because the
    # gradient of the filter is always 0

    min_ = torch.min(output).detach().numpy()
    max_ = torch.max(output).detach().numpy()

    min_ = min_ - (max_ - min_) * 0.01
    max_ = max_ + (max_ - min_) * 0.01

    min_error = np.inf
    best_threshold = None

    for threshold in np.linspace(min_, max_, num=max_steps):
        event_filter = EventFilter(threshold=threshold)

        final_model = torch.nn.Sequential(time_model, event_filter)
        output = final_model(input_data)
        error = criterion(output, output_data).detach().numpy()
        
        if error < min_error:
            min_error = error
            best_threshold = threshold

    print('Best Error: {}'.format(min_error))

    final_model = torch.nn.Sequential(
        time_model,
        EventFilter(threshold=best_threshold)
    )


    time_output = time_model(input_data).detach().numpy() - best_threshold
    final_output = final_model(input_data).detach().numpy()

    input_data = input_data.numpy()
    output_data = output_data.numpy()


    plt.scatter(input_data[output_data == 1], [1] * np.count_nonzero(output_data), c='g')
    plt.plot(input_data, time_output, 'y--')
    plt.scatter(input_data[final_output == 1], [0] * np.count_nonzero(final_output), c='b')
    plt.show()

if __name__ == '__main__':
    run()