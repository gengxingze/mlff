import torch
import torch.nn as nn
from pygad.torchga import torchga
import pygad
# test gad

def fitness_func(ga_instance, solution, sol_idx):
    global data_inputs, data_outputs, torch_ga, model, loss_function

    predictions = pygad.torchga.predict(model=model,
                                        solution=solution,
                                        data=data_inputs)

    abs_error = loss_function(predictions[0], data_outputs[0]).detach().numpy()


    solution_fitness = 1.0 / abs_error

    return solution_fitness

def on_generation(ga_instance):
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {ga_instance.best_solution()[1]}")

# Create the PyTorch model.
input_layer = torch.nn.Linear(3, 5)
relu_layer = torch.nn.ReLU()
output_layer = torch.nn.Linear(5, 1)


class model1(nn.Module):
    def __init__(self):
        super(model1, self).__init__()
        self.fc1 = nn.Linear(25, 1)
        self.fc2 = nn.Linear(50, 100)
        self.fc3 = nn.Linear(100, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        y = 2 * x
        return x, y
model = model1()
# print(model)

# Create an instance of the pygad.torchga.TorchGA class to build the initial population.
torch_ga = torchga.TorchGA(model=model,
                           num_solutions=50)

loss_function = torch.nn.MSELoss()

# Data inputs
data_inputs = torch.rand(100, 25)


# Data outputs
data_outputs = [(2 * data_inputs).sum(-1), 2 * (2 * data_inputs).sum(-1)]

# Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/pygad.html#pygad-ga-class
num_generations = 2500 # Number of generations.
num_parents_mating = 10 # Number of solutions to be selected as parents in the mating pool.
initial_population = torch_ga.population_weights # Initial population of network weights

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       on_generation=on_generation)

ga_instance.run()

# After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Fitness value of the best solution = {solution_fitness}")
print(f"Index of the best solution : {solution_idx}")

# Make predictions based on the best solution.
predictions = pygad.torchga.predict(model=model,
                                    solution=solution,
                                    data=data_inputs)
# print("Predictions : \n", predictions.detach().numpy())
#
# abs_error = loss_function(predictions, data_outputs)
# print("Absolute Error : ", abs_error.detach().numpy())