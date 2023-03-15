from os.path import exists
from sys import stderr
import torch
import torch.nn as nn


CHECKPOINT_INTERVAL = 100
LEARNING_RATE = 1e-5


class ActiveInference(nn.Module):
    def __init__(self, energy_model, prediction_model):
        self.energy_model = energy_model
        self.prediction_model = prediction_model

    def forward(self, x, y, z, predicted=None):
        prediction_losses = []
        if predicted is not None:
            x_predicted = predicted[0]
            y_predicted = predicted[1]
            z_predicted = predicted[2]
            prediction_losses.append(((x_predicted - x) ** 2).mean())
            prediction_losses.append(((y_predicted - y) ** 2).mean())
            prediction_losses.append(((z_predicted - z) ** 2).mean())
        predictions = self.prediction_model(x, y, z)
        energies = self.energy_model(predictions)
        choice = torch.argmax(energies)
        z_predicted = predictions[choice][2]
        loss = ((z_predicted - z) ** 2).mean()
        for prediction_loss in prediction_losses:
            loss += prediction_loss
        return (predictions[choice], choice, energies, predictions, loss)


class EnergyModel(nn.Module):
    def __init__(self):
        pass

    def forward(self, x, y, z):
        # TODO: return (n, x, y, z) predictions
        pass


class PredictionModel(nn.Module):
    def __init__(self):
        pass

    def forward(self, predictions):
        # TODO: return n energies
        pass


def run_active_inference(energy_model, prediction_model, environment,
                         checkpoint=None,
                         max_epochs=None,
                         learning_rate=LEARNING_RATE,
                         checkpoint_interval=CHECKPOINT_INTERVAL):

    if checkpoint is not None and exists(checkpoint):
        state = torch.load(checkpoint)
        epoch = state.get('epoch', 0)
    else:
        state = {}
        epoch = 0

    model = ActiveInference(energy_model, prediction_model)
    optim = torch.optim.Adam(lr=learning_rate, params=model.parameters())
    (y, z) = environment.initialize()
    prediction = None

    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    if 'optim_state_dict' in state:
        optim.load_state_dict(state['optim_state_dict'])
    if 'prediction_state' in state:
        prediction = state['prediction_state']
        y = prediction[1]
        z = prediction[2]

    while max_epochs is None or epoch < max_epochs:
        epoch += 1
        optim.zero_grad()

        (x, y) = environment.interact(y)
        (prediction, _, _, _, loss) = model(x, y, z, prediction)

        loss.backward()
        optim.step()

        y = prediction[1]
        z = prediction[2]

        if checkpoint is not None and epoch % CHECKPOINT_INTERVAL == 0:
            state['loss'] = loss
            state['epoch'] = epoch
            state['model_state_dict'] = model.state_dict()
            state['optim_state_dict'] = optim.state_dict()
            state['prediction_state'] = prediction
            torch.save(state, checkpoint)
            print("Checkpoint written to {}".format(checkpoint), file=stderr)


def train():
    energy_model = EnergyModel()
    prediction_model = PredictionModel()
