import { Component } from '@angular/core';
import {
  Sequential,
  sequential,
  layers,
  tensor2d,
  loadModel,
  Tensor
} from '@tensorflow/tfjs';

const data = {
  inputs: [[0, 0], [0, 1], [1, 0], [1, 1]],
  targets: [[0], [1], [1], [0]]
};

const hiddenLayer = layers.dense({
  units: 10,
  inputShape: [2],
  activation: 'sigmoid'
});
const outputLayer = layers.dense({
  units: 1,
  inputShape: [10],
  activation: 'sigmoid'
});

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  model: Sequential;

  logMessage: string;

  constructor() {}

  async train() {
    // Define a model for linear regression.
    this.model = sequential();
    this.model.add(hiddenLayer);
    this.model.add(outputLayer);

    // Prepare the model for training: Specify the loss and the optimizer.
    this.model.compile({ loss: 'meanSquaredError', optimizer: 'rmsprop' });

    // Generate some synthetic data for training.
    const xs = tensor2d(data.inputs);
    const ys = tensor2d(data.targets);

    for (let i = 1; i < 300; ++i) {
      const h = await this.model.fit(xs, ys, { epochs: 10 });
      this.log(`Loss after Epoch ${i} : ${h.history.loss[0]}`);
    }
  }

  test() {
    this.log(String(this.predict(0, 1)));
  }

  predict(x: number, y: number): number {
    return (this.model.predict(tensor2d([[x, y]])) as Tensor).dataSync()[0];
  }

  log(message: string) {
    this.logMessage = message;
  }
}
