import { Component } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import { CharacterTable } from './character-table';
import { generateData, convertDataToTensors } from './utils';

const CHARS = '0123456789+ ';
const DIGITS = 2;
const TRAINING_SIZE = 5000;
const RNN_TYPE = 'SimpleRNN';
const RNN_LAYERS = 1;
const RNN_LAYER_SIZE = 128;
const BATCH_SIZE = 128;
const TRAIN_ITERATIONS = 100;
const COUNT_OF_TESTS = 20;

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  charTable: CharacterTable;
  model: tf.Sequential;
  trainData;
  testData;
  trainXs;
  trainYs;
  testXs;
  testYs;
  testXsForDisplay;
  logMessage: string;

  lossValues = [];
  accuracyValues = [];
  examplesPerSecValues = [];

  displayData: { text: string; correct: boolean }[] = [];

  constructor() {
    this.charTable = new CharacterTable(CHARS);
    const data = generateData(DIGITS, TRAINING_SIZE, false);
    const split = Math.floor(TRAINING_SIZE * 0.9);
    this.trainData = data.slice(0, split);
    this.testData = data.slice(split);
    [this.trainXs, this.trainYs] = convertDataToTensors(
      this.trainData,
      this.charTable,
      DIGITS
    );
    [this.testXs, this.testYs] = convertDataToTensors(
      this.testData,
      this.charTable,
      DIGITS
    );
    this.model = this.createAndCompileModel(
      RNN_LAYERS,
      RNN_LAYER_SIZE,
      RNN_TYPE,
      DIGITS,
      CHARS.length
    );
  }

  async train() {
    this.lossValues = [];
    this.accuracyValues = [];
    this.examplesPerSecValues = [];

    for (let i = 0; i < TRAIN_ITERATIONS; ++i) {
      const beginMs = performance.now();
      const history = await this.model.fit(this.trainXs, this.trainYs, {
        epochs: 1,
        batchSize: BATCH_SIZE,
        validationData: [this.testXs, this.testYs],
        yieldEvery: 'epoch'
      });
      const elapsedMs = performance.now() - beginMs;
      const examplesPerSec = this.testXs.shape[0] / (elapsedMs / 1000);
      const trainLoss = history.history['loss'][0] as number;
      const trainAccuracy = history.history['acc'][0] as number;
      const valLoss = history.history['val_loss'][0] as number;
      const valAccuracy = history.history['val_acc'][0] as number;

      this.log(
        `Iteration ${i}: train loss = ${trainLoss.toFixed(6)}; ` +
          `train accuracy = ${trainAccuracy.toFixed(6)}; ` +
          `validation loss = ${valLoss.toFixed(6)}; ` +
          `validation accuracy = ${valAccuracy.toFixed(6)} ` +
          `(${examplesPerSec.toFixed(1)} examples/s)`
      );

      this.lossValues.push({ epoch: i, loss: trainLoss, set: 'train' });
      this.lossValues.push({ epoch: i, loss: valLoss, set: 'validation' });
      this.accuracyValues.push({
        epoch: i,
        accuracy: trainAccuracy,
        set: 'train'
      });
      this.accuracyValues.push({
        epoch: i,
        accuracy: valAccuracy,
        set: 'validation'
      });
      this.examplesPerSecValues.push({
        epoch: i,
        'examples/s': examplesPerSec
      });
    }
  }

  test() {
    if (!this.testXsForDisplay) {
      this.testXsForDisplay = this.testXs.slice(
        [0, 0, 0],
        [COUNT_OF_TESTS, this.testXs.shape[1], this.testXs.shape[2]]
      );
    }

    this.displayData = [];
    tf.tidy(() => {
      const predictOut = this.model.predict(this.testXsForDisplay) as tf.Tensor;
      for (let k = 0; k < COUNT_OF_TESTS; ++k) {
        const scores = predictOut
          .slice([k, 0, 0], [1, predictOut.shape[1], predictOut.shape[2]])
          .as2D(predictOut.shape[1], predictOut.shape[2]);
        const decoded = this.charTable.decode(scores);
        this.displayData.push({
          text: this.testData[k][0] + ' = ' + decoded,
          correct: this.testData[k][1].trim() === decoded.trim()
        });
      }
    });
  }

  log(message: string) {
    this.logMessage = message;
  }

  createAndCompileModel(
    layers: number,
    hiddenSize: number,
    rnnType: 'SimpleRNN' | 'GRU' | 'LSTM',
    digits: number,
    vocabularySize: number
  ) {
    const maxLen = digits + 1 + digits;

    const model = tf.sequential();
    switch (rnnType) {
      case 'SimpleRNN':
        model.add(
          tf.layers.simpleRNN({
            units: hiddenSize,
            recurrentInitializer: 'glorotNormal',
            inputShape: [maxLen, vocabularySize]
          })
        );
        break;
      case 'GRU':
        model.add(
          tf.layers.gru({
            units: hiddenSize,
            recurrentInitializer: 'glorotNormal',
            inputShape: [maxLen, vocabularySize]
          })
        );
        break;
      case 'LSTM':
        model.add(
          tf.layers.lstm({
            units: hiddenSize,
            recurrentInitializer: 'glorotNormal',
            inputShape: [maxLen, vocabularySize]
          })
        );
        break;
      default:
        throw new Error(`Unsupported RNN type: '${rnnType}'`);
    }
    model.add(tf.layers.repeatVector({ n: digits + 1 }));
    switch (rnnType) {
      case 'SimpleRNN':
        model.add(
          tf.layers.simpleRNN({
            units: hiddenSize,
            recurrentInitializer: 'glorotNormal',
            returnSequences: true
          })
        );
        break;
      case 'GRU':
        model.add(
          tf.layers.gru({
            units: hiddenSize,
            recurrentInitializer: 'glorotNormal',
            returnSequences: true
          })
        );
        break;
      case 'LSTM':
        model.add(
          tf.layers.lstm({
            units: hiddenSize,
            recurrentInitializer: 'glorotNormal',
            returnSequences: true
          })
        );
        break;
      default:
        throw new Error(`Unsupported RNN type: '${rnnType}'`);
    }
    model.add(
      tf.layers.timeDistributed({
        layer: tf.layers.dense({ units: vocabularySize })
      })
    );
    model.add(tf.layers.activation({ activation: 'softmax' }));
    model.compile({
      loss: 'categoricalCrossentropy',
      optimizer: 'adam',
      metrics: ['accuracy']
    });
    return model;
  }
}
