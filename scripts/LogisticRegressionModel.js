import { softmax, shuffle } from "./utils.js"

export default
class LogisticRegressionModel {
  constructor(inputSize = 784, numClasses = 10) {
    this.inputSize = inputSize;
    this.numClasses = numClasses;
    this.weights = null;
    this.biases = null;
    this.trained = false;
  }

  initializeParameters() {
    this.weights = Array(this.numClasses).fill().map(() =>
      Array(this.inputSize).fill().map(() => Math.random() * 0.01)
    );
    this.biases = Array(this.numClasses).fill(0);
    this.trained = false;
  }

  train(data, epochs = 20, learningRate = 0.01, batchSize = 128, momentum = 0.9) {
    if (!this.weights) this.initializeParameters();
    
    const vWeights = Array(this.numClasses).fill().map(() => Array(this.inputSize).fill(0));
    const vBiases = Array(this.numClasses).fill(0);

    for (let epoch = 0; epoch < epochs; epoch++) {
      const { correct, total } = this.processEpoch(data, batchSize, learningRate, momentum, vWeights, vBiases);
      this.updateStatus(`Epoch ${epoch + 1}: Accurate = ${(correct / total * 100).toFixed(2)}%`);
    }

    this.trained = true;
    return { weights: this.weights, biases: this.biases };
  }

  processEpoch(data, batchSize, learningRate, momentum, vWeights, vBiases) {
    shuffle(data);
    let correct = 0;
    let total = 0;

    for (let i = 0; i < data.length; i += batchSize) {
      const batch = data.slice(i, i + batchSize);
      const { gradW, gradB, batchCorrect } = this.processBatch(batch);
      
      correct += batchCorrect;
      total += batch.length;
      
      this.updateParameters(gradW, gradB, batch.length, learningRate, momentum, vWeights, vBiases);
    }

    return { correct, total };
  }

  processBatch(batch) {
    const gradW = Array(this.numClasses).fill().map(() => Array(this.inputSize).fill(0));
    const gradB = Array(this.numClasses).fill(0);
    let batchCorrect = 0;

    for (const sample of batch) {
      const { x, label } = sample;
      const { probs, predicted } = this.forwardPass(x);
      
      if (predicted === label) batchCorrect++;
      
      this.calculateGradients(x, label, probs, gradW, gradB);
    }

    return { gradW, gradB, batchCorrect };
  }

  forwardPass(x) {
    const scores = this.calculateScores(x);
    const probs = softmax(scores);
    const predicted = probs.indexOf(Math.max(...probs));
    return { probs, predicted };
  }

  calculateScores(x) {
    return this.weights.map((classWeights, c) => {
      return classWeights.reduce((sum, weight, j) => sum + weight * x[j], this.biases[c]);
    });
  }

  calculateGradients(x, label, probs, gradW, gradB) {
    for (let c = 0; c < this.numClasses; c++) {
      const target = (c === label) ? 1 : 0;
      const error = probs[c] - target;
      gradB[c] += error;
      
      for (let j = 0; j < this.inputSize; j++) {
        gradW[c][j] += error * x[j];
      }
    }
  }

  updateParameters(gradW, gradB, batchSize, learningRate, momentum, vWeights, vBiases) {
    for (let c = 0; c < this.numClasses; c++) {
      for (let j = 0; j < this.inputSize; j++) {
        vWeights[c][j] = momentum * vWeights[c][j] + learningRate * (gradW[c][j] / batchSize);
        this.weights[c][j] -= vWeights[c][j];
      }
      
      vBiases[c] = momentum * vBiases[c] + learningRate * (gradB[c] / batchSize);
      this.biases[c] -= vBiases[c];
    }
  }

  evaluate(data) {
    if (!this.trained) throw new Error("Model not trained");
    
    let correct = 0;
    for (const sample of data) {
      const { predicted } = this.forwardPass(sample.x);
      if (predicted === sample.label) correct++;
    }
    
    return correct / data.length;
  }

  predict(x) {
    if (!this.trained) throw new Error("Model not trained");
    return this.forwardPass(x);
  }

  updateStatus(message) {
    // To be implemented by the view controller
  }
}