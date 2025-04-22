import { shuffle } from "./utils.js";

export default class OptimizedLogisticRegression {
  constructor(inputSize, numClasses) {
    this.inputSize = inputSize;
    this.numClasses = numClasses;
    this.weights = null;
    this.biases = null;
    this.trained = false;
  }

  initialize() {
    this.weights = new Float32Array(this.numClasses * this.inputSize);
    this.biases = new Float32Array(this.numClasses);
    
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] = Math.random() * 0.01;
    }
    
    this.trained = false;
  }

  softmax(scores) {
    const max = Math.max(...scores);
    const exps = scores.map(s => Math.exp(s - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sum);
  }

  async train(data, epochs = 20, lr = 0.01, batchSize = 128, momentum = 0.9) {
    if (!this.weights) this.initialize();
    
    const processed = this.preprocess(data);
    const velocityW = new Float32Array(this.weights.length);
    const velocityB = new Float32Array(this.numClasses);
    
    const gradW = new Float32Array(this.weights.length);
    const gradB = new Float32Array(this.numClasses);
    const scores = new Float32Array(this.numClasses);
    const probs = new Float32Array(this.numClasses);
    
    const start = performance.now();

    for (let epoch = 0; epoch < epochs; epoch++) {
      const { correct, total } = this.processEpoch(
        processed, batchSize, lr, momentum, 
        velocityW, velocityB, gradW, gradB, scores, probs
      );
      
      this.updateStatus(`Epoch ${epoch + 1}: Accuracy = ${(correct / total * 100).toFixed(2)}%`);
      await new Promise(r => setTimeout(r, 0));
    }
    
    this.trained = true;
    this.updateStatus(`Training completed in ${((performance.now() - start)/1000).toFixed(2)}s`);
    return { weights: this.weights, biases: this.biases };
  }

  preprocess(data) {
    return data.map(({ x, label }) => ({
      x: x instanceof Float32Array ? x : new Float32Array(x),
      label
    }));
  }

  processEpoch(data, batchSize, lr, momentum, vW, vB, gradW, gradB, scores, probs) {
    shuffle(data);
    let correct = 0;

    for (let i = 0; i < data.length; i += batchSize) {
      const end = Math.min(i + batchSize, data.length);
      const size = end - i;
      
      gradW.fill(0);
      gradB.fill(0);
      
      let batchCorrect = 0;
      
      for (let j = i; j < end; j++) {
        const { x, label } = data[j];
        const { predicted } = this.forward(x, scores, probs);
        
        if (predicted === label) batchCorrect++;
        this.gradients(x, label, probs, gradW, gradB);
      }
      
      correct += batchCorrect;
      this.update(gradW, gradB, size, lr, momentum, vW, vB);
    }

    return { correct, total: data.length };
  }

  forward(x, scores, probs) {
    this.scores(x, scores);
    
    const max = Math.max(...scores);
    let sum = 0;
    
    for (let c = 0; c < this.numClasses; c++) {
      probs[c] = Math.exp(scores[c] - max);
      sum += probs[c];
    }
    
    for (let c = 0; c < this.numClasses; c++) {
      probs[c] /= sum;
    }
    
    let maxProb = probs[0];
    let predicted = 0;
    
    for (let c = 1; c < this.numClasses; c++) {
      if (probs[c] > maxProb) {
        maxProb = probs[c];
        predicted = c;
      }
    }
    
    return { probs, predicted };
  }

  scores(x, scores) {
    for (let c = 0; c < this.numClasses; c++) {
      let sum = this.biases[c];
      const offset = c * this.inputSize;
      
      let j = 0;
      for (; j < this.inputSize - 3; j += 4) {
        sum += this.weights[offset + j] * x[j] +
               this.weights[offset + j + 1] * x[j + 1] +
               this.weights[offset + j + 2] * x[j + 2] +
               this.weights[offset + j + 3] * x[j + 3];
      }
      
      for (; j < this.inputSize; j++) {
        sum += this.weights[offset + j] * x[j];
      }
      
      scores[c] = sum;
    }
  }

  gradients(x, label, probs, gradW, gradB) {
    for (let c = 0; c < this.numClasses; c++) {
      const error = probs[c] - (c === label ? 1 : 0);
      gradB[c] += error;
      
      const offset = c * this.inputSize;
      for (let j = 0; j < this.inputSize; j++) {
        gradW[offset + j] += error * x[j];
      }
    }
  }

  update(gradW, gradB, batchSize, lr, momentum, vW, vB) {
    const scale = lr / batchSize;
    
    for (let i = 0; i < this.weights.length; i++) {
      vW[i] = momentum * vW[i] + scale * gradW[i];
      this.weights[i] -= vW[i];
    }
    
    for (let c = 0; c < this.numClasses; c++) {
      vB[c] = momentum * vB[c] + scale * gradB[c];
      this.biases[c] -= vB[c];
    }
  }

  evaluate(data) {
    if (!this.trained) throw new Error("Model not trained");
    
    const scores = new Float32Array(this.numClasses);
    const probs = new Float32Array(this.numClasses);
    let correct = 0;
    
    for (const { x, label } of data) {
      const { predicted } = this.forward(x, scores, probs);
      if (predicted === label) correct++;
    }
    
    return correct / data.length;
  }

  predict(x) {
    if (!this.trained) throw new Error("Model not trained");
    const scores = new Float32Array(this.numClasses);
    const probs = new Float32Array(this.numClasses);
    return this.forward(x, scores, probs);
  }

  updateStatus(message) {
    console.log(message);
  }
}