const { Matrix } = require('./matrix');
const fs = require('fs');
const path = require('path');

class ActivationFunction {
  constructor(func, derivative) {
    if (typeof(func) != 'function' || typeof(derivative) != 'function') {
      throw Error('ActivationFunction requires two functions');
    }
    this.function = func;
    this.derivative = derivative;
  }
}

class MultiLayerPerceptron {
  constructor(options) {
    this.weightArray = [];
    this.biasArray = [];
    this.activationFunctions = [];
    this.inputDimension = options.inputDimension;
  }

  addLayer(layer) {
    let weights;
    if (this.weightArray.length === 0) {
      weights = new Matrix(layer.nodes, this.inputDimension);
    } else {
      weights = new Matrix(layer.nodes, this.weightArray[this.weightArray.length - 1].rows);
    }
    let biases = new Matrix(layer.nodes, 1);
    this.weightArray.push(weights);
    this.biasArray.push(biases);
    this.activationFunctions.push(layer.activation);
    return this;
  }

  randomizeWeights(lower, upper) {
    if (lower == undefined && upper == undefined) {
      this.weightArray.forEach(weights => weights.randomize(-1, 1));
      this.biasArray.forEach(bias => bias.randomize(-1, 1));
    } else {
      this.weightArray.forEach(weights => weights.randomize(lower, upper));
      this.biasArray.forEach(bias => bias.randomize(lower, upper));
    }
    return this;
  }

  predict(inputArray) {
    let input = Matrix.fromArray(inputArray);
    if (this.weightArray[0].columns !== input.rows) {
      throw Error('Prediction input does not fit in the network');
    }

    let sum = input;
    let activations = [];
    activations.push(input);
    for (let i = 0; i < this.weightArray.length; i++) {
      // figure out the next layer's node values
      sum = Matrix.dot(this.weightArray[i], sum);
      sum.add(this.biasArray[i]);
      activations.push(sum);
      // run those values through the activation function
      sum.map(this.activationFunctions[i].function);
    }
    return {
      prediction: Matrix.transpose(sum).toArray(),
      activations: activations
    }
  }

  trainIteration(input, target, learningRate) {
    if (input.length !== this.inputDimension || target.length !== this.weightArray[this.weightArray.length - 1].rows) {
      throw Error('Input and target output must match the dimensions of the network!');
    }
    if (learningRate <= 0) {
      throw Error('Learning rate must be greater than 0');
    }
    let { prediction, activations } = this.predict(input, target);
    let gradients, weightDeltas, previousTransposed;
    let targets = Matrix.fromArray(target);
    let layerOutputs = Matrix.fromArray(prediction);
    let layerErrors = Matrix.subtract(targets, layerOutputs);
    for (let i = this.weightArray.length - 1; i >= 0 ; i--) {
      // calculate gradient
      gradients = Matrix.map(layerOutputs, this.activationFunctions[i].derivative)
        .multiply(layerErrors)
        .multiply(learningRate)
      // calculate deltas
      previousTransposed = Matrix.transpose(activations[i]);
      weightDeltas = Matrix.dot(gradients, previousTransposed);
      // update the weights and biases
      this.weightArray[i].add(weightDeltas);
      this.biasArray[i].add(gradients);
      // calculate next layer's errors
      layerOutputs = activations[i];
      layerErrors = Matrix.dot(Matrix.transpose(this.weightArray[i]), layerErrors);
    }
  }

  train(options) {
    if (options.trainData.length !== options.trainLabels.length ||
        options.validationData.length !== options.validationLabels.length) {
        throw Error('You have to supply one label for each data item!')
      }
    for (let epoch = 1; epoch <= options.numEpochs; epoch++) {
      [...Array(options.trainData.length).keys()].sort(() => 0.5 - Math.random()).forEach(dataElement => {
        this.trainIteration(options.trainData[dataElement], options.trainLabels[dataElement], options.learningRate);
      })
      if (options.verbose) {
        console.log(`Epoch ${epoch}; ${this.evaluate(options.validationData, options.validationLabels)}`);
      }
    }
  }

  evaluate(dataInputs, dataLabels) {
    if (dataInputs.length !== dataLabels.length) {
      throw Error('You have to supply one label for each data item!')
    }
    let error = 0;
    let prediction, target;
    for (let dataElement = 0; dataElement < dataInputs.length; dataElement++) {
      prediction = this.predict(dataInputs[dataElement]).prediction;
      target = dataLabels[dataElement];
      for (let i = 0; i < prediction.length; i++) {
        error += Math.abs(prediction[i] - target[i]);
      }
    }
    return error
  }

  saveWeights(filepath) {
    const saveObject = {
      weights: [],
      biases: []
    };
    this.weightArray.forEach(weights =>
      saveObject.weights.push(weights.data));
    this.biasArray.forEach(bias =>
      saveObject.biases.push(bias.data));
    fs.writeFileSync(filepath, JSON.stringify(saveObject, null, 2), 'utf8');
  }

  loadWeights(filepath) {
    if (fs.existsSync(filepath)) {
      const loadObject = JSON.parse(fs.readFileSync(filepath, 'utf8'));
      for (let i = 0; i < loadObject.weights.length; i++) {
        this.weightArray[i].rows = loadObject.weights[i].length;
        this.weightArray[i].columns = loadObject.weights[i][0].length;
        this.weightArray[i].data = loadObject.weights[i];
        this.biasArray[i].rows = loadObject.biases[i].length;
        this.biasArray[i].columns = loadObject.biases[i][0].length;
        this.biasArray[i].data = loadObject.biases[i];
      }
    }
  }
}

module.exports = {
  MultiLayerPerceptron,
  ActivationFunction
}
