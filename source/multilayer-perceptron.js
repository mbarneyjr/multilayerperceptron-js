const { Matrix } = require('./matrix');

String.prototype.center = function( width, padding ) {
  padding = padding || " ";
  padding = padding.substr( 0, 1 );
  if( this.length < width ) {
    var len		= width - this.length;
    var remain	= ( len % 2 == 0 ) ? "" : padding;
    var pads	= padding.repeat( parseInt( len / 2 ) );
    return pads + this + pads + remain;
  }
  else
    return this;
}

class ActivationFunction {
  constructor(func, derivative) {
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
  }

  randomizeWeights() {
    this.weightArray.forEach(weights => weights.randomize(-1, 1));
    this.biasArray.forEach(bias => bias.randomize(-1, 1));
  }

  predict(inputArray) {
    let input = Matrix.fromArray(inputArray);
    if (this.weightArray[0].columns !== input.rows) {
      throw Error('Prediction input does not fit in the network');
    }

    let sum = input;
    let activations = [];
    for (let i = 0; i < this.weightArray.length; i++) {
      // figure out the next layer's node values
      sum = Matrix.dot(this.weightArray[i], sum);
      sum.add(this.biasArray[i]);
      activations.push(sum);
      // run those values through the activation function
      sum.map(this.activationFunctions[i].function);
    }
    return {
      prediction: sum.toArray(),
      activations: activations
    }
  }

  trainIteration(input, target, learningRate) {
    let { prediction, activations } = this.predict(input, target);
    let gradients, weightDeltas, previousTransposed;
    let targets = Matrix.fromArray(target);
    let layerOutputs = Matrix.fromArray(prediction);
    let layerErrors = Matrix.subtract(targets, layerOutputs);
    for (let i = this.weightArray.length - 1; i > 0 ; i--) {
      // calculate gradient
      gradients = Matrix.map(layerOutputs, this.activationFunctions[i].derivative)
        .multiply(layerErrors)
        .multiply(learningRate)
      // calculate deltas
      previousTransposed = Matrix.transpose(activations[i-1]);
      weightDeltas = Matrix.dot(gradients, previousTransposed);
      // update the weights and biases
      this.weightArray[i].add(weightDeltas);
      this.biasArray[i].add(gradients);
      // calculate next layer's errors
      layerOutputs = activations[i-1];
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
      if (options.verbose)
        console.log(`Epoch ${epoch}; Error ${this.evaluate(options.validationData, options.validationLabels)}`);
    }
  }

  evaluate(dataInputs, dataLabels) {
    if (dataInputs.length !== dataLabels.length) {
      throw Error('You have to supply one label for each data item!')
    }
    let error = 0;
    for (let dataElement = 0; dataElement < dataInputs.length; dataElement++) {
      error += Math.abs(this.predict(dataInputs[dataElement]).prediction - dataLabels[dataElement]);
    }
    return error;
  }

  print(node='*') {
    node += ' ';
    let longestLayer = Math.max(
      ...this.weightArray.map(w => w.rows),
      this.inputDimension) * node.length;
    console.log(node.repeat(this.inputDimension).center(longestLayer + 1));
    this.weightArray.forEach(weight => {
      console.log(`\n${node.repeat(weight.rows).center(longestLayer + 1)}`);
    });
  }

  printWeights() {
    for (let i = 0; i < this.weightArray.length; i++) {
      console.log('weight for layer: ', i);
      this.weightArray[i].print();
      console.log('bias for layer: ', i);
      this.biasArray[i].print();
    }
  }

}

module.exports = {
  MultiLayerPerceptron,
  ActivationFunction
}
