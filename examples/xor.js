const { MultiLayerPerceptron, ActivationFunction } = require('../source/multilayer-perceptron');

let sigmoid = new ActivationFunction(
  x => 1 / (1 + Math.exp(-x)), // sigmoid
  y => y * (1 - y) // dsigmoid
);
let tanh = new ActivationFunction(
  x => Math.tanh(x),
  y => 1 - (y * y)
);

let mlp = new MultiLayerPerceptron({inputDimension: 2})
  .addLayer({nodes: 2, activation: sigmoid})
  .addLayer({nodes: 1, activation: sigmoid})
  .randomizeWeights()
  .loadWeights('xor.json');

let dataset = {
  inputs: [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
  ],
  targets: [
    [0],
    [1],
    [1],
    [0]
  ]
};

mlp.train({
  trainData: dataset.inputs,
  trainLabels: dataset.targets,
  validationData: dataset.inputs,
  validationLabels: dataset.targets,
  numEpochs: 100000,
  learningRate: 0.1,
  verbose: true
})
  .saveWeights('xor.json');

dataset.inputs.forEach(input => {
  console.log(`${input} => ${mlp.predict(input).prediction}`);
})
