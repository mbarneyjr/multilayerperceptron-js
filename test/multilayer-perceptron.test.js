const chai = require('chai');
const expect = chai.expect;
const MLP = require('../source/multilayer-perceptron');
const ActivationFunction = MLP.ActivationFunction;
const MultiLayerPerceptron = MLP.MultiLayerPerceptron;
const matrix = require('../source/matrix');
const Matrix = matrix.Matrix;

describe('ActivationFunction Tests', () => {
  describe('#constructor', () => {
    it('should verify parameters are functions', () => {
      expect(() => new ActivationFunction(1, '')).to.throw(Error);
    });
    it('should register function and derivative', () => {
      const y = x => x ** 2;
      const dy = x => 2 * x;
      const result = new ActivationFunction(y, dy);
      expect(result.function).to.equal(y);
      expect(result.derivative).to.equal(dy);
    });
  });
});

describe('MultiLayerPerceptron Tests', () => {
  describe('#constructor', () => {
    it('creates a MultiLayerPerceptron object', () => {
      const options = {inputDimension: 3};
      const result = new MultiLayerPerceptron(options);
      expect(result.weightArray).to.deep.equal([])
      expect(result.biasArray).to.deep.equal([])
      expect(result.activationFunctions).to.deep.equal([])
      expect(result.inputDimension).to.equal(3)
    });
  });
  describe('#addLayer', () => {
    it('correctly uses input dimension if adding first weighted layer', () => {
      const options = {inputDimension: 3};
      let func = new ActivationFunction(
        x => x ** 2,
        x => 2 * x
      );
      const mlp = new MultiLayerPerceptron(options)
        .addLayer({nodes:2, activation: func});
      expect(mlp.weightArray[0].rows).to.equal(2);
      expect(mlp.weightArray[0].columns).to.equal(3);
    });
    it('correctly adds weights, biases, and activation function', () => {
      const options = {inputDimension: 3};
      let func = new ActivationFunction(
        x => x ** 2,
        x => 2 * x
      );
      const mlp = new MultiLayerPerceptron(options)
        .addLayer({nodes:2, activation: func})
        .addLayer({nodes:3, activation: func})
        .addLayer({nodes:8, activation: func});
      expect(mlp.weightArray[0].rows).to.equal(2);
      expect(mlp.weightArray[0].columns).to.equal(3);
      expect(mlp.weightArray[1].rows).to.equal(3);
      expect(mlp.weightArray[1].columns).to.equal(2);
      expect(mlp.weightArray[2].rows).to.equal(8);
      expect(mlp.weightArray[2].columns).to.equal(3);
    });
  });
  describe('#randomizeWeights', () => {
    it('randomizes weights', () => {
      const options = {inputDimension: 3};
      let func = new ActivationFunction(
        x => x ** 2,
        x => 2 * x
      );
      const mlp = new MultiLayerPerceptron(options)
        .addLayer({nodes:2, activation: func})
      const oldWeightData = JSON.parse(JSON.stringify(mlp.weightArray[0].data));
      mlp.randomizeWeights();
      expect(mlp.weightArray[0].data).to.not.deep.equal(oldWeightData);
    });
    it('randomizes weights between -1 and 1 by default', () => {
      const options = {inputDimension: 3};
      let func = new ActivationFunction(
        x => x ** 2,
        x => 2 * x
      );
      const mlp = new MultiLayerPerceptron(options)
        .addLayer({nodes:2, activation: func})
        .randomizeWeights();
      mlp.weightArray.forEach(weights =>
        weights.data.forEach(row =>
          row.forEach(element =>
            expect(-1 < element && element < 1).to.be.true)));
    });
    it('randomizes weights between lower and upper bounds', () => {
      const options = {inputDimension: 3};
      const lower = 6;
      const upper = 7;
      let func = new ActivationFunction(
        x => x ** 2,
        x => 2 * x
      );
      const mlp = new MultiLayerPerceptron(options)
        .addLayer({nodes:2, activation: func})
        .randomizeWeights(lower, upper);
      mlp.weightArray.forEach(weights =>
        weights.data.forEach(row =>
          row.forEach(element =>
            expect(lower < element && element < upper).to.be.true)));
    });
    it('randomizes biases', () => {
      const options = {inputDimension: 3};
      let func = new ActivationFunction(
        x => x ** 2,
        x => 2 * x
      );
      const mlp = new MultiLayerPerceptron(options)
        .addLayer({nodes:2, activation: func})
      const oldBiasData = JSON.parse(JSON.stringify(mlp.biasArray[0].data));
      mlp.randomizeWeights();
      expect(mlp.biasArray[0].data).to.not.deep.equal(oldBiasData);
    });
    it('randomizes biases between -1 and 1 by default', () => {
      const options = {inputDimension: 3};
      let func = new ActivationFunction(
        x => x ** 2,
        x => 2 * x
      );
      const mlp = new MultiLayerPerceptron(options)
        .addLayer({nodes:2, activation: func})
        .randomizeWeights();
      mlp.biasArray.forEach(bias =>
        bias.data.forEach(row =>
          row.forEach(element =>
            expect(-1 < element && element < 1).to.be.true)));
    });
    it('randomizes biases between lower and upper bounds', () => {
      const options = {inputDimension: 3};
      const lower = 6;
      const upper = 7;
      let func = new ActivationFunction(
        x => x ** 2,
        x => 2 * x
      );
      const mlp = new MultiLayerPerceptron(options)
        .addLayer({nodes:2, activation: func})
        .randomizeWeights(lower, upper);
      mlp.biasArray.forEach(bias =>
        bias.data.forEach(row =>
          row.forEach(element =>
            expect(lower < element && element < upper).to.be.true)));
    });
  });
  describe('#predict', () => {
    it('correctly makes predictions', () => {
      const options = {inputDimension: 3};
      let func = new ActivationFunction(
        x => x,
        x => 1
      );
      const mlp = new MultiLayerPerceptron(options)
        .addLayer({nodes:2, activation: func})
        .randomizeWeights(1, 1);
      const result = mlp.predict([1, 1, 1]);
      expect(result.prediction).to.deep.equal([4, 4]);
    });
    it('returns the state (activations) of the network', () => {
      const options = {inputDimension: 3};
      let func = new ActivationFunction(
        x => x,
        x => 1
      );
      const mlp = new MultiLayerPerceptron(options)
        .addLayer({nodes:2, activation: func})
        .randomizeWeights(1, 1);
      const result = mlp.predict([1, 1, 1]);
      const expected = [new Matrix(2, 1)];
      expected[0].data = [[4], [4]];
      expect(result.activations).to.deep.equal(expected);
    });
  });
  describe('#trainIteration', () => {
    it('makes the network more accurate', () => {
      let sigmoid = new ActivationFunction(
        x => 1 / (1 + Math.exp(-x)), // sigmoid
        y => y * (1 - y) // dsigmoid
      );
      const input = [1, 0];
      const output = [1];
      const mlp = new MultiLayerPerceptron({inputDimension: 2})
        .addLayer({nodes:2, activation: sigmoid})
        .addLayer({nodes:1, activation: sigmoid})
        .randomizeWeights();
      const oldResult = mlp.predict(input);
      mlp.trainIteration(input, output, 0.1);
      const newResult = mlp.predict(input);
      expect(Math.abs(oldResult.prediction - output[0])).to.be.greaterThan(Math.abs(newResult.prediction - output[0]));
    });
    it('verifies input and target dimensions with the network', () => {
      let sigmoid = new ActivationFunction(
        x => 1 / (1 + Math.exp(-x)), // sigmoid
        y => y * (1 - y) // dsigmoid
      );
      const input = [1, 0, 0];
      const output = [1, 0];
      const mlp = new MultiLayerPerceptron({inputDimension: 2})
        .addLayer({nodes:2, activation: sigmoid})
        .addLayer({nodes:1, activation: sigmoid})
        .randomizeWeights();
      expect(() => mlp.trainIteration(input, output, 0.1)).to.throw(Error);
    });
    it('verifies the given learning rate is positive', () => {
      let sigmoid = new ActivationFunction(
        x => 1 / (1 + Math.exp(-x)), // sigmoid
        y => y * (1 - y) // dsigmoid
      );
      const input = [1, 0];
      const output = [1];
      const mlp = new MultiLayerPerceptron({inputDimension: 2})
        .addLayer({nodes:2, activation: sigmoid})
        .addLayer({nodes:1, activation: sigmoid})
        .randomizeWeights();
      expect(() => mlp.trainIteration(input, output, -0.1)).to.throw(Error);
    });
  });
  describe('#train', () => {
    it('verifies the dataset data and label array length', () => {
      let sigmoid = new ActivationFunction(
        x => 1 / (1 + Math.exp(-x)), // sigmoid
        y => y * (1 - y) // dsigmoid
      );
      const mlp = new MultiLayerPerceptron({inputDimension: 2})
        .addLayer({nodes:2, activation: sigmoid})
        .addLayer({nodes:1, activation: sigmoid})
        .randomizeWeights();
      const options = {
        trainData: [[1, 2, 3], [1, 2, 3]],
        trainLabels: [[1, 2], [1, 2]],
        validationData: [[1, 2, 3], [1, 2, 3]],
        validationLabels: [[1, 2], [1, 2]],
        numEpochs: 100,
        learningRate: 0.1,
        verbose: true
      };
      expect(() => mlp.train(options)).to.throw(Error);
    });
    it('back-propagates for each dataset sample for each epoch', () => {
      let sigmoid = new ActivationFunction(
        x => 1 / (1 + Math.exp(-x)), // sigmoid
        y => y * (1 - y) // dsigmoid
      );
      let trainIterationCount = 0;
      const fakeTrainIteration = () => {
        trainIterationCount++;
      }
      fakeTrainData = [[1, 2], [1, 2]];
      fakeTrainLabels = [[1], [1]];
      const numEpochs = 100;
      const mlp = new MultiLayerPerceptron({inputDimension: 2})
        .addLayer({nodes:2, activation: sigmoid})
        .addLayer({nodes:1, activation: sigmoid})
        .randomizeWeights();
      const options = {
        trainData: fakeTrainData,
        trainLabels: fakeTrainLabels,
        validationData: fakeTrainData,
        validationLabels: fakeTrainLabels,
        numEpochs: numEpochs,
        learningRate: 0.1,
        verbose: false
      };
      mlp.trainIteration = fakeTrainIteration;
      mlp.train(options);
      expect(trainIterationCount).to.equal(fakeTrainData.length * numEpochs);
    });
  });
  describe('#evaluate', () => {
    it('verifies dataset data and label array length', () => {
      let func = new ActivationFunction(
        x => x,
        x => 1
      );
      const input = [1, 1];
      const target = [1];
      const mlp = new MultiLayerPerceptron({inputDimension: 3})
        .addLayer({nodes:2, activation: func})
        .randomizeWeights(1, 1);
      expect(() => mlp.evaluate([input], [target])).to.throw(Error)
    });
    it('sums the errors of each prediction into one value', () => {
      let func = new ActivationFunction(
        x => x,
        x => 1
      );
      const input = [1, 1, 1];
      const target = [1, 1];
      const mlp = new MultiLayerPerceptron({inputDimension: 3})
        .addLayer({nodes:2, activation: func})
        .randomizeWeights(1, 1);
      const result = mlp.evaluate([input], [target]);
      expect(result).to.equal(6)
    });
  });
});
