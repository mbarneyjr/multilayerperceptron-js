const sinon = require('sinon');
const chai = require('chai');
const MLP = require('../source/multilayer-perceptron');
const MATRIX = require('../source/matrix');

const expect = chai.expect;
const ActivationFunction = MLP.ActivationFunction;
const MultiLayerPerceptron = MLP.MultiLayerPerceptron;
const Matrix = MATRIX.Matrix;

describe('ActivationFunction Tests', () => {
  describe('#constructor', () => {
    it('should return an object with the function and derivative', () => {
      const func = x => x**2;
      const derv = x => 2*x;
      const result = new ActivationFunction(func, derv);
      expect(result.function(9)).to.equal(81);
      expect(result.derivative(9)).to.equal(18);
    });
    it('should check to see if parameters are functions', () => {
      expect(() => new ActivationFunction(1, '')).to.throw(Error);
    });
  });
});

describe('MultiLayerPerceptron Tests', () => {
  describe('#constructor', () => {
    it('should initialize an empty multilayer perceptron with the given input dimension', () => {
      const options = { inputDimension: 3 };
      const result = new MultiLayerPerceptron(options);
      expect(result.weightArray).to.deep.equal([]);
      expect(result.biasArray).to.deep.equal([]);
      expect(result.activationFunctions).to.deep.equal([]);
      expect(result.inputDimension).to.equal(options.inputDimension);
    });
    it('should only accept a positive non-zero input dimension', () => {
      expect(() => new MultiLayerPerceptron({inputDimension: -1})).to.throw(Error);
      expect(() => new MultiLayerPerceptron({inputDimension: 0})).to.throw(Error);
    });
  });
  describe('#addLayer', () => {
    it('should add to weights, biases, and activations', () => {
      const layer = {
        nodes: 3,
        activation: new ActivationFunction(x => x**2, x => 2*x)
      };
      const mlp = new MultiLayerPerceptron({inputDimension: 2});
      mlp.addLayer(layer);
      expect(mlp.weightArray.length).to.equal(1);
      expect(mlp.biasArray.length).to.equal(1);
      expect(mlp.activationFunctions.length).to.equal(1);
    });
    it('should use the previous layer\'s dimension', () => {
      const layer1 = {
        nodes: 3,
        activation: new ActivationFunction(x => x**2, x => 2*x)
      };
      const layer2 = {
        nodes: 4,
        activation: new ActivationFunction(x => x**2, x => 2*x)
      };
      const mlp = new MultiLayerPerceptron({inputDimension: 2});
      mlp.addLayer(layer1);
      mlp.addLayer(layer2);
      expect(mlp.weightArray[1].columns).to.equal(mlp.weightArray[0].rows);
    });
    it('should use input dimension when adding the first layer', () => {
      const layer = {
        nodes: 3,
        activation: new ActivationFunction(x => x**2, x => 2*x)
      };
      const mlp = new MultiLayerPerceptron({inputDimension: 2});
      mlp.addLayer(layer);
      expect(mlp.weightArray[0].columns).to.equal(mlp.inputDimension);
    });
    it('should only accept positive non-zero dimensions', () => {
      const activation = new ActivationFunction(x => x**2, x => 2*x);
      const mlp = new MultiLayerPerceptron({inputDimension: 2});
      expect(() => mlp.addLayer({nodes: -1, activation: activation})).to.throw(Error);
      expect(() => mlp.addLayer({nodes: 0, activation: activation})).to.throw(Error);
    });
    it('should make sure input nodes and activation functions are given', () => {
      const mlp = new MultiLayerPerceptron({inputDimension: 2});
      expect(() => mlp.addLayer({})).to.throw(Error);
    });
    it('should return the object itself', () => {
      const layer = {
        nodes: 3,
        activation: new ActivationFunction(x => x**2, x => 2*x)
      };
      const mlp = new MultiLayerPerceptron({inputDimension: 2});
      const result = mlp.addLayer(layer);
      expect(result).to.equal(mlp);
    });
  });
  describe('#randomizeWeights', () => {
    it('should randomize the weights between lower and upper bounds', () => {
      const activation = new ActivationFunction(x => x**2, x => 2*x);
      const lower = 13, upper = 17;
      const mlp = new MultiLayerPerceptron({inputDimension: 2})
        .addLayer({nodes: 2, activation: activation});
      for (let i = 0; i < 1000; i++) {
        mlp.randomizeWeights(lower, upper);
        mlp.weightArray.forEach(weights =>
          weights.data.forEach(row =>
            row.forEach(element => 
              expect(lower <= element && element <= upper).to.be.true)));
      }
    });
    it('should default to -1 and 1 by default when not given bounds', () => {
      const activation = new ActivationFunction(x => x**2, x => 2*x);
      const mlp = new MultiLayerPerceptron({inputDimension: 2})
        .addLayer({nodes: 2, activation: activation});
      for (let i = 0; i < 1000; i++) {
        mlp.randomizeWeights();
        mlp.weightArray.forEach(weights =>
          weights.data.forEach(row =>
            row.forEach(element => 
              expect(-1 <= element && element <= 1).to.be.true)));
      }
    });
    it('should only accept upper bound larger than or equal to lower bound', () => {
      const activation = new ActivationFunction(x => x**2, x => 2*x);
      const lower = 13, upper = 12;
      const mlp = new MultiLayerPerceptron({inputDimension: 2})
        .addLayer({nodes: 2, activation: activation});
      expect(() => mlp.randomizeWeights(lower, upper)).to.throw(Error);
    });
    it('should return the object itself', () => {
      const activation = new ActivationFunction(x => x**2, x => 2*x);
      const lower = 13, upper = 15;
      const mlp = new MultiLayerPerceptron({inputDimension: 2})
        .addLayer({nodes: 2, activation: activation});
      const result = mlp.randomizeWeights(lower, upper);
      expect(result).to.equal(mlp);
    });
  });
  describe('#predict', () => {
    it('should make predictions returned as an Array', () => {
      const activation = new ActivationFunction(x => x, x => 1);
      const mlp = new MultiLayerPerceptron({inputDimension: 2})
        .addLayer({nodes: 2, activation: activation})
        .addLayer({nodes: 2, activation: activation})
        .randomizeWeights(1, 1);
      const result = mlp.predict([1, 1]);
      expect(result.prediction).to.deep.equal([7, 7]);
    });
    it('should return state (activations) of the network for the prediction', () => {
      const activation = new ActivationFunction(x => x, x => 1);
      const mlp = new MultiLayerPerceptron({inputDimension: 2})
        .addLayer({nodes: 2, activation: activation})
        .addLayer({nodes: 2, activation: activation})
        .randomizeWeights(1, 1);
      const result = mlp.predict([1, 1]);
      const expected = [
        new Matrix(2, 1).randomize(1, 1),
        new Matrix(2, 1).randomize(3, 3),
        new Matrix(2, 1).randomize(7, 7)
      ]
      expect(result.activations).to.deep.equal(expected);
    });
    it('should make sure the input fits in the network', () => {
      const activation = new ActivationFunction(x => x, x => 1);
      const mlp = new MultiLayerPerceptron({inputDimension: 2})
        .addLayer({nodes: 2, activation: activation})
        .addLayer({nodes: 2, activation: activation})
        .randomizeWeights(1, 1);
      expect(() => mlp.predict([1, 1, 1])).to.throw(Error);
    });
  });
  describe('#trainIteration', () => {
    it('should become more accurate', () => {
      const input = [0, 1];
      const target = [1];
      const activation = new ActivationFunction(x => x, x => 1);
      const mlp = new MultiLayerPerceptron({inputDimension: input.length})
        .addLayer({nodes: target.length, activation: activation})
        .randomizeWeights();
      const oldPrediction = mlp.predict(input);
      let oldError = 0;
      for (let i = 0; i < target.length; i++) {
        oldError += Math.abs(target[i] - oldPrediction.prediction[i]);
      }
      mlp.trainIteration(input, target, 0.1);
      const newPrediction = mlp.predict(input);
      let newError = 0;
      for (let i = 0; i < target.length; i++) {
        newError += Math.abs(target[i] - newPrediction.prediction[i]);
      }
      expect(newError).to.be.lessThan(oldError);
    });
    it('should make sure the inputs and targets fit', () => {
      const activation = new ActivationFunction(x => x, x => 1);
      const mlp = new MultiLayerPerceptron({inputDimension: 2})
        .addLayer({nodes: 1, activation: activation})
        .randomizeWeights();
      expect(() => mlp.trainIteration([1, 1, 1], [1], 0.1)).to.throw(Error);
      expect(() => mlp.trainIteration([1, 1], [1, 1, 1], 0.1)).to.throw(Error);
      expect(() => mlp.trainIteration([1, 1, 1], [1, 1, 1], 0.1)).to.throw(Error);
    });
    it('should only accept a positive learning rate', () => {
      const input = [0, 1];
      const target = [1];
      const activation = new ActivationFunction(x => x, x => 1);
      const mlp = new MultiLayerPerceptron({inputDimension: input.length})
        .addLayer({nodes: target.length, activation: activation})
        .randomizeWeights();
      expect(() => mlp.trainIteration(input, target, -0.1)).to.throw(Error);
      expect(() => mlp.trainIteration(input, target, 0)).to.throw(Error);
    });
  });
  describe('#train', () => {
    it('should run train-iterate for each epoch for the given number of epochs', () => {
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
      const activation = new ActivationFunction(x => x, x => 1);
      const mlp = new MultiLayerPerceptron({inputDimension: 2})
        .addLayer({nodes: 1, activation: activation})
        .randomizeWeights();
      let trainIterationCount = 0;
      let fakeTrainIterate = () => trainIterationCount++;
      mlp.trainIteration = fakeTrainIterate;
      const numEpochs = 10000;
      mlp.train({
        trainData: dataset.inputs,
        trainLabels: dataset.targets,
        validationData: dataset.inputs,
        validationLabels: dataset.targets,
        numEpochs: numEpochs,
        learningRate: 0.1,
        verbose: false
      });
      expect(trainIterationCount).to.equal(dataset.inputs.length * numEpochs);
    });
    it('should validate train data/labels length and validation data/labels length', () => {
      const activation = new ActivationFunction(x => x, x => 1);
      const mlp = new MultiLayerPerceptron({inputDimension: 1})
        .addLayer({nodes: 1, activation: activation})
        .randomizeWeights();
      let trainIterationCount = 0;
      let fakeTrainIterate = () => trainIterationCount++;
      mlp.trainIteration = fakeTrainIterate;
      const numEpochs = 10000;
      const trainOptions = {
        trainData: [[1], [2]],
        trainLabels: [[1]],
        validationData: [[1], [2]],
        validationLabels: [[1]],
        numEpochs: numEpochs,
        learningRate: 0.1,
        verbose: false
      };
      expect(() => mlp.train(trainOptions)).to.throw(Error);
    });
    it('should return the object itself', () => {
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
      const activation = new ActivationFunction(x => x, x => 1);
      const mlp = new MultiLayerPerceptron({inputDimension: 2})
        .addLayer({nodes: 1, activation: activation})
        .randomizeWeights();
      const result = mlp.train({
        trainData: dataset.inputs,
        trainLabels: dataset.targets,
        validationData: dataset.inputs,
        validationLabels: dataset.targets,
        numEpochs: 100,
        learningRate: 0.1,
        verbose: false
      });
      expect(result).to.equal(mlp);
    });
    it('should randomize the train dataset each epoch', () => {
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
      const activation = new ActivationFunction(x => x, x => 1);
      const mlp = new MultiLayerPerceptron({inputDimension: 2})
        .addLayer({nodes: 1, activation: activation})
        .randomizeWeights();
      let trainIterationData = [];
      let fakeTrainIterate = (input, target) => trainIterationData.push(input);
      mlp.trainIteration = fakeTrainIterate;
      mlp.train({
        trainData: dataset.inputs,
        trainLabels: dataset.targets,
        validationData: dataset.inputs,
        validationLabels: dataset.targets,
        numEpochs: 1,
        learningRate: 0.1,
        verbose: false
      });
      expect(trainIterationData).to.not.deep.equal(dataset.inputs);
    });
  });
  describe('#evaluate', () => {
    it('should sum the error from running predictions on the given dataset', () => {
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
      const activation = new ActivationFunction(x => x, x => 1);
      const mlp = new MultiLayerPerceptron({inputDimension: 2})
        .addLayer({nodes: 1, activation: activation})
        .randomizeWeights(1, 1);
      const result = mlp.evaluate(dataset.inputs, dataset.targets)
      expect(result).to.equal(6);
    });
    it('should verify data/labels length', () => {
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
          [1]
        ]
      };
      const activation = new ActivationFunction(x => x, x => 1);
      const mlp = new MultiLayerPerceptron({inputDimension: 2})
        .addLayer({nodes: 1, activation: activation})
        .randomizeWeights(1, 1);
      expect(() => mlp.evaluate(dataset.inputs, dataset.targets)).to.throw(Error);
    });
  });
  describe('#saveWeights', () => {
    it('should save the weights and biases', () => {
      const fsWriteFileSync = sinon.stub(require('fs'), 'writeFileSync');
      const activation = new ActivationFunction(x => x, x => 1);
      const mlp = new MultiLayerPerceptron({inputDimension: 2})
        .addLayer({nodes: 1, activation: activation})
        .randomizeWeights(1, 1);
      mlp.saveWeights('./file.json');
      expect(fsWriteFileSync.calledWith('./file.json')).to.be.true;
      fsWriteFileSync.restore();
    });
    it('should return the object itself', () => {
      const fsWriteFileSync = sinon.stub(require('fs'), 'writeFileSync');
      const activation = new ActivationFunction(x => x, x => 1);
      const mlp = new MultiLayerPerceptron({inputDimension: 2})
        .addLayer({nodes: 1, activation: activation})
        .randomizeWeights(1, 1);
      const result = mlp.saveWeights('./file.json');
      expect(result).to.equal(mlp);
      fsWriteFileSync.restore();
    });
  });
  describe('#loadWeights', () => {
    it('should load the weights and biases', () => {
      const fsReadFileSync = sinon.stub(require('fs'), 'readFileSync');
      const fsExistsSync = sinon.stub(require('fs'), 'existsSync');
      fsExistsSync.returns(true);
      fsReadFileSync.returns(JSON.stringify({
        "weights": [
          [
            [
              1,
              1
            ]
          ]
        ],
        "biases": [
          [
            [
              1
            ]
          ]
        ]
      }, null, 2));
      const activation = new ActivationFunction(x => x, x => 1);
      const mlp = new MultiLayerPerceptron({inputDimension: 2})
        .addLayer({nodes: 1, activation: activation})
        .randomizeWeights(2, 2);
      mlp.loadWeights('./file.json');
      mlp.weightArray.forEach(weights =>
        weights.data.forEach(row =>
          row.forEach(element =>
            expect(element).to.equal(1))));
      fsReadFileSync.restore();
      fsExistsSync.restore();
    });
    it('should not update the network if the file doesn\'t exist', () => {
      const fsReadFileSync = sinon.stub(require('fs'), 'readFileSync');
      const fsExistsSync = sinon.stub(require('fs'), 'existsSync');
      fsExistsSync.returns(false);
      fsReadFileSync.returns(JSON.stringify({
        "weights": [
          [
            [
              1,
              1
            ]
          ]
        ],
        "biases": [
          [
            [
              1
            ]
          ]
        ]
      }, null, 2));
      const activation = new ActivationFunction(x => x, x => 1);
      const mlp = new MultiLayerPerceptron({inputDimension: 2})
        .addLayer({nodes: 1, activation: activation})
        .randomizeWeights(2, 2);
      mlp.loadWeights('./file.json');
      mlp.weightArray.forEach(weights =>
        weights.data.forEach(row =>
          row.forEach(element =>
            expect(element).to.equal(2))));
      fsReadFileSync.restore();
      fsExistsSync.restore();
    });
    it('should return the object itself', () => {
      const fsReadFileSync = sinon.stub(require('fs'), 'readFileSync');
      const fsExistsSync = sinon.stub(require('fs'), 'existsSync');
      const activation = new ActivationFunction(x => x, x => 1);
      const mlp = new MultiLayerPerceptron({inputDimension: 2})
        .addLayer({nodes: 1, activation: activation})
        .randomizeWeights(1, 1);
      const result = mlp.loadWeights('./file.json');
      expect(result).to.equal(mlp);
      fsReadFileSync.restore();
      fsExistsSync.restore();
    });
  });
});
