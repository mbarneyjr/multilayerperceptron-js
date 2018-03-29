const chai = require('chai');
const expect = chai.expect;
const MLP = require('../source/multilayer-perceptron');
const ActivationFunction = MLP.ActivationFunction;
const MultiLayerPerceptron = MLP.MultiLayerPerceptron;

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
})
