const chai = require('chai');
const expect = chai.expect;
const matrix = require('../source/matrix');
const Matrix = matrix.Matrix;

describe('Matrix Tests', () => {
  describe('#constructor', () => {
    it('should create a matrix with the right dimensions', () => {
      const numRows = 4;
      const numColumns = 7;
      const result = new Matrix(numRows, numColumns);
      expect(result.rows).to.equal(numRows);
      expect(result.columns).to.equal(numColumns);
    });
    it('should initialize data to zeros', () => {
      const numRows = 1;
      const numColumns = 1;
      const result = new Matrix(numRows, numColumns);
      expect(result.data).to.deep.equal([[0]]);
    });
  });
  describe('#fromArray', () => {
    it('should create a matrix of the correct dimensions', () => {
      const array = [[1,2,3], [4, 5, 6]];
      const result = Matrix.fromArray(array);
      expect(result.rows).to.equal(2);
      expect(result.columns).to.equal(3);
    });
    it('should correctly load the array data', () => {
      const array = [[1,2,3], [4, 5, 6]];
      const result = Matrix.fromArray(array);
      expect(result.data).to.equal(array);
    });
    it('should handle one-dimensional arrays', () => {
      const array = [1, 2, 3];
      const result = Matrix.fromArray(array);
      expect(result.rows).to.equal(3);
      expect(result.columns).to.equal(1);
      expect(result.data).to.deep.equal([[1], [2], [3]]);
    });
    it('should throw error on irregular multi-dimensional array', () => {
      const array = [[1], [2, 2], [3, 3, 3]];
      expect(() => Matrix.fromArray(array)).to.throw(Error);
    });
    it('should handle empty arrays', () => {
      const array = [[], [], []];
      const result = Matrix.fromArray(array);
      expect(result.rows).to.equal(3);
      expect(result.columns).to.equal(0);
    });
  });
  describe('#toArray', () => {
    it('should export matrix to an array', () => {
      const mat = new Matrix(2, 3);
      const result = mat.toArray();
      expect(result).to.deep.equal([[0, 0, 0], [0, 0, 0]]);
    });
    it('should export to one-dimensional array', () => {
      const mat = new Matrix(1, 3);
      const result = mat.toArray();
      expect(result).to.deep.equal([0, 0, 0]);
    });
  });
  describe('#randomize', () => {
    it('should validate upper and lower bounds', () => {
      const mat = new Matrix(4, 4);
      expect(() => mat.randomize(9, 5)).to.throw(Error);
    });
    it('should only use values within the given range', () => {
      const mat = new Matrix(2, 3);
      const lower = 3;
      const upper = 5;
      for (let i = 0; i < 1000; i++) {
        mat.randomize(lower, upper);
        mat.data.forEach(row =>
          row.forEach(element =>
            expect(lower < element && element < upper).to.be.true));
      }
    });
    it('should handle the same upper and lower bound', () => {
      const mat = new Matrix(2, 3);
      mat.randomize(8,8);
      mat.data.forEach(row =>
        row.forEach(element =>
          expect(element).to.equal(8)));
    });
  });
  describe('#map', () => {
    it('should apply the given function to every element in the matrix', () => {
      const func = x => x * 2;
      const mat = new Matrix(2, 3);
      mat.randomize(2, 2);
      mat.map(func);
      mat.data.forEach(row =>
        row.forEach(element =>
          expect(element).to.equal(4)));
    });
    it('should not modify the matrix', () => {
      const func = x => x * 2;
      const mat = new Matrix(2, 3);
      mat.randomize(2, 2);
      const result = Matrix.map(mat, func);
      mat.data.forEach(row =>
        row.forEach(element =>
          expect(element).to.equal(2)));
      result.data.forEach(row =>
        row.forEach(element =>
          expect(element).to.equal(4)));
    });
  });
  describe('#dot', () => {
    it('should be mathematically correct', () => {
      const mat1 = Matrix.fromArray([
        [1, 2, 3],
        [4, 5, 6]]);
      const mat2 = Matrix.fromArray([
        [1, 2, 3, 4, 5],
        [3, 4, 5, 6, 7],
        [6, 7, 8, 9, 0]]);
      const result = Matrix.dot(mat1, mat2);
      expect(result.data).to.deep.equal([
        [25, 31, 37, 43, 19],
        [55, 70, 85, 100, 55]]);
    });
    it('should verify matrix dimensions', () => {
      const mat1 = new Matrix(2, 3);
      const mat2 = new Matrix(4, 5);
      expect(() => Matrix.dot(mat1, mat2)).to.throw(Error);
    });
  });
  describe('#multiply', () => {
    it('should handle multiplying by a matrix', () => {
      const mat1 = new Matrix(2, 3).randomize(3, 3);
      const mat2 = new Matrix(2, 3).randomize(4, 4);
      mat1.multiply(mat2);
      mat1.data.forEach(row =>
        row.forEach(element =>
          expect(element).to.equal(12)));
    });
    it('should not modify matrix when used statically', () => {
      const mat1 = new Matrix(2, 3).randomize(3, 3);
      const mat2 = new Matrix(2, 3).randomize(4, 4);
      const result = Matrix.multiply(mat1, mat2);
      mat1.data.forEach(row =>
        row.forEach(element =>
          expect(element).to.equal(3)));
      mat2.data.forEach(row =>
        row.forEach(element =>
          expect(element).to.equal(4)));
      result.data.forEach(row =>
        row.forEach(element =>
          expect(element).to.equal(12)));
    });
    it('should validate matrix dimensions', () => {
      const mat1 = new Matrix(2, 3).randomize(3, 3);
      const mat2 = new Matrix(4, 5).randomize(4, 4);
      expect(() => mat1.multiply(mat2)).to.throw(Error);
    });
    it('should validate matrix dimensions when called statically', () => {
      const mat1 = new Matrix(2, 3).randomize(3, 3);
      const mat2 = new Matrix(4, 5).randomize(4, 4);
      expect(() => Matrix.multiply(mat1, mat2)).to.throw(Error);
    });
    it('should handle multiplying by a single value', () => {
      const mat = new Matrix(2, 3).randomize(3, 3);
      mat.multiply(2);
      mat.data.forEach(row =>
        row.forEach(element =>
          expect(element).to.equal(6)));
    });
  });
  describe('#add', () => {
    it('should handle adding by a single value', () => {
      const mat = new Matrix(2, 3).randomize(3, 3);
      mat.add(9);
      mat.data.forEach(row =>
        row.forEach(element =>
          expect(element).to.equal(12)));
    });
    it('should handle adding by a matrix', () => {
      const mat1 = new Matrix(2, 3).randomize(3, 3);
      const mat2 = new Matrix(2, 3).randomize(9, 9);
      mat1.add(mat2);
      mat1.data.forEach(row =>
        row.forEach(element =>
          expect(element).to.equal(12)));
    });
    it('should verify matrix dimensions', () => {
      const mat1 = new Matrix(2, 3).randomize(3, 3);
      const mat2 = new Matrix(3, 4).randomize(9, 9);
      expect(() => mat1.add(mat2)).to.throw(Error);
    });
  });
  describe('#subtract', () => {
    it('should handle subtracting matrices', () => {
      const mat1 = new Matrix(2, 3).randomize(9, 9);
      const mat2 = new Matrix(2, 3).randomize(5, 5);
      const result = Matrix.subtract(mat1, mat2);
      mat1.data.forEach(row =>
        row.forEach(element =>
          expect(element).to.equal(9)));
      mat2.data.forEach(row =>
        row.forEach(element =>
          expect(element).to.equal(5)));
      result.data.forEach(row =>
        row.forEach(element =>
          expect(element).to.equal(4)));
    });
    it('should verify matrix dimensions', () => {
      const mat1 = new Matrix(2, 3).randomize(9, 9);
      const mat2 = new Matrix(3, 4).randomize(5, 5);
      expect(() => Matrix.subtract(mat1, mat2)).to.throw(Error);
    });
  });
  describe('#transpose - static', () => {
    it('should transpose the matrix', () => {
      const mat = Matrix.fromArray([
        [1, 2, 3],
        [4, 5, 6]
      ]);
      const result = Matrix.transpose(mat);
      expect(result.data).to.deep.equal([
        [1, 4],
        [2, 5],
        [3, 6]
      ])
    });
  });
})
