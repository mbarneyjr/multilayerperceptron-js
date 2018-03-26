class Matrix  {
  constructor(numRows, numColumns) {
    this.rows = numRows;
    this.columns = numColumns;
    this.data = [];

    for (let row = 0; row < this.rows; row++) {
      this.data[row] = [];
      for (let col = 0; col < this.columns; col++) {
        this.data[row][col] = 0;
      }
    }
  }

  static fromArray(inputArray) {
    if (!inputArray instanceof Array) {
      throw Error('Cannot create Matrix from non-Array type!');
    }
    if (inputArray[0] instanceof Array) {
      if (new Set(inputArray.map(a => a.length)).size != 1) {
        throw Error('Array must be square');
      }
      let result = new Matrix(inputArray.length, inputArray[0].length);
      result.data = inputArray;
      return result;
    } else {
      let result = new Matrix(inputArray.length, 1);
      for (let row = 0; row < inputArray.length; row++) {
        result.data[row][0] = inputArray[row];
      }
      return result;
    }
  }

  toArray() {
    if (this.rows == 1) {
      return this.data[0];
    } else {
      return this.data;
    }
  }

  randomize(lower, upper) {
    if (lower === undefined || upper === undefined || upper < lower) {
      throw Error('Please specify upper and lower bounds!');
    }
    for (let row = 0; row < this.rows; row++) {
      for (let col = 0; col < this.columns; col++) {
        this.data[row][col] = Math.random() * (upper - lower) + lower;
      }
    }
    return this;
  }

  map(func) {
    for (let row = 0; row < this.rows; row++) {
      for (let col = 0; col < this.columns; col++) {
        let value = this.data[row][col]
        this.data[row][col] = func(value, row, col);
      }
    }
    return this
  }

  static map(mat, func) {
    let result = new Matrix(mat.rows, mat.columns);
    for (let row = 0; row < mat.rows; row++) {
      for (let col = 0; col < mat.columns; col++) {
        result.data[row][col] = func(mat.data[row][col], row, col);
      }
    }
    return result
  }

  static dot(mat1, mat2) {
    if (mat1.columns !== mat2.rows) {
      throw Error('Columns of first must match rows of second!');
    }
    let result = new Matrix(mat1.rows, mat2.columns);
    for (let row = 0; row < mat1.rows; row++) {
      for (let col = 0; col < mat2.columns; col++) {
        let sum = 0;
        for (let newCol = 0; newCol < mat1.columns; newCol++) {
          sum += mat1.data[row][newCol] * mat2.data[newCol][col];
        }
        result.data[row][col] = sum;
      }
    }
    return result;
  }

  multiply(other) {
    if (other instanceof Matrix) {
      if (this.rows !== other.rows || this.columns !== other.columns) {
        throw Error('Incompatible matrices for element-wise multiplication');
      }
      for (let row = 0; row < this.rows; row++) {
        for (let col = 0; col < this.columns; col++) {
          this.data[row][col] *= other.data[row][col];
        }
      }
    } else {
      for (let row = 0; row < this.rows; row++) {
        for (let col = 0; col < this.columns; col++) {
          this.data[row][col] *= other;
        }
      }
    }
    return this;
  }

  static multiply(mat1, mat2) {
    if (mat2 instanceof Matrix) {
      if (mat1.rows !== mat2.rows || mat1.columns !== mat2.columns) {
        throw Error('Incompatible matrices for element-wise multiplication');
      }
      let result = new Matrix(mat1.rows, mat1.columns);
      for (let row = 0; row < mat1.rows; row++) {
        for (let col = 0; col < mat1.columns; col++) {
          result.data[row][col] = mat1.data[row][col] * mat2.data[row][col];
        }
      }
      return result;
    } else {
      for (let row = 0; row < mat1.rows; row++) {
        for (let col = 0; col < mat1.columns; col++) {
          mat1.data[row][col] *= mat2;
        }
      }
      return this;
    }
  }

  add(other) {
    if (other instanceof Matrix) {
      if (this.rows !== other.rows || this.columns !== other.columns) {
        throw Error('Incompatible matrices for element-wise addition');
      }
      for (let row = 0; row < this.rows; row++) {
        for (let col = 0; col < this.columns; col++) {
          this.data[row][col] += other.data[row][col];
        }
      }
    } else {
      for (let row = 0; row < this.rows; row++) {
        for (let col = 0; col < this.columns; col++) {
          this.data[row][col] += other;
        }
      }
    }
    return this
  }

  static subtract(mat1, mat2) {
    if (mat1.rows !== mat2.rows || mat1.columns !== mat2.columns) {
      throw Error('Incompatible matrices for element-wise subtraction');
    }
    let result = new Matrix(mat1.rows, mat1.columns);
    for (let row = 0; row < mat1.rows; row++) {
      for (let col = 0; col < mat1.columns; col++) {
        result.data[row][col] = mat1.data[row][col] - mat2.data[row][col];
      }
    }
    return result;
  }

  static transpose(matrix) {
    let result = new Matrix(matrix.columns, matrix.rows);
    for (let row = 0; row < matrix.rows; row++) {
      for (let col = 0; col < matrix.columns; col++) {
        result.data[col][row] = matrix.data[row][col];
      }
    }
    return result
  }

  print() {
    let fstring = '['; 
    for (let i = 0; i < this.data.length; i++) {
      fstring +=  (i != 0 ? ' ' : '' ) + ` [${this.data[i].map(x => ' ' + x.toString() + ' ')}],\n`;
    }
    console.log(fstring.substring(0, fstring.length - 2) + ' ]');
  }
}

module.exports = {
  Matrix
}
