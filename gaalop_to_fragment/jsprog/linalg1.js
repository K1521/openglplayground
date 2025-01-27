export { Vector, Matrix };


class Vector {
    constructor(array) {
        this.array = array;
    }

    // Get the size (length) of the vector
    size() {
        return this.array.length;
    }

    // Helper function to check if two vectors can be operated on (same size)
    _checkSizeAndType(B) {
        if (!(B instanceof Vector)) {
            throw new TypeError("The argument must be an instance of Vector");
        }
        if (this.size() !== B.size()) {
            throw new TypeError("Vectors must be of the same size for this operation");
        }
    }

    // Vector addition
    add(B) {
        this._checkSizeAndType(B);
        const result = this.array.map((val, idx) => val + B.array[idx]);
        return new Vector(result);
    }

    // Vector subtraction
    subtract(B) {
        this._checkSizeAndType(B);
        const result = this.array.map((val, idx) => val - B.array[idx]);
        return new Vector(result);
    }

    // Scalar multiplication
    mul(scalar) {
        if (typeof scalar !== 'number') {
            throw new TypeError("Multiplication must be by a scalar (number)");
        }
        const result = this.array.map(val => val * scalar);
        return new Vector(result);
    }

    div(scalar){
        return this.mul(1/scalar);
    }

    // Dot product of two vectors
    dot(B) {
        this._checkSizeAndType(B);
        const result = this.array.reduce((sum, val, idx) => sum + val * B.array[idx], 0);
        return result;
    }

    // Cross product of two 3D vectors
    cross(B) {
        this._checkSizeAndType(B);
        if (this.size() !== 3 || B.size() !== 3) {
            throw new TypeError("Cross product is only defined for 3D vectors");
        }

        const [x1, y1, z1] = this.array;
        const [x2, y2, z2] = B.array;

        const result = [
            y1 * z2 - z1 * y2,  // x-component
            z1 * x2 - x1 * z2,  // y-component
            x1 * y2 - y1 * x2   // z-component
        ];
        return new Vector(result);
    }

    abs(){return Math.sqrt(this.array.reduce((sum, val) => sum + val * val, 0));}

    normalize(){
        const l=this.abs();
        if(l===0)return new Vector(new Array(this.size()).fill(0));//this.mul(0);
        return this.div(l);
    }
}

class Matrix {
    constructor(array) {
      this.array = array;
    }

    size(){
        const rows = this.array.length;
        const cols = this.array[0].length;
        return [rows,cols];
    }

    #matmul(B){
        const MA=this.array;
        const MB=B.array;
        const [rowsA,colsA] = this.size();
        const [rowsB,colsB] = B.size();
       
        // Check if multiplication is possible
        if (colsA !== rowsB) {
            throw new Error("Matrix dimensions do not allow multiplication");
        }

        // Initialize result matrix with zeroes
        const result = new Array(rowsA);
        for (let i = 0; i < rowsA; i++) {
            result[i] = new Array(colsB).fill(0);
        }

        // Perform matrix multiplication
        for (let i = 0; i < rowsA; i++) {
            for (let j = 0; j < colsB; j++) {
                for (let k = 0; k < colsA; k++) {
                    result[i][j] += MA[i][k] * MB[k][j];
                }
            }
        }

        return new Matrix(result);
    }

    #scalarmul(B){
        return new Matrix(this.array.map(row => row.map(element => element * scalar)));
    }

    #vecmul(B){
        if(this.size()[1]!==B.size())
            throw new Error("Matrix dimensions do not allow matrix vector multiplication");
        Barr=B.array;
        return new Vector(
            this.array.map(row => 
                row.reduce((sum, val, idx) => 
                    sum + val * Barr[idx], 0)
        ));
    }
  
    mul(B){
        if(B instanceof Matrix)return this.#matmul(B);
        if(B instanceof Vector)return this.#vecmul(B);
        if(typeof value === "number")return this.#scalarmul(B);
        throw new TypeError(`Incompatible types for matmul: ${typeof this} and ${typeof B}`);
    }

    add(B) {
        const [rowsA, colsA] = this.size();
        const [rowsB, colsB] = B.size();

        // Check if matrices have the same dimensions
        if (rowsA !== rowsB || colsA !== colsB) {
            throw new Error("Matrices must have the same dimensions for addition");
        }

        // Perform element-wise addition
        const result = this.array.map((row, i) =>
            row.map((val, j) => val + B.array[i][j])
        );

        return new Matrix(result);
    }

    // Subtract two matrices
    sub(B) {
        const [rowsA, colsA] = this.size();
        const [rowsB, colsB] = B.size();

        // Check if matrices have the same dimensions
        if (rowsA !== rowsB || colsA !== colsB) {
            throw new Error("Matrices must have the same dimensions for subtraction");
        }

        // Perform element-wise subtraction
        const result = this.array.map((row, i) =>
            row.map((val, j) => val - B.array[i][j])
        );

        return new Matrix(result);
    }

    // Function to compute the rotation matrix around an arbitrary axis k by an angle theta (in radians)
    static rotationMatrix(k, theta) {
        // Normalize k to ensure it's a unit vector
        k = k.normalize().array;
    
        // Cosine and Sine of the angle
        const cosTheta = Math.cos(theta);
        const sinTheta = Math.sin(theta);
    
        // Identity matrix (3x3)
        const identity = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ];
    
        // Outer product of k with itself (k * k^T)
        const outerProduct = [
            [k[0] * k[0], k[0] * k[1], k[0] * k[2]],
            [k[1] * k[0], k[1] * k[1], k[1] * k[2]],
            [k[2] * k[0], k[2] * k[1], k[2] * k[2]]
        ];
    
    
        // Cross-product matrix of k
        const crossMatrix = [
            [0, -k[2], k[1]],  // First row
            [k[2], 0, -k[0]],  // Second row
            [-k[1], k[0], 0]   // Third row
        ];
    
        // Rotation matrix = I * cos(theta) + (1 - cos(theta)) * (k * k^T) + sin(theta) * [k]_{\times}
        const rotationMat = identity.map((row, i) => 
            row.map((val, j) => 
            val * cosTheta +
            (outerProduct[i][j] * (1 - cosTheta)) +
            crossMatrix[i][j] * sinTheta
            )
        );
    
        return new Matrix(rotationMat);
    }

    
    // Static method to create an identity matrix of given size
    static eye(size) {
        if (typeof size !== 'number' || size <= 0) {
            throw new Error("Size must be a positive integer");
        }
        
        const identity = [];
        for (let i = 0; i < size; i++) {
            const row = new Array(size).fill(0);  // Fill the row with 0s
            row[i] = 1;  // Set the diagonal element to 1
            identity.push(row);
        }
        
        return new Matrix(identity);
    }

    orthogonalize() {
        //https://stackoverflow.com/questions/23080791/eigen-re-orthogonalization-of-rotation-matrix
        // i modified it a little
        if (this.size()[0] !== 3 || this.size()[1] !== 3) {
            throw new Error("This method only works for 3x3 matrices.");
        }

        const [x, y, z] = this.array.map( val=>new Vector(val));

        const error = x.dot(y)/2;

        // Spread the error equally between x and y
        const x_ort = x.subtract(y.mul(error));
        const y_ort = y.subtract(x.mul(error));


        //const z_ort = x_ort.cross(y_ort);


        const x_new = x_ort.mul(0.5 * (3 - x_ort.dot(x_ort)));
        const y_new = y_ort.mul(0.5 * (3 - y_ort.dot(y_ort)));
        //const z_new = z_ort.mul(0.5 * (3 - z_ort.dot(z_ort)));

        let z_new = x_new.cross(y_new);

        // If z_new's dot product with original z is negative, flip z_new
        if (z_new.dot(z) < 0) {
            z_new = z_new.mul(-1);
        }

        const orthogonalMatrix = new Matrix([x_new.array, y_new.array, z_new.array]);

        return orthogonalMatrix;
    }
}