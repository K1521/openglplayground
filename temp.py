const int MAX_ITERATIONS = 20;
const float TOLERANCE = 1e-6;

// Evaluate the polynomial P(x) and its derivative P'(x)
void evaluatePolynomial(float x, float a, float b, float c, float d, float e, out float p, out float dp) {
    // P(x) = ax^4 + bx^3 + cx^2 + dx + e
    p = ((a * x + b) * x + c) * x * x + d * x + e;
    // P'(x) = 4ax^3 + 3bx^2 + 2cx + d
    dp = (4.0 * a * x + 3.0 * b) * x * x + 2.0 * c * x + d;
}

// Newton iterations to find a root
float newtonRoot(float a, float b, float c, float d, float e, float initialGuess) {
    float x = initialGuess;
    for (int i = 0; i < MAX_ITERATIONS; ++i) {
        float p, dp;
        evaluatePolynomial(x, a, b, c, d, e, p, dp);
        if (abs(p) < TOLERANCE || abs(dp) < TOLERANCE) {
            break; // Convergence or flat derivative
        }
        x -= p / dp; // Newton's update
    }
    return x;
}

// Shift the polynomial P(x) to P(x + shift)
void shiftPolynomial(float a, float b, float c, float d, float e, float shift, out float newA, out float newB, out float newC, out float newD, out float newE) {
    // Compute the shifted coefficients using nested evaluation
    newE = e + shift * (d + shift * (c + shift * (b + shift * a)));
    newD = d + shift * (2.0 * c + shift * (3.0 * b + shift * 4.0 * a));
    newC = c + shift * (3.0 * b + shift * 6.0 * a);
    newB = b + shift * (4.0 * a);
    newA = a;
}

// Main function to shift a polynomial based on Newton root approximation
void solveQuarticWithShift(
    float a, float b, float c, float d, float e,
    float initialGuess,
    out float shiftedA, out float shiftedB, out float shiftedC, out float shiftedD, out float shiftedE
) {
    // Step 1: Use Newton's method to find an approximate root
    float shift = newtonRoot(a, b, c, d, e, initialGuess);

    // Step 2: Shift the polynomial to center the root
    shiftPolynomial(a, b, c, d, e, shift, shiftedA, shiftedB, shiftedC, shiftedD, shiftedE);
}
