# MinFinder and Simulated Annealing Implementation

This repository contains implementations of the MinFinder algorithm and the Corana Simulated Annealing method for finding local minima of multidimensional functions. Both methods utilize Gradient Descent for local optimization.

## MinFinder
- **Description**: A stochastic clustering algorithm designed to locate all local minima of a continuous and differentiable function within a bounded domain.
- **Features**:
  - Uses a predetermined number of iterations as the termination rule.
  - Employs Gradient Descent for local minimization.
  - Suitable for functions with multiple local minima or roots of equations.

## Simulated Annealing (Corana)
- **Description**: A global optimization algorithm derived from the Simulated Annealing method, adapted for continuous variables.
- **Features**:
  - Combines random search with Gradient Descent for local optimization.
  - Effective for multimodal functions with many local minima.
  - Includes temperature reduction and adaptive step adjustments.

## Files
- `minfinder.cpp`: Implementation of the MinFinder algorithm.
- `simulated_annealing.py`: Implementation of the Corana Simulated Annealing method.
- `journal5_minfinder.pdf`: Research paper detailing the MinFinder algorithm.
- `siman_corana.pdf`: Research paper detailing the Corana Simulated Annealing method.

## Usage
1. **MinFinder**:
   - Compile and run `minfinder.cpp` to find local minima of predefined functions.
   - Results are saved in `results_minfinder.txt`.

2. **Simulated Annealing**:
   - Run `simulated_annealing.py` to apply the Corana method to test functions.
   - Output includes the best solution and function value for each test function.

## Dependencies
- C++ compiler (for MinFinder).
- Python with NumPy (for Simulated Annealing).

## References
- Tsoulos, I. G., & Lagaris, I. E. (2006). "MinFinder: Locating all the local minima of a function." *Computer Physics Communications*.
- Corana, A., et al. (1987). "Minimizing multimodal functions of continuous variables with the ‘Simulated Annealing’ algorithm." *ACM Transactions on Mathematical Software*.

## License
This project is open-source and available under the MIT License.
