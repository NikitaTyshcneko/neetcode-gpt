class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        def dfs(iteration, total):
            if iteration == 0:
                return round(total, 5)
            
            total = total - learning_rate * 2 * total

            return dfs(iteration-1, total)
        
        return dfs(iterations, init)

