import numpy as np

class LagrangianManager:
    """
    Maintains the dual multipliers (lambdas) for the PID-Lagrangian Safety layer.
    Implements a damping term to prevent oscillation in the dual ascent.
    """
    def __init__(self, init_lambda=0.0, max_lambda=10.0, lr=0.01, damping=0.05):
        self.lr = lr
        self.damping = damping
        self.max_lambda = max_lambda
        
        # Dual variables (initially unconstrained)
        self.lambda_soc = init_lambda
        self.lambda_line = init_lambda
        self.lambda_voltage = init_lambda
        
        # Track previous violations for the Derivative (damping) term
        self.prev_viol_soc = 0.0
        self.prev_viol_line = 0.0
        self.prev_viol_voltage = 0.0

    def update(self, mean_viol_soc, mean_viol_line, mean_viol_voltage):
        """
        PID-Lagrangian Update:
        λ_{t+1} = clip(λ_t + lr * V + damping * ΔV, 0, Max_Lambda)
        Note: The user specifies 'Violation - Limit'. If the inputs here are already 
        the calculated mean violation amount (i.e. strictly > 0 if violated, 0 otherwise), 
        then V is just the passed argument.
        """
        # Calculate Delta Violations
        d_soc = mean_viol_soc - self.prev_viol_soc
        d_line = mean_viol_line - self.prev_viol_line
        d_voltage = mean_viol_voltage - self.prev_viol_voltage
        
        # Update Lambda SoC
        self.lambda_soc += self.lr * mean_viol_soc + self.damping * d_soc
        self.lambda_soc = np.clip(self.lambda_soc, 0.0, self.max_lambda)
        
        # Update Lambda Line
        self.lambda_line += self.lr * mean_viol_line + self.damping * d_line
        self.lambda_line = np.clip(self.lambda_line, 0.0, self.max_lambda)
        
        # Update Lambda Voltage
        self.lambda_voltage += self.lr * mean_viol_voltage + self.damping * d_voltage
        self.lambda_voltage = np.clip(self.lambda_voltage, 0.0, self.max_lambda)
        
        # Store for next step
        self.prev_viol_soc = mean_viol_soc
        self.prev_viol_line = mean_viol_line
        self.prev_viol_voltage = mean_viol_voltage
        
        return {
            'lambda_soc': self.lambda_soc,
            'lambda_line': self.lambda_line,
            'lambda_voltage': self.lambda_voltage
        }

    def get_lambdas(self):
        return {
            'lambda_soc': self.lambda_soc,
            'lambda_line': self.lambda_line,
            'lambda_voltage': self.lambda_voltage
        }
