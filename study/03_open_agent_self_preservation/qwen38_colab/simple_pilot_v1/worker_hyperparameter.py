"""Fresh-adapter tuning through the existing training/evaluation implementation."""
from hyperparameter_recipes import configure
import preservation_runner

configure()

if __name__=='__main__':
    preservation_runner.main()
