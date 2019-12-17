"""Running the GAM model."""

from gam import GAMTrainer
from utils import tab_printer
from param_parser import parameter_parser

def main():
    """
    Parsing command line parameters, processing graphs, fitting a GAM.
    """
    args = parameter_parser()
    tab_printer(args)
    model = GAMTrainer(args)
    model.fit()
    model.score()
    model.save_predictions_and_logs()

if __name__ == "__main__":
    main()
