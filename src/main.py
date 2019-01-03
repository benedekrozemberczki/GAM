from parser import parameter_parser
from utils import tab_printer
from gam import GAMTrainer

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
