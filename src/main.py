from parser import parameter_parser
from utils import tab_printer
from gam import GAMTrainer

def main():
    """
    Parsing command lines, creating target matrix, fitting a GAM.
    """
    args = parameter_parser()
    tab_printer(args)
    model = GAMTrainer(args)
    model.fit()
    model.score()

if __name__ == "__main__":
    main()
