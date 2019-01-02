from parser import parameter_parser
from utils import tab_printer
from gam import GAMTrainer, MemoryGAMTrainer

def main():
    """
    Parsing command lines, creating target matrix, fitting a GAM.
    """
    args = parameter_parser()
    tab_printer(args)
    if args.model_memory == False: 
        model = GAMTrainer(args)
    else:
        model = MemoryGAMTrainer(args)
    model.fit()
    model.score()
    model.save_predictions_and_logs()

if __name__ == "__main__":
    main()
