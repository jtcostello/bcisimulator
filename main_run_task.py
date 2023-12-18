import argparse
import pickle
from data_recorder import DataRecorder
from inputs.decoder import RealTimeDecoder


def load_decoder(decoder_name, num_dof, integration_beta):
    # load in a pre-trained decoder
    if not decoder_name.endswith(".pkl"):
        decoder_name += ".pkl"
    with open(f"data/trained_decoders/{decoder_name}", "rb") as f:
        model, neuralsim, neural_scaler, output_scaler, seq_len = pickle.load(f)
    decoder = RealTimeDecoder(num_dof, model, neuralsim, neural_scaler, output_scaler, seq_len, integration_beta)
    print(f"Loaded decoder: {decoder_name}")
    return decoder


def get_task(task_choice):
    if task_choice == "cursor":
        from tasks import cursor2d
        num_dof = 2
        return cursor2d.cursor_task, num_dof

    elif task_choice == "hand":
        from tasks import handtask
        num_dof = 5
        return handtask.hand_task, num_dof

    else:
        raise ValueError(f"Invalid task choice: {task_choice}")


def main():
    parser = argparse.ArgumentParser(description="BCI Simulator")
    parser.add_argument("-t", "--task", default="cursor", choices=["cursor", "hand"],
                        help="Task choice: cursor or hand.")
    parser.add_argument("-d", "--decoder", default=None,
                        help="Name of the decoder file (e.g., rnndecoder1). If specified, a real-time wrapper will load the decoder.")
    parser.add_argument("-tt", "--target_type", default="random", choices=["random", "centerout"],
                        help="Target type: random or centerout.")
    # add argument for the decoder integration beta
    parser.add_argument("-b", "--integration_beta", default=0.98, type=float,
                        help="Integration beta: the percentage of decoded position that is integrated velocity.")
    args = parser.parse_args()

    # get task
    task, num_dof = get_task(args.task)

    # If a decoder is specified, load it using a real-time wrapper
    decoder = None
    if args.decoder:
        decoder = load_decoder(args.decoder, num_dof, args.integration_beta)

    # run the task
    task(DataRecorder(), decoder, target_type=args.target_type)


if __name__ == "__main__":
    main()
