import argparse
import pickle
from inputs import mouse, hand_tracker
from tasks import cursor2d, handtask
from data_recorder import DataRecorder
from inputs.decoder import RealTimeDecoder

NUM_DOF = 2


def load_decoder(decoder_name):
    with open(f"data/trained_decoders/{decoder_name}", "rb") as f:
        model, neuralsim, neural_scaler, output_scaler = pickle.load(f)
    decoder = RealTimeDecoder(NUM_DOF, model, neuralsim, neural_scaler, output_scaler)
    print(f"Loaded decoder: {decoder_name}")
    return decoder


def get_input_source(input_choice):
    if input_choice == "mouse":
        return mouse
    elif input_choice == "hand_tracker":
        return hand_tracker
    else:
        raise ValueError(f"Invalid input choice: {input_choice}")


def get_task(task_choice):
    if task_choice == "cursor":
        return cursor2d.cursor_task
    elif task_choice == "other":
        return handtask.hand_task
    else:
        raise ValueError(f"Invalid task choice: {task_choice}")


def main():
    parser = argparse.ArgumentParser(description="BMI Simulator")
    parser.add_argument("-i", "--input", default="mouse", choices=["mouse", "hand_tracker"],
                        help="Input source choice: mouse or hand_tracker.")
    parser.add_argument("-t", "--task", default="cursor", choices=["cursor", "other"],
                        help="Task choice: cursor or other.")
    parser.add_argument("-d", "--decoder", default=None,
                        help="Name of the decoder file (e.g., rnndecoder1.pkl). If specified, a real-time wrapper will load the decoder.")

    args = parser.parse_args()
    input_source = get_input_source(args.input)

    # If a decoder is specified, load it using a real-time wrapper
    decoder = None
    if args.decoder:
        decoder = load_decoder(args.decoder)

    task = get_task(args.task)
    task(input_source, DataRecorder(), decoder)


if __name__ == "__main__":
    main()
