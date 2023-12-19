import neuralsim
import argparse
import os
import pickle


def create_fakebrain(num_chans, num_dof, neural_noise):
    return neuralsim.LogLinUnitGenerator(num_chans,
                                         num_dof,
                                         pos_mult=0.5,              # put less emphasis on position
                                         vel_mult=2,                # put more emphasis on velocity
                                         noise_level=neural_noise)


def save_fakebrain(fakebrain, num_chans, num_dof, save_name):
    if not save_name.endswith(".pkl"):
        save_name += ".pkl"

    with open(os.path.join("data", "fakebrains", save_name), 'wb') as f:
        pickle.dump((fakebrain, num_chans, num_dof), f)

    print(f"Fake brain saved as {save_name}\n")


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--num_chans', type=int, default=100)
    parser.add_argument('-n', '--neural_noise_level', type=float, default=0.2)
    parser.add_argument("-t", "--task", default="cursor", choices=["cursor", "hand"],
                        help="Task choice: cursor or hand.")
    parser.add_argument('-o', '--save_name', type=str, default=None)
    args = parser.parse_args()

    if args.save_name is None:
        raise ValueError("Must specify a save_name")

    if args.task == "cursor":
        num_dof = 2
    elif args.task == "hand":
        num_dof = 5

    print(f"\nCreating fake brain with {args.num_chans} channels, noise level {args.neural_noise_level}, and {num_dof} DoF")

    # create fakebrain
    fakebrain = create_fakebrain(args.num_chans, num_dof, args.neural_noise_level)

    # save fakebrain
    save_fakebrain(fakebrain, args.num_chans, num_dof, args.save_name)


if __name__ == "__main__":
    main()
