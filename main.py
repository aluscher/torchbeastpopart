import os
import argparse
import datetime

parser = argparse.ArgumentParser()

parser.add_argument("--env", type=str, default="PongNoFrameskip-v4",
                    help="Gym environments")
parser.add_argument("--steps", type=int, default=100000, help="Number of steps")

def main(flags):
    cwd = os.getcwd()
    directory = os.path.join(cwd, "logs", datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    os.makedirs(directory)

    arguments = (
        f'--env {flags.env} '
        f'--savedir {directory} '
        f'--total_steps {flags.steps} '
        '--batch_size 32 '
    )
    print(f'python -m torchbeast.monobeast {arguments}')
    os.system(f'python -m torchbeast.monobeast {arguments}')


if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)
