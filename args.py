import argparse

# Argument parsing
parser = argparse.ArgumentParser(description="Description of your Dicom PreProcess")
parser.add_argument(
    "-a",
    "--algorithm",
    help='Choose the algorithm to use. It can be "grail" or "fedbs"',
    default="fedbs",
)
parser.add_argument(
    "-m",
    "--method",
    help='Choose the method to use. It can be "dog", "log" or "bbp". This command is'
    ' validated if and only if the "fedbs" algorithm is chosen',
    default="dog",
)
parser.add_argument(
    "-f",
    "--file",
    help="Choose the DICOM file path to be processed by default the file is located "
    '"data/" ',
    required=True,
    default="20587054.dcm",
)
argument = parser.parse_args()
