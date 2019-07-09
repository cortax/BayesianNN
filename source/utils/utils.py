import os

def is_float(input):
  try:
    val = float(input)
    return True
  except ValueError:
    return False

def is_int(input):
  try:
    val = int(input)
    return True
  except ValueError:
    return False

def make_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    return file_path