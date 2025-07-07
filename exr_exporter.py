import os
import sys # Added sys for argument checking and exit
import numpy as np

# Attempt to import OpenEXR and Imath, provide helpful error messages if they fail
try:
    import OpenEXR
    import Imath
except ImportError:
    print("Error: The 'OpenEXR' or 'Imath' Python packages are not installed.")
    print("Please install them (e.g., 'pip install OpenEXR-python') to use this script.")
    # Make sure to exit after printing the message if dependencies are critical.
    sys.exit(1)

def save_tensor_as_exr(tensor: np.ndarray, out_path: str):
    """
    Saves a NumPy tensor as an EXR image file.

    The tensor is expected to be in HWC (Height, Width, Channels) format.
    If it's 2D (HW), it's treated as a single-channel image.
    If it's >3D, dimensions beyond the third are flattened into the channel dimension.

    Args:
        tensor: The NumPy array to save. Expected to be convertible to float32.
        out_path: The path where the EXR file will be saved.
    """
    if not isinstance(tensor, np.ndarray):
        raise TypeError(f"Input tensor must be a NumPy array, got {type(tensor)}.")

    try:
        arr = tensor.astype(np.float32)
    except Exception as e:
        print(f"Error converting tensor to float32: {e}")
        return

    # Reshape array to HWC (Height, Width, Channels)
    if arr.ndim == 2:
        # Grayscale image, add a channel dimension
        arr = arr[:, :, np.newaxis]
    elif arr.ndim > 3:
        # Flatten higher dimensions into the channel dimension
        original_shape = arr.shape
        arr = arr.reshape(original_shape[0], original_shape[1], -1)
        print(f"Info: Tensor with shape {original_shape} reshaped to {arr.shape} for EXR export.")

    if arr.ndim != 3:
        print(f"Error: Tensor must be 2D, 3D or convertible to 3D for EXR export. Got shape {arr.shape} after processing.")
        return

    height, width, channels = arr.shape

    if height == 0 or width == 0 or channels == 0:
        print(f"Warning: Tensor has zero dimension (shape: {arr.shape}) for {out_path}. Skipping.")
        return

    header = OpenEXR.Header(width, height)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    # Robust channel naming: C1, C2, ... or R, G, B, A for common cases
    if channels == 1:
        channel_names = ['R']
    elif channels == 2:
        channel_names = ['R', 'G']
    elif channels == 3:
        channel_names = ['R', 'G', 'B']
    elif channels == 4:
        channel_names = ['R', 'G', 'B', 'A']
    else:
        channel_names = [f'C{i+1}' for i in range(channels)]

    header['channels'] = {name: Imath.Channel(FLOAT) for name in channel_names}

    exr = None  # Initialize exr to None for the finally block
    try:
        exr = OpenEXR.OutputFile(out_path, header)
        # Prepare data for each channel
        channel_data_dict = {}
        for i, name in enumerate(channel_names):
            channel_data_dict[name] = arr[:, :, i].tobytes()
        exr.writePixels(channel_data_dict)
    except Exception as e:
        print(f"Error: Could not write EXR file {out_path}. Reason: {e}")
        return  # Skip saving this file if an error occurs
    finally:
        if exr is not None and not exr.isClosed():
            try:
                exr.close()
            except Exception as e:
                print(f"Error: Could not close EXR file {out_path}. Reason: {e}")

def export_model_layers_to_exr(state_dict: dict, out_dir: str):
    """
    Exports layers (tensors) from a model's state_dict to EXR files.

    Each tensor in the state_dict is saved as a separate EXR file using `save_tensor_as_exr`.
    Tensors that are not NumPy arrays (e.g., PyTorch tensors) are attempted to be converted.
    Empty tensors or tensors with a zero dimension are skipped.
    Layer names are sanitized for use as filenames.

    Args:
        state_dict: A dictionary where keys are layer names (strings) and
                    values are the tensors (e.g., NumPy arrays or PyTorch tensors).
        out_dir: The directory where EXR files will be saved.
                 It will be created if it doesn't exist.
    """
    if not isinstance(state_dict, dict):
        print("Error: Input 'state_dict' must be a dictionary.")
        return

    try:
        # Create the output directory; exist_ok=True means no error if it already exists.
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        # This can happen due to permission issues, path being a file, etc.
        print(f"Error: Could not create output directory '{out_dir}'. Reason: {e}")
        return # Cannot proceed if output directory is not usable

    if not state_dict:
        print("Warning: The provided state_dict is empty. No tensors to export.")
        return

    print(f"Starting export of {len(state_dict)} tensor(s) to directory: {out_dir}")
    exported_count = 0
    skipped_count = 0

    for name, item in state_dict.items():
        current_tensor = None

        # Ensure the item is a NumPy array for processing
        if isinstance(item, np.ndarray):
            current_tensor = item
        elif hasattr(item, 'cpu') and hasattr(item, 'numpy'): # Common for PyTorch tensors
            try:
                current_tensor = item.cpu().numpy() # Move to CPU and convert
            except Exception as e:
                print(f"Skipping '{name}': Error during PyTorch tensor to NumPy conversion. Reason: {e}")
                skipped_count += 1
                continue
        else:
            print(f"Skipping '{name}': Item is not a NumPy array or a convertible tensor object (type: {type(item)}).")
            skipped_count += 1
            continue

        # Validate tensor properties before saving
        if current_tensor.size == 0:
            print(f"Skipping '{name}': Tensor is empty (total elements is 0, shape: {current_tensor.shape}).")
            skipped_count += 1
            continue
        if 0 in current_tensor.shape: # Check if any dimension is zero
             print(f"Skipping '{name}': Tensor has a zero dimension (shape: {current_tensor.shape}).")
             skipped_count += 1
             continue

        # Sanitize the layer name to be used as a filename
        # Replace periods and slashes, common in layer names, with underscores.
        sanitized_name = name.replace('.', '_').replace('/', '_').replace('\\', '_')
        fname = os.path.join(out_dir, f"{sanitized_name}.exr")

        # Announce processing and save
        print(f"Processing '{name}' (shape: {current_tensor.shape}, type: {current_tensor.dtype}) -> {fname}")
        save_tensor_as_exr(current_tensor, fname) # This function handles its own print for success/failure
        # Assuming save_tensor_as_exr either saves or prints an error and returns.
        # To accurately count, save_tensor_as_exr should return a status or raise an exception on failure.
        # For now, we'll assume if it doesn't error out here, it's processed.
        # A more robust way would be for save_tensor_as_exr to return True on success.
        # If save_tensor_as_exr prints an error and returns, it's effectively skipped.
        if os.path.exists(fname): # Check if file was actually created
             exported_count +=1
        else: # Implies saving failed in save_tensor_as_exr
             skipped_count +=1


    print(f"Export complete. Exported: {exported_count} tensor(s). Skipped: {skipped_count} tensor(s).")

# Example usage: This script can be run from the command line to export model weights.
# It expects a path to a PyTorch model file (.pth) and an output directory.
if __name__ == "__main__":
    # --- Argument Parsing and Validation ---
    if len(sys.argv) != 3:
        # Ensure the script is called with the correct number of arguments
        print("Usage: python exr_exporter.py <model_path.pth> <output_directory>")
        sys.exit(1) # Exit if arguments are incorrect

    model_path = sys.argv[1] # First argument: path to the model file
    out_dir = sys.argv[2]    # Second argument: directory to save EXR files

    # Validate that the model_path is an existing file
    if not os.path.isfile(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        sys.exit(1)

    # --- PyTorch Dependency Check and Model Loading ---
    try:
        # Dynamically import torch only when this script is run directly
        import torch
    except ImportError:
        print("Error: The 'torch' Python package is not installed.")
        print("Please install PyTorch (e.g., 'pip install torch') to use this example loader.")
        sys.exit(1)

    print(f"Loading model state_dict from: {model_path}")
    try:
        # Load the model/state_dict from the specified path, mapping to CPU to avoid GPU issues
        loaded_object = torch.load(model_path, map_location='cpu')

        state_dict = None
        # Check if the loaded object is a model instance or already a state_dict
        if hasattr(loaded_object, 'state_dict') and callable(loaded_object.state_dict):
            # If it's a model object, get its state_dict
            state_dict = loaded_object.state_dict()
            print("Successfully extracted state_dict from model object.")
        elif isinstance(loaded_object, dict):
            # If it's already a dictionary, assume it's a state_dict
            state_dict = loaded_object
            print("Successfully loaded state_dict (assumed to be a dictionary).")
        else:
            # If it's neither, print an error and exit
            print(f"Error: Loaded object from '{model_path}' is not a recognized model or state_dict. Found type: {type(loaded_object)}")
            sys.exit(1)

    except FileNotFoundError: # Should be caught by os.path.isfile, but good for robustness
        print(f"Error: Model file not found at '{model_path}' (FileNotFoundError).")
        sys.exit(1)
    except Exception as e:
        # Catch any other errors during model loading
        print(f"Error loading model or state_dict from '{model_path}'. Reason: {e}")
        sys.exit(1)

    # --- Perform Export ---
    # Call the main export function with the loaded state_dict and output directory
    export_model_layers_to_exr(state_dict, out_dir)
    print("Script execution finished.")
