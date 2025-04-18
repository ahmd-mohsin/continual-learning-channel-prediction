import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import torch.nn.functional as F


class ChannelSequenceDataset(Dataset):
    def __init__(self, file_path, file_extension, device):
        self.file_path = file_path + file_extension
        self.file_extension = file_extension
        self.device = device
        
        if self.file_extension == "npy":
            self.data = np.load(self.file_path, mmap_mode='r')
            self.num_users = self.data.shape[0]
            self.time_length = self.data.shape[-1]
        elif self.file_extension == "mat":
            with h5py.File(self.file_path, "r") as f:
                self.num_users = f["channel_matrix"].shape[0]
                self.time_length = f["channel_matrix"].shape[-1]
                channel_group = f["channel_matrix"]
                real = np.array(channel_group["real"])
                imag = np.array(channel_group["imag"])
                self.data = real + 1j * imag
        else:
            raise ValueError("Unsupported file format. Please use npy or mat.")
    
        self.overlapping_index = 16

        # Normalize the entire complex vector
        # ---------------------------------------------------------
        real = self.data.real
        imag = self.data.imag

        # Normalize real and imaginary separately
        real_min, real_max = real.min(), real.max()
        imag_min, imag_max = imag.min(), imag.max()

        normalized_real = (real - real_min) / (real_max - real_min)
        normalized_imag = (imag - imag_min) / (imag_max - imag_min)

        # Recombine or feed separately to the model
        self.data = normalized_real + 1j * normalized_imag

        # print("###############################")
        # print("max_value real", np.min(normalized_real))
        # print("min_value real", np.max(normalized_real))
        # print("max_value imag", np.min(normalized_imag))
        # print("min_value imag", np.max(normalized_imag))
        # print("###############################")
        # ---------------------------------------------------------

        # # Find the max values for the entire dataset
        # self.max_value = np.max(self.data)  # The absolute max value across both real and imaginary parts
        # self.min_value = np.min(self.data)  # The absolute max value across both real and imaginary parts
        # # self.max_value = np.max(np.abs(self.data))  # The absolute max value across both real and imaginary parts
        # # self.min_value = np.min(np.abs(self.data))  # The absolute max value across both real and imaginary parts
        # print("###############################")
        # print("max_value", self.max_value)
        # print("min_value", self.min_value)
        # print("###############################")
        # # Get absolute values (magnitudes)
        # # Separate real and imaginary parts and get their absolute values
        # components = [
        #     abs(self.max_value.real),
        #     abs(self.max_value.imag),
        #     abs(self.min_value.real),
        #     abs(self.min_value.imag),
        # ]


        # # Get the order of magnitude (how many decimal zeros before significant digit)
        # power_of_10 = abs(np.floor(np.log10(components)))
        # self.highest_power_in_dataset = int(abs(max(power_of_10)))
        # # print("power of component:", power_of_10)
        # # print("Max power:", abs(max(power_of_10)))
        # print("sample: ", 10**self.highest_power_in_dataset)
        # self.normalizing_factor = 10**self.highest_power_in_dataset
        # ----------------------------------------------------------------------------
        # self.real_min = np.min(self.data.real)
        # self.real_max = np.max(self.data.real)
        # self.imag_min = np.min(self.data.imag)
        # self.imag_max = np.max(self.data.imag)

        print("###############################")
        print(self.data.shape)
        print("###############################")

    def __len__(self):
        return self.num_users * (self.time_length - (self.overlapping_index + 1))

    def __getitem__(self, idx):
        sample_idx = idx // (self.time_length - (self.overlapping_index + 1))
        time_idx = idx % (self.time_length - (self.overlapping_index + 1))

        real_input = self.data.real[sample_idx, :, :, :, time_idx:time_idx+self.overlapping_index]
        imag_input = self.data.imag[sample_idx, :, :, :, time_idx:time_idx+self.overlapping_index]
        
        real_output = self.data.real[sample_idx, :, :, :, time_idx+self.overlapping_index]
        imag_output = self.data.imag[sample_idx, :, :, :, time_idx+self.overlapping_index]

        # Normalize the input and output values using the max values
        # real_input = real_input 
        # imag_input = imag_input 
        # real_output = real_output 
        # imag_output = imag_output 
        # -----------------------------
        # print("before normalization real_input", real_input[0])
        # print("before normalization imag_input", imag_input[0])
        # print("before normalization real_output", real_output[0])
        # print("before normalization imag_output", imag_output[0])
        # real_input = real_input / self.max_value
        # imag_input = imag_input / self.max_value
        # real_output = real_output / self.max_value
        # imag_output = imag_output / self.max_value
        # Normalize to [0, 1]
        # real_input = ((real_input - self.real_min) / (self.real_max - self.real_min)) 
        # imag_input = ((imag_input - self.imag_min) / (self.imag_max - self.imag_min)) 
        # real_output = ((real_output - self.real_min) / (self.real_max - self.real_min)) 
        # imag_output = ((imag_output - self.imag_min) / (self.imag_max - self.imag_min)) 
        
        
        # real_input = real_input * self.normalizing_factor   
        # imag_input = imag_input * self.normalizing_factor   
        # real_output =  real_output * self.normalizing_factor
        # imag_output =  imag_output * self.normalizing_factor


        # print("after normalization real_input", real_input[0])
        # print("after normalization imag_input", imag_input[0])
        # print("after normalization real_output", real_output[0])
        # print("after normalization imag_output", imag_output[0])

        real_input = torch.tensor(real_input, dtype=torch.float32, device=self.device)
        imag_input = torch.tensor(imag_input, dtype=torch.float32, device=self.device)
        real_output = torch.tensor(real_output, dtype=torch.float32, device=self.device)
        imag_output = torch.tensor(imag_output, dtype=torch.float32, device=self.device)

        input_data = torch.cat([real_input, imag_input], dim=0)
        output_data = torch.cat([real_output, imag_output], dim=0)
        # Get the second last dimension
        # print("------------------------------------------")
        # print("input_data shape", output_data.shape)
        # print("------------------------------------------")
        
        # for input
        current_size = input_data.shape[2]
        if current_size < 16:
            pad_amount = 16 - current_size
            # Padding format: (last_dim_left, last_dim_right, ..., dim3_left, dim3_right)
            # Pad dim=3 (the 4th dimension), on the right
            pad = [0, 0, 0, pad_amount]  # pad dim=3 only
            pad = [0, 0] * (input_data.dim() - 4) + pad
            input_data = F.pad(input_data, pad)
        
        
        # for output
        if output_data.shape[-1] < 16:
            pad_amount = 16 - output_data.shape[-1]
            # Padding format for 3D: (last_dim_left, last_dim_right)
            output_data = F.pad(output_data, (0, pad_amount))  # pad only on the right side of last dimension

                
        return input_data, output_data

    def get_max_values(self):
        return {
            'overall_max': self.max_value,
        }