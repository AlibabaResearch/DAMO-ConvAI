#
# class TrainDataset(Dataset):
#
#     def __init__(self, args, raw_datasets):
#         # This tab processor is for table truncation and linearize.
#         self.raw_datasets = raw_datasets
#
#     def __getitem__(self, index) -> T_co:
#         raw_data = self.raw_datasets[index]
#
#         return raw_data.update({"struct_in": struct_in, "text_in": text_in, "seq_out": seq_out})
#
#
# class DevDataset(Dataset):
#
#     def __init__(self, args, raw_datasets):
#         # This tab processor is for table truncation and linearize.
#         self.raw_datasets = raw_datasets
#
#     def __getitem__(self, index):
#         raw_data = self.raw_datasets[index]
#
#         return raw_data.update({"struct_in": struct_in, "text_in": text_in, "seq_out": seq_out})
#
#
# class TestDataset(Dataset):
#
#     def __init__(self, args, raw_datasets):
#         # This tab processor is for table truncation and linearize.
#         self.raw_datasets = raw_datasets
#
#     def __getitem__(self, index):
#         raw_data = self.raw_datasets[index]
#
#         return raw_data.update({"struct_in": struct_in, "text_in": text_in, "seq_out": seq_out})
