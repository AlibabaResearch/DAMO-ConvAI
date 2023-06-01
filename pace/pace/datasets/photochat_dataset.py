from .base_dataset import BaseDataset


class PhotochatDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["photochat_context_train"]  
        elif split == "val":
            names = ["photochat_context_dev"]
        elif split == "test":
            names = ["photochat_context_test"]

        super().__init__(*args, **kwargs, names=names, text_column_name="caption")

    def __getitem__(self, index):
        suite = self.get_suite(index)

        if "test" in self.split:
            _index, _question_index = self.index_mapper[index]
            iid = self.table["image_id"][_index].as_py()
            iid = str(iid.split(".")[0].split("_")[-1])
            suite.update({"iid": iid})

        return suite
