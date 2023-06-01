from .base_dataset import BaseDataset


class MMDialCaptionDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["mmdial_context_train"]
        elif split == "val":
            names = ["mmdial_context_dev"]
        elif split == "test":
            names = ["mmdial_context_test"]

        super().__init__(*args, **kwargs, names=names, text_column_name="caption")

    def load_evalset(self):
        all_image = self.table["image"]
        all_name = self.table["image_id"]
        assert self.table["split"][0].as_py() in ["test", "val"]
        print(self.table["split"][0].as_py())
        assert len(all_image) == len(all_name)
        self.image_mapper = dict()
        for i in range(len(all_name)):
            image_name = all_name[i].as_py().split('.')[0]
            cur_image = all_image[i].as_py()
            self.image_mapper[image_name] = cur_image
        return self.image_mapper

    def __getitem__(self, index):
        suite = self.get_suite(index)

        if  self.split in ["test", "val"] :
            _index, _question_index = self.index_mapper[index]
            iid = self.table["image_id"][_index].as_py()
            iid = str(iid.split(".")[0].split("_")[-1])
            suite.update({"iid": iid})
            negs_images = self.table["neg_images"][_index].as_py()
            suite.update({"negs_imgs": negs_images})

        return suite
