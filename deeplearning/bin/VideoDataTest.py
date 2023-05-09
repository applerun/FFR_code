from deeplearning.utils.VideoData import PadCollate,FFRDataset
import torchvision.transforms as transforms

if __name__ == '__main__':
    dataroot = "../../../FFR_data/CAG_raw_jpg"
    train_transformer = transforms.Compose([

        # transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomAffine(degrees = 0, translate = (0.1, 0.1)),
        transforms.ToTensor(),
    ])
    dataset = FFRDataset(dataroot, transform = train_transformer)
    data = dataset[1]
    label2data = dataset.get_data_sorted_by_label()
    print(data[0].max())
