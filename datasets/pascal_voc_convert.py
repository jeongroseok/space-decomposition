import torchvision


def main():
    # load dataset
    dataset = torchvision.datasets.VOCDetection(
        "data/voc2012")  # , download=True

    # separate images
    dataset_dict = {0: [], 1: [], 2: [], 3: []}
    for input, label in dataset:
        people = sum([
            1 if obj['name'] == "person" else 0
            for obj in label['annotation']['object']
        ])
        if people >= 3:
            dataset_dict[3].append(input)
        else:
            dataset_dict[people].append(input)

    # stat
    for key, value in dataset_dict.items():
        print("Loaded {} images under {}".format(len(value), key))

    # save
    for key, value in dataset_dict.items():
        for idx, img in enumerate(value):
            img.save(f'data/rough_population/{key}/{idx}.jpg')


if __name__ == "__main__":
    main()