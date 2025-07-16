# How to install datasets

We suggest putting all datasets under the same folder (say `$DATA`) to ease management and following the instructions below to organize datasets to avoid modifying the source code. The file structure looks like

```
$DATA/
|–– caltech-101/
|–– oxford_pets/
|–– stanford_cars/
...
```

If you have some datasets already installed somewhere else, you can create symbolic links in `$DATA/dataset_name` that point to the original data to avoid duplicate download.

Datasets list:
- [ImageNet](#imagenet)
- [Caltech101](#caltech101)
- [OxfordPets](#oxfordpets)
- [StanfordCars](#stanfordcars)
- [Flowers102](#flowers102)
- [Food101](#food101)
- [FGVCAircraft](#fgvcaircraft)
- [SUN397](#sun397)
- [DTD](#dtd)
- [EuroSAT](#eurosat)
- [UCF101](#ucf101)
- [StanfordDogs](#stanforddogs)
- [CUB](#cub)
- [PLANTDOC](#plantdoc)
- [DogsImnet](#dogs_imagenet)
- [BirdsImnet](#birds_imagenet)
- [VehiclesImnet](#vehicles_imagenet)


The instructions to prepare each dataset are detailed below. To ensure reproducibility and fair comparison for future work, we provide fixed train/val/test splits for all datasets except ImageNet where the validation set is used as test set. The fixed splits are either from the original datasets (if available) or created by us.

### ImageNet
- Create a folder named `imagenet/` under `$DATA`.
- Create `images/` under `imagenet/`.
- Download the dataset from the [official website](https://image-net.org/index.php) and extract the training and validation sets to `$DATA/imagenet/images`. The directory structure should look like
```
imagenet/
|–– images/
|   |–– train/ # contains 1,000 folders like n01440764, n01443537, etc.
|   |–– val/
```
- If you had downloaded the ImageNet dataset before, you can create symbolic links to map the training and validation sets to `$DATA/imagenet/images`.
- Download the `classnames.txt` to `$DATA/imagenet/` from this [link](https://drive.google.com/file/d/1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF/view?usp=sharing). The class names are copied from [CLIP](https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb).

### Caltech101
- Create a folder named `caltech-101/` under `$DATA`.
- Download `101_ObjectCategories.tar.gz` from http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz and extract the file under `$DATA/caltech-101`.
- Download `split_zhou_Caltech101.json` from this [link](https://drive.google.com/file/d/1hyarUivQE36mY6jSomru6Fjd-JzwcCzN/view?usp=sharing) and put it under `$DATA/caltech-101`. 

The directory structure should look like
```
caltech-101/
|–– 101_ObjectCategories/
|–– split_zhou_Caltech101.json
```

### OxfordPets
- Create a folder named `oxford_pets/` under `$DATA`.
- Download the images from https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz.
- Download the annotations from https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz.
- Download `split_zhou_OxfordPets.json` from this [link](https://drive.google.com/file/d/1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs/view?usp=sharing). 

The directory structure should look like
```
oxford_pets/
|–– images/
|–– annotations/
|–– split_zhou_OxfordPets.json
```

### StanfordCars
- Create a folder named `stanford_cars/` under `$DATA`.
- Download the train images http://ai.stanford.edu/~jkrause/car196/cars_train.tgz.
- Download the test images http://ai.stanford.edu/~jkrause/car196/cars_test.tgz.
- Download the train labels https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz.
- Download the test labels http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat.
- Download `split_zhou_StanfordCars.json` from this [link](https://drive.google.com/file/d/1ObCFbaAgVu0I-k_Au-gIUcefirdAuizT/view?usp=sharing).

The directory structure should look like
```
stanford_cars/
|–– cars_test\
|–– cars_test_annos_withlabels.mat
|–– cars_train\
|–– devkit\
|–– split_zhou_StanfordCars.json
```

### Flowers102
- Create a folder named `oxford_flowers/` under `$DATA`.
- Download the images and labels from https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz and https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat respectively.
- Download `cat_to_name.json` from [here](https://drive.google.com/file/d/1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0/view?usp=sharing). 
- Download `split_zhou_OxfordFlowers.json` from [here](https://drive.google.com/file/d/1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT/view?usp=sharing).

The directory structure should look like
```
oxford_flowers/
|–– cat_to_name.json
|–– imagelabels.mat
|–– jpg/
|–– split_zhou_OxfordFlowers.json
```

### Food101
- Download the dataset from https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/ and extract the file `food-101.tar.gz` under `$DATA`, resulting in a folder named `$DATA/food-101/`.
- Download `split_zhou_Food101.json` from [here](https://drive.google.com/file/d/1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl/view?usp=sharing).

The directory structure should look like
```
food-101/
|–– images/
|–– license_agreement.txt
|–– meta/
|–– README.txt
|–– split_zhou_Food101.json
```

### FGVCAircraft
- Download the data from https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz.
- Extract `fgvc-aircraft-2013b.tar.gz` and keep only `data/`.
- Move `data/` to `$DATA` and rename the folder to `fgvc_aircraft/`.

The directory structure should look like
```
fgvc_aircraft/
|–– images/
|–– ... # a bunch of .txt files
```

### SUN397
- Create a folder named  `sun397/` under `$DATA`.
- Download the images http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz.
- Download the partitions https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip.
- Extract these files under `$DATA/sun397/`.
- Download `split_zhou_SUN397.json` from this [link](https://drive.google.com/file/d/1y2RD81BYuiyvebdN-JymPfyWYcd8_MUq/view?usp=sharing).

The directory structure should look like
```
sun397/
|–– SUN397/
|–– split_zhou_SUN397.json
|–– ... # a bunch of .txt files
```

### DTD
- Download the dataset from https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz and extract it to `$DATA`. This should lead to `$DATA/dtd/`.
- Download `split_zhou_DescribableTextures.json` from this [link](https://drive.google.com/file/d/1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x/view?usp=sharing).

The directory structure should look like
```
dtd/
|–– images/
|–– imdb/
|–– labels/
|–– split_zhou_DescribableTextures.json
```

### EuroSAT
- Create a folder named `eurosat/` under `$DATA`.
- Download the dataset from http://madm.dfki.de/files/sentinel/EuroSAT.zip and extract it to `$DATA/eurosat/`.
- Download `split_zhou_EuroSAT.json` from [here](https://drive.google.com/file/d/1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o/view?usp=sharing).

The directory structure should look like
```
eurosat/
|–– 2750/
|–– split_zhou_EuroSAT.json
```

### UCF101
- Create a folder named `ucf101/` under `$DATA`.
- Download the zip file `UCF-101-midframes.zip` from [here](https://drive.google.com/file/d/10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O/view?usp=sharing) and extract it to `$DATA/ucf101/`. This zip file contains the extracted middle video frames.
- Download `split_zhou_UCF101.json` from this [link](https://drive.google.com/file/d/1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y/view?usp=sharing).

The directory structure should look like
```
ucf101/
|–– UCF-101-midframes/
|–– split_zhou_UCF101.json
```


### StanfordDogs
- Create a folder named `stanford_dogs/` under `$DATA`.
- Download the zip files `images.tar` and `annotation.tar` from [here](wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar) and [here](wget http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar) and extract it to `$DATA/stanford_dogs/`. 
- Move `stanford_dogs/split_alexey_stanford_dogs.json` into `$DATA/stanford_dogs` from `assets` folder in this repo. 

The directory structure should look like
```
stanford_dogs/
|–– Annotation/
|–– Images/
|–– split_alexey_stanford_dogs.json
```

### CUB
- Create a folder named `cub/` under `$DATA`.
- Download the zip file `CUB_200_2011.tgz` from [here](wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1) and extract it to `$DATA/cub/`. 
- Move `cub/split_alexey_cub.json`  into `$DATA/cub` from `assets` folder in this repo. 

The directory structure should look like
```
stanford_dogs/
|–– images/
|–– attributes/
|–– parts/
|–– split_alexey_cub.json
```

### PLANTDOC
- Create a folder named `plantdoc/` under `$DATA`.
- Download the images from [here](git clone https://github.com/pratikkayal/PlantDoc-Dataset.git) and place them into `$DATA/plantdoc/`. 
- Move `plantdoc/split_alexey_plantdoc.json`  into `$DATA/plantdoc` from `assets` folder in this repo. 

The directory structure should look like
```
stanford_dogs/
|–– all_images/
|–– split_alexey_plantdoc.json
```


### DogsImnet
- Create a folder named `dogs_imagenet/` under `$DATA`.
- Move images from ImageNet folder according to `assets/split_alexey_dogs_imagenet.json` and place them into `$DATA/dogs_imagenet/`. 
- Move `dogs_imagenet/split_alexey_dogs_imagenet.json` and `dogs_imagenet/classes.txt`  into `$DATA/dogs_imagenet` from `assets` folder in this repo. 

The directory structure should look like
```
dogs_imagenet/
|–– images/
|–– split_alexey_dogs_imagenet.json
|-- classes.txt
```


### BirdsImnet
- Create a folder named `birds_imagenet/` under `$DATA`.
- Move images from ImageNet folder according to `assets/split_alexey_birds_imagenet.json` and place them into `$DATA/birds_imagenet/`. 
- Move `birds_imagenet/split_alexey_birds_imagenet.json` and `birds_imagenet/classes.txt` into `$DATA/birds_imagenet` from `assets` folder in this repo. 

The directory structure should look like
```
birds_imagenet/
|–– images/
|–– split_alexey_birds_imagenet.json
|-- classes.txt
```


### VehiclesImnet
- Create a folder named `vehicles_imagenet/` under `$DATA`.
- Move images from ImageNet folder according to `assets/split_alexey_vehicles_imagenet.json` and place them into `$DATA/vehicles_imagenet/`. 
- Move `vehicles_imagenet/split_alexey_vehicles_imagenet.json` and `vehicles_imagenet/classes.txt` into `$DATA/vehicles_imagenet` from `assets` folder in this repo. 

The directory structure should look like
```
vehicles_imagenet/
|–– images/
|–– split_alexey_vehicles_imagenet.json
|-- classes.txt
```


