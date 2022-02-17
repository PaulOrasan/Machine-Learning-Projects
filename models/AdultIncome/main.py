from AdultDataset import AdultDataset

train_dataset = AdultDataset('../../datasets/adult.data')
train_dataset.showDataTypes()
train_dataset.showDescription()

train_dataset.categoricalPlot('race')
train_dataset.categoricalPlot('sex')
train_dataset.categoricalPlot('education')
train_dataset.categoricalPlot('native-country')
train_dataset.showDataTypes()
train_dataset.showPlots()