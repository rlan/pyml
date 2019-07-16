import pyml.torch.dataset as ds


def test_load_mnist():
  train_loader, test_loader = ds.load_mnist(batch_size=100, test_batch_size=1000)

def test_load_mnist_with_validation():
  train_loader, val_loader, test_loader = ds.load_mnist_with_validation(batch_size=100, test_batch_size=1000, validation_size=50000)

def test_load_cifar10():
  train_loader, test_loader = ds.load_cifar10(batch_size=100, test_batch_size=1000)
  train_loader, test_loader = ds.load_cifar10(batch_size=100, test_batch_size=1000, use_data_augmentation=True)

def test_load_cifar10_with_validation():
  train_loader, val_loader, test_loader = ds.load_cifar10_with_validation(batch_size=100, test_batch_size=1000, validation_size=5000)
  train_loader, val_loader, test_loader = ds.load_cifar10_with_validation(batch_size=100, test_batch_size=1000, validation_size=5000, use_data_augmentation=True)

def test_load_cifar10_with_preselected_validation():
  ds.generate_train_val_indices(50000, 5000, 'cifar10.json')
  ds.load_cifar10_with_preselected_validation(batch_size=100, test_batch_size=1000, index_json_file='cifar10.json', use_data_augmentation=True)

if __name__ == "__main__":
  test_load_mnist()
  test_load_mnist_with_validation()
  test_load_cifar10()
  test_load_cifar10_with_validation()
  test_load_cifar10_with_preselected_validation()
