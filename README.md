# AE22 feature extraction package
Package extracts features from time series using autoencoder trained on UAE & UCR datasets. 
Autoencoder was trained to extract 22 features out of 88 length time series. When time series are larger than 88, multiple windows are created and features are extracted out of them.

Example:

```
ae = AE22()
example = np.zeros((2, 180))
example_results = ae.transform(example)

```