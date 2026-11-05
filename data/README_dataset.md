# Dataset layout

The raw image dataset is not redistributed in this anonymous repository.

Use the public Kaggle **Knee Osteoarthritis Dataset with Severity**. The scripts expect a directory containing the official train/validation/test folders. Common accepted split folder names include:

```text
train/
val/ or valid/ or validation/
test/
```

Each split should contain class subfolders for KL grades 0--4.

Example Kaggle location:

```text
/kaggle/input/knee-osteoarthritis-dataset-with-severity/
```

If your Kaggle dataset mount uses a different name, pass the corresponding path through the notebook or script configuration.
