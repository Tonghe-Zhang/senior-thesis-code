How to use hydra to manage super complex hyperparameter and configuration file systems:

![image-20241111200034672](./notes.assets/image-20241111200034672.png)

Here, in `test.yaml`, 

```python
_target_: model.my_model.MyModel

model:
  constant_1: 114514
```

In `test2.yaml`

```python
_target_: model.my_model.MyModel

model:
  constant_1: 1919810
```

