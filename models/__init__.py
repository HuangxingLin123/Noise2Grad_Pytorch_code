import importlib
from models.base_model import BaseModel


def find_model_using_name():
    # Given the option --model [modelname],
    # the file "models/modelname_model.py"
    # will be imported.
    # model_filename = "models." + model_name + "_model"
    model_filename = "models." +  "denoise_model"
    modellib = importlib.import_module(model_filename)

    # In the file, the class called ModelNameModel() will
    # be instantiated. It has to be a subclass of BaseModel,
    # and it is case-insensitive.
    model = None
    target_model_name = 'denoisemodel'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model


def get_option_setter():
    model_class = find_model_using_name()
    return model_class.modify_commandline_options


def create_model(opt):
    model = find_model_using_name()
    instance = model()
    instance.initialize(opt)
    print("model [%s] was created" % (instance.name()))
    return instance
